# main.py COMPLETO - TradeLog Analyzer API v3
# Batch Analysis Robusta com Geração Dinâmica de Setups

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
from itertools import combinations

app = FastAPI(title="TradeLog Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeAnalyzer:
    def __init__(self, tick_value=0.50, tick_size=0.25):
        self.tick_value = tick_value
        self.tick_size = tick_size

    def parse_json_file(self, content: str) -> List[dict]:
        content = re.sub(r'(?<=[\d])(,)(?=[\d])', '.', content)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            pass

        lines = content.strip().split('\n\n')
        data_list = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, list):
                        data_list.extend(parsed)
                    else:
                        data_list.append(parsed)
                except:
                    continue

        if not data_list:
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except:
                        continue
        return data_list

    def convert_to_dataframe(self, data_list: List[dict]) -> pd.DataFrame:
        records = []
        for item in data_list:
            try:
                timestamp_str = item.get('timestamp_barra') or item.get('timestamp')
                if not timestamp_str:
                    continue

                barra = item.get('barra', {})
                indicadores = item.get('indicadores', {})

                record = {
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(barra.get('open', 0)),
                    'high': float(barra.get('high', 0)),
                    'low': float(barra.get('low', 0)),
                    'close': float(barra.get('close', 0)),
                    'volume': int(barra.get('volume', 0)),
                    'direcao': barra.get('direcao', ''),
                }

                # Extrair indicadores básicos para o DataFrame
                if 'rsi' in indicadores:
                    record['rsi'] = float(indicadores['rsi'].get('valor', 50))
                else:
                    record['rsi'] = 50

                if 'ema' in indicadores:
                    record['ema'] = float(indicadores['ema'].get('valor', 0))
                    record['ema_dist_ticks'] = int(indicadores['ema'].get('distancia_close_ticks', 0))
                else:
                    record['ema'] = 0
                    record['ema_dist_ticks'] = 0
                
                # Para close_vs_ema: se ema_dist_ticks > 0 (preço acima), < 0 (abaixo)
                # Vamos simplificar: se ema_dist_ticks > 0 -> 1, se < 0 -> -1
                record['close_vs_ema'] = 1 if record['ema_dist_ticks'] > 0 else -1

                if 'fibonacci_pivots' in indicadores:
                    fp = indicadores['fibonacci_pivots']
                    record['pivot_zona'] = fp.get('zona', '')
                else:
                    record['pivot_zona'] = ''

                records.append(record)
            except Exception:
                continue

        if not records:
             # Retorna DataFrame vazio se falhar
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def backtest(self, df: pd.DataFrame, conditions: List[dict], stop_ticks: int, target_ticks: int) -> List[dict]:
        """
        Testa um setup específico (lista de condições) no DataFrame.
        Retorna lista de trades.
        """
        # 1. Aplicar filtros dinâmicos
        # Começa com tudo True
        mask = pd.Series([True] * len(df))

        for cond in conditions:
            campo = cond['campo']
            op = cond['operador']
            val = cond['valor']

            if campo not in df.columns:
                # Se campo não existe (ex: volume não parseado direito), falha seguro
                mask = mask & False
                continue

            if op == '<':
                mask = mask & (df[campo] < val)
            elif op == '>':
                mask = mask & (df[campo] > val)
            elif op == '==':
                mask = mask & (df[campo] == val)
            elif op == '>=':
                mask = mask & (df[campo] >= val)
            elif op == '<=':
                mask = mask & (df[campo] <= val)

        # Índices onde todas as condições são atendidas
        entry_indices = df[mask].index.tolist()
        
        # Se não houver sinais, retorna vazio
        if not entry_indices:
            return []

        # 2. Executar trades (Uma entrada por vez)
        trades = []
        max_bars = 20
        next_allowed_bar = 0

        # Para simplificar direção, vamos assumir direção baseada no 'direcao' do candle
        # ou se o setup for apenas 'rsi < 30', assumimos LONG?
        # A lógica do usuário pede para gerar setups. Se o setup tem "direcao == ALTA", é LONG.
        # Se tem "direcao == BAIXA", é SHORT.
        # Se não tiver direção explícita, idealmente ignoramos ou testamos ambos.
        # Pelo gerador de setups, vamos incluir sempre uma condição de direção.

        # Identificar direção do setup pelos filters
        direction = 'LONG' # Default
        for cond in conditions:
            if cond['campo'] == 'direcao':
                if cond['valor'].upper() == 'BAIXA':
                    direction = 'SHORT'
                break
        
        for idx in entry_indices:
            if idx < next_allowed_bar:
                continue
            if idx + max_bars >= len(df):
                continue
            
            entry_price = df.iloc[idx]['close']
            
            # Simulação Simples
            profit = 0
            bars = max_bars
            
            # Lógica de simulação (cópia simplificada do simulate_trade anterior)
            if direction == 'LONG':
                stop_price = entry_price - (stop_ticks * self.tick_size)
                target_price = entry_price + (target_ticks * self.tick_size)
                result_status = 'OPEN'
                
                for i in range(1, max_bars + 1):
                    future = df.iloc[idx + i]
                    if future['low'] <= stop_price:
                        profit = -stop_ticks
                        bars = i
                        result_status = 'STOP'
                        break
                    elif future['high'] >= target_price:
                        profit = target_ticks
                        bars = i
                        result_status = 'TARGET'
                        break
            else: # SHORT
                stop_price = entry_price + (stop_ticks * self.tick_size)
                target_price = entry_price - (target_ticks * self.tick_size)
                result_status = 'OPEN'

                for i in range(1, max_bars + 1):
                    future = df.iloc[idx + i]
                    if future['high'] >= stop_price:
                        profit = -stop_ticks
                        bars = i
                        result_status = 'STOP'
                        break
                    elif future['low'] <= target_price:
                        profit = target_ticks
                        bars = i
                        result_status = 'TARGET'
                        break
            
            if result_status != 'OPEN':
                trades.append({
                    'profit': profit,
                    'result': result_status,
                    'bars': bars
                })
                # Pula barras até trade fechar
                next_allowed_bar = idx + bars + 1

        return trades


analyzer = TradeAnalyzer()


def generate_all_setups():
    """Gera todas as combinações de condições de entrada."""
    setups = []
    
    # Variações de RSI
    rsi_conditions = [
        {"campo": "rsi", "operador": "<", "valor": 30},
        {"campo": "rsi", "operador": "<", "valor": 40},
        {"campo": "rsi", "operador": ">", "valor": 60},
        {"campo": "rsi", "operador": ">", "valor": 70},
    ]
    
    # Variações de EMA (preço vs EMA)
    # close_vs_ema: 1 (acima), -1 (abaixo)
    ema_conditions = [
        {"campo": "close_vs_ema", "operador": "==", "valor": 1},  # acima
        {"campo": "close_vs_ema", "operador": "==", "valor": -1}, # abaixo
    ]
    
    # Direção da barra
    dir_conditions = [
        {"campo": "direcao", "operador": "==", "valor": "ALTA"},
        {"campo": "direcao", "operador": "==", "valor": "BAIXA"},
    ]
    
    # Zona pivot (Simplificado)
    # Se o log traz pivot_zona como string "ENTRE_S1_PP", "ACIMA_R1", etc.
    # Vamos usar genericamente alguns valores comuns se existirem nos dados.
    # Para ser robusto, podemos ignorar pivot se não soubermos os valores exatos,
    # OU assumir que o "parse_json_file" traz valores consistentes.
    pivot_conditions = [
        # Exemplo: testar zonas se existirem. Se não, ficarão vazios.
        # Vamos omitir para garantir que não quebre se o dado não tiver pivots.
        # Ou adicionar genericamente se quiser.
    ]
    
    # Gerar combinações (2 a 3 condições por setup)
    # Importante: Um setup deve ter uma DIREÇÃO para sabermos se é Long ou Short.
    # Então vamos iterar sobre dir_conditions e combinar com outros.
    
    base_filters = rsi_conditions + ema_conditions
    
    idx = 0
    # setups com 1 indicador + direção
    for direct in dir_conditions:
        for f1 in base_filters:
            idx += 1
            # Evitar contradições óbvias (ex: RSI > 70 e Direção ALTA é ok, mas RSI > 70 costuma ser venda em reversão.
            # O gerador é "burro", testa tudo. O filtro de WinRate elimina o ruim).
            
            combo = [direct, f1]
            name = f"Setup {idx}: {direct['valor']} + {f1['campo']}{f1['operador']}{f1['valor']}"
            setups.append({
                "name": name,
                "conditions": combo
            })
            
            # setups com 2 indicadores + direção
            for f2 in base_filters:
                if f1 == f2: continue
                # Evitar conflitos de mesmo campo (ex: rsi < 30 e rsi > 70)
                if f1['campo'] == f2['campo']: continue 
                
                idx += 1
                combo2 = [direct, f1, f2]
                name2 = f"Setup {idx}: {direct['valor']} + {f1['campo']} + {f2['campo']}"
                setups.append({
                    "name": name2,
                    "conditions": combo2
                })
    
    return setups


@app.post("/api/v1/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    config: str = Form('{}')
):
    try:
        try:
            cfg = json.loads(config)
        except:
            cfg = {}
            
        min_consistency = cfg.get('minConsistency', 0.9)
        min_wr = cfg.get('minWinRate', 60)
        
        # 1. Parsear todos os arquivos e converter para DataFrames EM MEMÓRIA
        all_days_data = {} # filename -> DataFrame
        
        for f in files:
            content = await f.read()
            filename = f.filename
            content_str = content.decode('utf-8')
            entries = analyzer.parse_json_file(content_str)
            if entries:
                df = analyzer.convert_to_dataframe(entries)
                if not df.empty:
                    all_days_data[filename] = df

        if not all_days_data:
            return {
                "status": "completed", 
                "total_days": 0,
                "total_setups_tested": 0, 
                "consistent_setups": []
            }

        # 2. Gerar setups dinâmicos
        setups = generate_all_setups()
        
        # 3. Testar CADA setup em CADA dia com MÚLTIPLOS stops/targets
        possible_stops = [10, 15, 20]
        possible_targets = [20, 30, 40]
        
        # Dicionário para agregar resultados: Key -> {setup, stop, target, daily_stats: []}
        aggregated_results = {}
        
        # Para otimizar, iteramos setups e dias
        for setup in setups:
            for stop in possible_stops:
                for target in possible_targets:
                    # Chave única do teste
                    test_key = f"{setup['name']}|{stop}|{target}"
                    
                    daily_stats = []
                    
                    for day_name, df in all_days_data.items():
                        trades = analyzer.backtest(df, setup['conditions'], stop, target)
                        
                        count = len(trades)
                        profit = sum(t['profit'] for t in trades)
                        profit_usd = profit * analyzer.tick_value
                        wins = sum(1 for t in trades if t['result'] == 'TARGET')
                        wr = (wins / count * 100) if count > 0 else 0
                        
                        daily_stats.append({
                            "day": day_name,
                            "total_trades": count,
                            "profit": profit_usd,
                            "win_rate": wr,
                            "profitable": profit_usd > 0
                        })
                    
                    aggregated_results[test_key] = {
                        "setup": setup,
                        "stop": stop,
                        "target": target,
                        "daily_stats": daily_stats
                    }

        # 4. Filtrar consistência
        consistent = []
        
        for key, data in aggregated_results.items():
            stats = data['daily_stats']
            
            # Filtra dias que tiveram trades
            active_days = [d for d in stats if d['total_trades'] > 0]
            
            # Se operou em poucos dias, talvez não seja consistente o suficiente
            # (Critério arbitrário: tem que ter operado em pelo menos 1 dia? 
            #  Ou consistência conta dias sem trade como falha? 
            #  Vamos assumir: dias sem trade NÃO contam para a média, 
            #  mas precisamos de um mínimo de dias operados para validar, ex: 10% dos arquivos)
            if not active_days:
                continue
                
            num_active = len(active_days)
            num_profitable = sum(1 for d in active_days if d['profitable'])
            
            # Consistência = % de dias positivos DENTRE os dias operados
            consistency = num_profitable / num_active
            
            if consistency >= min_consistency:
                # Checar WinRate Médio
                avg_wr = sum(d['win_rate'] for d in active_days) / num_active
                if avg_wr >= min_wr:
                    total_profit = sum(d['profit'] for d in active_days)
                    
                    consistent.append({
                        "setup_name": data['setup']['name'],
                        "stop_ticks": data['stop'],
                        "target_ticks": data['target'],
                        "days_tested": len(all_days_data),
                        "days_with_trades": num_active,
                        "days_profitable": num_profitable,
                        "consistency": round(consistency, 2),
                        "avg_win_rate": round(avg_wr, 1),
                        "total_profit_usd": round(total_profit, 2),
                        "avg_daily_profit_usd": round(total_profit / num_active, 2),
                        "daily_results": active_days,
                        # Passar regras de forma estruturada se possível
                        "rules": {"conditions": data['setup']['conditions']} 
                    })

        # Top 50 por lucro total
        consistent.sort(key=lambda x: x['total_profit_usd'], reverse=True)
        
        return {
            "status": "completed",
            "total_days": len(all_days_data),
            "total_setups_tested": len(aggregated_results),
            "consistent_setups": consistent[:50]
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "detail": traceback.format_exc()}


@app.get("/")
def home():
    return {"message": "TradeLog Analyzer API v3", "status": "online"}

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Manter endpoint antigo para compatibilidade se necessário, ou redirecionar.
# Por enquanto, deixamos ele lá em cima se não foi removido, 
# mas aqui estamos substituindo o arquivo todo.
# Vamos reimplementar o /analyze individual simples para não quebrar o frontend antigo se houver.
@app.post("/api/v1/analyze")
async def analyze_file(file: UploadFile = File(...), config: str = Form('{}')):
    # Versão simplificada que usa a nova engine, ou retorna erro dizendo use batch
    return {"status": "deprecated", "message": "Use /api/v1/analyze-batch para análise completa."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)