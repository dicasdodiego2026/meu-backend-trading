# main.py COMPLETO E CORRIGIDO - TradeLog Analyzer API
# Compatível com formato NinjaTrader (timestamp_barra)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import io

app = FastAPI(title="TradeLog Analyzer API")

# CORS para permitir frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeAnalyzer:
    def __init__(self, tick_value=0.25, contract_value=20):
        self.tick_value = tick_value
        self.contract_value = contract_value
    
    def parse_json_file(self, content: str) -> List[dict]:
        """Parseia arquivo JSON do NinjaTrader"""
        # Corrigir vírgula decimal (formato BR -> US)
        content = re.sub(r'(?<=[\d])(,)(?=[\d])', '.', content)
        
        # Dividir múltiplos objetos JSON
        lines = content.strip().split('\n\n')
        data_list = []
        
        for line in lines:
            line = line.strip()
            if line and line != '':
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Erro ao parsear linha: {e}")
                    continue
        
        print(f"Total de objetos JSON parseados: {len(data_list)}")
        return data_list
    
    def convert_to_dataframe(self, data_list: List[dict]) -> pd.DataFrame:
        """Converte lista de dados em DataFrame"""
        records = []
        
        for i, item in enumerate(data_list):
            try:
                # CORREÇÃO PRINCIPAL: Buscar 'timestamp_barra' primeiro
                timestamp_str = item.get('timestamp_barra') or item.get('timestamp')
                if not timestamp_str:
                    print(f"Item {i}: timestamp não encontrado")
                    continue
                
                # Parsear barra
                barra = item.get('barra', {})
                indicadores = item.get('indicadores', {})
                
                record = {
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(barra.get('open', 0)),
                    'high': float(barra.get('high', 0)),
                    'low': float(barra.get('low', 0)),
                    'close': float(barra.get('close', 0)),
                    'volume': int(barra.get('volume', 0)),
                }
                
                # Extrair indicadores com fallback seguro
                if 'ema' in indicadores:
                    record['ema'] = float(indicadores['ema'].get('valor', 0))
                    record['ema_dist_ticks'] = int(indicadores['ema'].get('distancia_close_ticks', 0))
                else:
                    record['ema'] = 0
                    record['ema_dist_ticks'] = 0
                
                if 'rsi' in indicadores:
                    record['rsi'] = float(indicadores['rsi'].get('valor', 50))
                else:
                    record['rsi'] = 50
                
                if 'atr' in indicadores:
                    record['atr_ticks'] = int(indicadores['atr'].get('valor_ticks', 10))
                else:
                    record['atr_ticks'] = 10
                
                records.append(record)
                
            except Exception as e:
                print(f"Erro ao processar item {i}: {e}")
                continue
        
        if not records:
            raise ValueError("Nenhum registro válido encontrado no arquivo")
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['hora'] = df['timestamp'].dt.hour
        df['minuto'] = df['timestamp'].dt.minute
        
        print(f"DataFrame criado com {len(df)} linhas")
        print(f"Colunas: {list(df.columns)}")
        print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
        
        return df
    
    def simulate_trade(self, df: pd.DataFrame, idx: int, direction: str, 
                      stop_ticks: int, target_ticks: int, max_bars: int = 20) -> Optional[Dict]:
        """Simula um trade"""
        if idx + max_bars >= len(df):
            return None
            
        row = df.iloc[idx]
        entry = row['close']
        
        if direction == 'LONG':
            stop_price = entry - (stop_ticks * self.tick_value)
            target_price = entry + (target_ticks * self.tick_value)
            
            for i in range(1, max_bars + 1):
                future = df.iloc[idx + i]
                if future['low'] <= stop_price:
                    return {'result': 'STOP', 'profit': -stop_ticks, 'bars': i}
                elif future['high'] >= target_price:
                    return {'result': 'TARGET', 'profit': target_ticks, 'bars': i}
        else:  # SHORT
            stop_price = entry + (stop_ticks * self.tick_value)
            target_price = entry - (target_ticks * self.tick_value)
            
            for i in range(1, max_bars + 1):
                future = df.iloc[idx + i]
                if future['high'] >= stop_price:
                    return {'result': 'STOP', 'profit': -stop_ticks, 'bars': i}
                elif future['low'] <= target_price:
                    return {'result': 'TARGET', 'profit': target_ticks, 'bars': i}
        
        return {'result': 'OPEN', 'profit': 0, 'bars': max_bars}
    
    def analyze(self, df: pd.DataFrame, config: dict) -> Dict:
        """Executa análise completa"""
        results = []
        
        min_win_rate = config.get('minWinRate', 70)
        min_trades = config.get('minTrades', 10)
        
        print(f"Iniciando análise: min_win_rate={min_win_rate}, min_trades={min_trades}")
        
        # Gerar setups
        setups = []
        
        # Setup 1: Abertura USA com variações
        for rsi_long in [40, 45, 50]:
            for rsi_short in [55, 60, 65]:
                for minuto in [30, 35, 40]:
                    cond_long = (
                        (df['hora'] == 9) & 
                        (df['minuto'] >= minuto) & 
                        (df['rsi'] < rsi_long) & 
                        (df['close'] > df['open'])
                    )
                    cond_short = (
                        (df['hora'] == 9) & 
                        (df['minuto'] >= minuto) & 
                        (df['rsi'] > rsi_short) & 
                        (df['close'] < df['open'])
                    )
                    setups.append({
                        'name': f'Abertura_RSI{rsi_long}_{rsi_short}_M{minuto}',
                        'long': cond_long,
                        'short': cond_short
                    })
        
        # Setup 2: RSI Extremo
        setups.append({
            'name': 'RSI_Extremo_25_75',
            'long': (df['rsi'] < 25) & (df['close'] > df['open']),
            'short': (df['rsi'] > 75) & (df['close'] < df['open'])
        })
        
        # Setup 3: EMA Distância
        for dist in [20, 30, 40]:
            setups.append({
                'name': f'EMA_Dist_{dist}',
                'long': (df['ema_dist_ticks'] < -dist) & (df['rsi'] < 40),
                'short': (df['ema_dist_ticks'] > dist) & (df['rsi'] > 60)
            })
        
        # Configurações de stop/target
        configs_test = [
            {'stop': 10, 'target': 20},
            {'stop': 15, 'target': 30},
            {'stop': 20, 'target': 40},
            {'stop': 10, 'target': 30},
            {'stop': 15, 'target': 45},
        ]
        
        total_tests = len(setups) * len(configs_test)
        print(f"Total de combinações a testar: {total_tests}")
        
        for setup_idx, setup in enumerate(setups):
            for cfg in configs_test:
                trades = []
                stop_ticks = cfg['stop']
                target_ticks = cfg['target']
                
                # LONGs
                long_signals = df[setup['long']].index
                for idx in long_signals:
                    if idx < len(df) - 20:
                        result = self.simulate_trade(df, idx, 'LONG', stop_ticks, target_ticks)
                        if result:
                            trades.append({
                                'tipo': 'LONG',
                                'result': result['result'],
                                'profit_ticks': result['profit'],
                                'bars': result['bars']
                            })
                
                # SHORTs
                short_signals = df[setup['short']].index
                for idx in short_signals:
                    if idx < len(df) - 20:
                        result = self.simulate_trade(df, idx, 'SHORT', stop_ticks, target_ticks)
                        if result:
                            trades.append({
                                'tipo': 'SHORT',
                                'result': result['result'],
                                'profit_ticks': result['profit'],
                                'bars': result['bars']
                            })
                
                # Calcular estatísticas
                if len(trades) >= min_trades:
                    closed = [t for t in trades if t['result'] != 'OPEN']
                    if len(closed) > 0:
                        wins = len([t for t in closed if t['result'] == 'TARGET'])
                        win_rate = (wins / len(closed)) * 100
                        
                        if win_rate >= min_win_rate:
                            total_profit = sum(t['profit_ticks'] for t in closed)
                            gross_profit = sum(t['profit_ticks'] for t in closed if t['profit_ticks'] > 0)
                            gross_loss = abs(sum(t['profit_ticks'] for t in closed if t['profit_ticks'] < 0))
                            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
                            
                            results.append({
                                'setup_name': setup['name'],
                                'stop_ticks': stop_ticks,
                                'target_ticks': target_ticks,
                                'ratio': f"1:{target_ticks//stop_ticks}",
                                'total_trades': len(closed),
                                'wins': wins,
                                'losses': len(closed) - wins,
                                'win_rate': round(win_rate, 1),
                                'profit_factor': round(profit_factor, 2),
                                'net_profit_ticks': total_profit,
                                'net_profit_usd': round(total_profit * self.tick_value * self.contract_value, 2),
                                'avg_profit_per_trade': round((total_profit * self.tick_value * self.contract_value) / len(closed), 2)
                            })
        
        # Ordenar por win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return {
            'total_setups_tested': total_tests,
            'profitable_setups': len(results),
            'best_setups': results[:10],
            'summary': {
                'data_bars': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            }
        }

analyzer = TradeAnalyzer()

@app.get("/")
def home():
    return {"message": "TradeLog API está rodando!", "status": "online"}

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    config: str = Form('{"minWinRate": 70, "minTrades": 10}')
):
    """
    Recebe arquivo JSON do NinjaTrader e retorna análise
    """
    try:
        print(f"Recebendo arquivo: {file.filename}")
        
        # Ler arquivo
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        print(f"Tamanho: {len(content_str)} caracteres")
        
        # Parsear config
        try:
            config_dict = json.loads(config)
        except:
            config_dict = {"minWinRate": 70, "minTrades": 10}
        
        print(f"Config: {config_dict}")
        
        # Processar
        data_list = analyzer.parse_json_file(content_str)
        
        if not data_list:
            return {"status": "error", "message": "Nenhum dado JSON válido encontrado"}
        
        df = analyzer.convert_to_dataframe(data_list)
        
        if len(df) == 0:
            return {"status": "error", "message": "Nenhum dado válido após conversão"}
        
        # Analisar
        results = analyzer.analyze(df, config_dict)
        
        return {
            "status": "completed",
            "filename": file.filename,
            "results": results
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERRO: {str(e)}")
        return {"status": "error", "message": str(e), "detail": error_detail}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)