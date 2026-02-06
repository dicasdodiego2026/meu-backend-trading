from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import List, Dict
import io

app = FastAPI(title="TradeLog Analyzer API")

# Permitir que o frontend acesse (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradeAnalyzer:
    def __init__(self, tick_value=0.25, contract_value=20):
        self.tick_value = tick_value
        self.contract_value = contract_value
    
    def parse_json_file(self, content: str):
        """Parseia arquivo JSON do NinjaTrader"""
        # Corrigir vírgula decimal
        content = re.sub(r'(?<=[\d])(,)(?=[\d])', '.', content)
        
        # Dividir múltiplos objetos JSON
        lines = content.strip().split('\n\n')
        data_list = []
        
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except:
                    continue
        
        return data_list
    
    def convert_to_dataframe(self, data_list: List[dict]):
        """Converte lista de dados em DataFrame"""
        records = []
        
        for item in data_list:
            try:
                record = {
                    'timestamp': pd.to_datetime(item['timestamp_barra']),
                    'open': float(item['barra']['open']),
                    'high': float(item['barra']['high']),
                    'low': float(item['barra']['low']),
                    'close': float(item['barra']['close']),
                    'volume': int(item['barra']['volume']),
                    'ema': float(item['indicadores']['ema']['valor']),
                    'ema_dist_ticks': int(item['indicadores']['ema']['distancia_close_ticks']),
                    'rsi': float(item['indicadores']['rsi']['valor']),
                    'atr_ticks': int(item['indicadores']['atr']['valor_ticks']),
                }
                records.append(record)
            except Exception as e:
                continue
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['hora'] = df['timestamp'].dt.hour
        df['minuto'] = df['timestamp'].dt.minute
        
        return df
    
    def simulate_trade(self, df, idx, direction, stop_ticks, target_ticks, max_bars=20):
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
        else:
            stop_price = entry + (stop_ticks * self.tick_value)
            target_price = entry - (target_ticks * self.tick_value)
            
            for i in range(1, max_bars + 1):
                future = df.iloc[idx + i]
                if future['high'] >= stop_price:
                    return {'result': 'STOP', 'profit': -stop_ticks, 'bars': i}
                elif future['low'] <= target_price:
                    return {'result': 'TARGET', 'profit': target_ticks, 'bars': i}
        
        return {'result': 'OPEN', 'profit': 0, 'bars': max_bars}
    
    def analyze(self, df, config: dict):
        """Executa análise completa"""
        results = []
        
        # Configurações
        min_win_rate = config.get('minWinRate', 70)
        min_trades = config.get('minTrades', 10)
        
        # Gerar setups para testar
        setups = []
        
        # Setup 1: Abertura USA
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
        
        # Testar cada setup com diferentes stops/targets
        configs_test = [
            {'stop': 10, 'target': 20},
            {'stop': 15, 'target': 30},
            {'stop': 20, 'target': 40},
            {'stop': 10, 'target': 30},
        ]
        
        for setup in setups:
            for cfg in configs_test:
                trades = []
                stop_ticks = cfg['stop']
                target_ticks = cfg['target']
                
                # Testar LONGs
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
                
                # Testar SHORTs
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
                                'net_profit_usd': total_profit * self.tick_value * self.contract_value,
                                'avg_profit_per_trade': (total_profit * self.tick_value * self.contract_value) / len(closed)
                            })
        
        # Ordenar por win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return {
            'total_setups_tested': len(setups) * len(configs_test),
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

@app.post("/api/v1/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    config: str = Form('{"minWinRate": 70, "minTrades": 10}')
):
    """
    Recebe arquivo JSON do NinjaTrader e retorna análise
    """
    try:
        # Ler arquivo
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # Parsear config
        config_dict = json.loads(config)
        
        # Processar
        data_list = analyzer.parse_json_file(content_str)
        df = analyzer.convert_to_dataframe(data_list)
        
        if len(df) == 0:
            return {"error": "Nenhum dado válido encontrado no arquivo"}
        
        # Analisar
        results = analyzer.analyze(df, config_dict)
        
        return {
            "status": "completed",
            "filename": file.filename,
            "results": results
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}