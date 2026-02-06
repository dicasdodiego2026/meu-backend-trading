# main.py COMPLETO - TradeLog Analyzer API
# Retorna regras estruturadas para automação

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

        # Try as JSON array first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                print(f"Parsed as JSON array: {len(parsed)} items")
                return parsed
            else:
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Split by double newline
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
                except json.JSONDecodeError as e:
                    print(f"Erro ao parsear linha: {e}")
                    continue

        # Try line by line
        if not data_list:
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except:
                        continue

        print(f"Total de objetos JSON parseados: {len(data_list)}")
        return data_list

    def convert_to_dataframe(self, data_list: List[dict]) -> pd.DataFrame:
        records = []

        for i, item in enumerate(data_list):
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

                tf = item.get('timeframe', {})
                record['timeframe_tipo'] = tf.get('tipo', 'Minute')
                record['timeframe_valor'] = tf.get('valor', 1)

                if 'ema' in indicadores:
                    record['ema'] = float(indicadores['ema'].get('valor', 0))
                    record['ema_periodo'] = int(indicadores['ema'].get('periodo', 14))
                    record['ema_dist_ticks'] = int(indicadores['ema'].get('distancia_close_ticks', 0))
                else:
                    record['ema'] = 0
                    record['ema_periodo'] = 14
                    record['ema_dist_ticks'] = 0

                if 'rsi' in indicadores:
                    record['rsi'] = float(indicadores['rsi'].get('valor', 50))
                    record['rsi_periodo'] = int(indicadores['rsi'].get('periodo', 14))
                    record['rsi_smooth'] = int(indicadores['rsi'].get('smooth', 3))
                else:
                    record['rsi'] = 50
                    record['rsi_periodo'] = 14
                    record['rsi_smooth'] = 3

                if 'atr' in indicadores:
                    record['atr_ticks'] = int(indicadores['atr'].get('valor_ticks', 10))
                    record['atr_periodo'] = int(indicadores['atr'].get('periodo', 14))
                else:
                    record['atr_ticks'] = 10
                    record['atr_periodo'] = 14

                if 'fibonacci_pivots' in indicadores:
                    fp = indicadores['fibonacci_pivots']
                    record['pivot_pp'] = float(fp.get('pp', 0))
                    record['pivot_r1'] = float(fp.get('r1', 0))
                    record['pivot_s1'] = float(fp.get('s1', 0))
                    record['pivot_zona'] = fp.get('zona', '')
                    record['pivot_range'] = fp.get('range', 'Daily')
                else:
                    record['pivot_pp'] = 0
                    record['pivot_r1'] = 0
                    record['pivot_s1'] = 0
                    record['pivot_zona'] = ''
                    record['pivot_range'] = 'Daily'

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
        if idx + max_bars >= len(df):
            return None

        entry = df.iloc[idx]['close']
        entry_time = df.iloc[idx]['timestamp'].isoformat()

        if direction == 'LONG':
            stop_price = entry - (stop_ticks * self.tick_size)
            target_price = entry + (target_ticks * self.tick_size)
            for i in range(1, max_bars + 1):
                future = df.iloc[idx + i]
                if future['low'] <= stop_price:
                    return {'result': 'STOP', 'profit': -stop_ticks, 'bars': i,
                            'entry_price': entry, 'entry_time': entry_time,
                            'exit_price': stop_price, 'exit_time': future['timestamp'].isoformat()}
                elif future['high'] >= target_price:
                    return {'result': 'TARGET', 'profit': target_ticks, 'bars': i,
                            'entry_price': entry, 'entry_time': entry_time,
                            'exit_price': target_price, 'exit_time': future['timestamp'].isoformat()}
        else:
            stop_price = entry + (stop_ticks * self.tick_size)
            target_price = entry - (target_ticks * self.tick_size)
            for i in range(1, max_bars + 1):
                future = df.iloc[idx + i]
                if future['high'] >= stop_price:
                    return {'result': 'STOP', 'profit': -stop_ticks, 'bars': i,
                            'entry_price': entry, 'entry_time': entry_time,
                            'exit_price': stop_price, 'exit_time': future['timestamp'].isoformat()}
                elif future['low'] <= target_price:
                    return {'result': 'TARGET', 'profit': target_ticks, 'bars': i,
                            'entry_price': entry, 'entry_time': entry_time,
                            'exit_price': target_price, 'exit_time': future['timestamp'].isoformat()}

        return {'result': 'OPEN', 'profit': 0, 'bars': max_bars,
                'entry_price': entry, 'entry_time': entry_time,
                'exit_price': entry, 'exit_time': df.iloc[idx + max_bars]['timestamp'].isoformat()}

    def analyze(self, df: pd.DataFrame, config: dict) -> Dict:
        results = []

        min_win_rate = config.get('minWinRate', 70)
        min_trades = config.get('minTrades', 10)

        timeframe_tipo = df['timeframe_tipo'].iloc[0] if 'timeframe_tipo' in df.columns else 'Minute'
        timeframe_valor = int(df['timeframe_valor'].iloc[0]) if 'timeframe_valor' in df.columns else 1
        ema_periodo = int(df['ema_periodo'].iloc[0]) if 'ema_periodo' in df.columns else 14
        rsi_periodo = int(df['rsi_periodo'].iloc[0]) if 'rsi_periodo' in df.columns else 14
        rsi_smooth = int(df['rsi_smooth'].iloc[0]) if 'rsi_smooth' in df.columns else 3
        atr_periodo = int(df['atr_periodo'].iloc[0]) if 'atr_periodo' in df.columns else 14
        pivot_range = df['pivot_range'].iloc[0] if 'pivot_range' in df.columns else 'Daily'

        print(f"Iniciando análise: min_win_rate={min_win_rate}, min_trades={min_trades}")
        print(f"Timeframe: {timeframe_valor} {timeframe_tipo}, EMA({ema_periodo}), RSI({rsi_periodo},{rsi_smooth}), ATR({atr_periodo})")

        setups = []

        # Setup 1: Abertura USA
        for rsi_long in [35, 40, 45, 50]:
            for rsi_short in [50, 55, 60, 65]:
                for minuto in [30, 35, 40]:
                    cond_long = (
                        (df['hora'] == 9) &
                        (df['minuto'] >= minuto) &
                        (df['rsi'] < rsi_long) &
                        (df['direcao'] == 'ALTA')
                    )
                    cond_short = (
                        (df['hora'] == 9) &
                        (df['minuto'] >= minuto) &
                        (df['rsi'] > rsi_short) &
                        (df['direcao'] == 'BAIXA')
                    )
                    setups.append({
                        'name': f'Abertura_RSI{rsi_long}_{rsi_short}_M{minuto}',
                        'long': cond_long,
                        'short': cond_short,
                        'rules': {
                            'timeframe': {'tipo': timeframe_tipo, 'valor': timeframe_valor},
                            'indicators': [
                                {'name': 'RSI', 'periodo': rsi_periodo, 'smooth': rsi_smooth},
                                {'name': 'Candle Direction', 'campo': 'direcao'},
                            ],
                            'entry_long': [
                                {'campo': 'hora', 'operador': '==', 'valor': 9, 'descricao': 'Horário de abertura EUA'},
                                {'campo': 'minuto', 'operador': '>=', 'valor': minuto, 'descricao': f'A partir do minuto {minuto}'},
                                {'campo': 'rsi', 'operador': '<', 'valor': rsi_long, 'descricao': f'RSI abaixo de {rsi_long} (sobrevendido)'},
                                {'campo': 'direcao', 'operador': '==', 'valor': 'ALTA', 'descricao': 'Candle de alta (bullish)'},
                            ],
                            'entry_short': [
                                {'campo': 'hora', 'operador': '==', 'valor': 9, 'descricao': 'Horário de abertura EUA'},
                                {'campo': 'minuto', 'operador': '>=', 'valor': minuto, 'descricao': f'A partir do minuto {minuto}'},
                                {'campo': 'rsi', 'operador': '>', 'valor': rsi_short, 'descricao': f'RSI acima de {rsi_short} (sobrecomprado)'},
                                {'campo': 'direcao', 'operador': '==', 'valor': 'BAIXA', 'descricao': 'Candle de baixa (bearish)'},
                            ],
                            'exit': {
                                'tipo': 'STOP_TARGET',
                                'max_bars': 20,
                                'descricao': 'Saída por stop loss ou take profit, o que ocorrer primeiro (máx 20 barras)'
                            }
                        }
                    })

        # Setup 2: RSI Extremo
        for rsi_low in [20, 25, 30]:
            for rsi_high in [70, 75, 80]:
                cond_long = (df['rsi'] < rsi_low) & (df['direcao'] == 'ALTA')
                cond_short = (df['rsi'] > rsi_high) & (df['direcao'] == 'BAIXA')
                setups.append({
                    'name': f'RSI_Extremo_{rsi_low}_{rsi_high}',
                    'long': cond_long,
                    'short': cond_short,
                    'rules': {
                        'timeframe': {'tipo': timeframe_tipo, 'valor': timeframe_valor},
                        'indicators': [
                            {'name': 'RSI', 'periodo': rsi_periodo, 'smooth': rsi_smooth},
                            {'name': 'Candle Direction', 'campo': 'direcao'},
                        ],
                        'entry_long': [
                            {'campo': 'rsi', 'operador': '<', 'valor': rsi_low, 'descricao': f'RSI extremo abaixo de {rsi_low}'},
                            {'campo': 'direcao', 'operador': '==', 'valor': 'ALTA', 'descricao': 'Candle de alta confirmando reversão'},
                        ],
                        'entry_short': [
                            {'campo': 'rsi', 'operador': '>', 'valor': rsi_high, 'descricao': f'RSI extremo acima de {rsi_high}'},
                            {'campo': 'direcao', 'operador': '==', 'valor': 'BAIXA', 'descricao': 'Candle de baixa confirmando reversão'},
                        ],
                        'exit': {
                            'tipo': 'STOP_TARGET',
                            'max_bars': 20,
                            'descricao': 'Saída por stop loss ou take profit'
                        }
                    }
                })

        # Setup 3: EMA Distância
        for dist in [5, 10, 15, 20, 30]:
            for rsi_filter in [35, 40, 45]:
                cond_long = (df['ema_dist_ticks'] < -dist) & (df['rsi'] < rsi_filter)
                cond_short = (df['ema_dist_ticks'] > dist) & (df['rsi'] > (100 - rsi_filter))
                setups.append({
                    'name': f'EMA_Dist_{dist}_RSI{rsi_filter}',
                    'long': cond_long,
                    'short': cond_short,
                    'rules': {
                        'timeframe': {'tipo': timeframe_tipo, 'valor': timeframe_valor},
                        'indicators': [
                            {'name': 'EMA', 'periodo': ema_periodo},
                            {'name': 'RSI', 'periodo': rsi_periodo, 'smooth': rsi_smooth},
                        ],
                        'entry_long': [
                            {'campo': 'ema_dist_ticks', 'operador': '<', 'valor': -dist, 'descricao': f'Preço {dist}+ ticks abaixo da EMA({ema_periodo})'},
                            {'campo': 'rsi', 'operador': '<', 'valor': rsi_filter, 'descricao': f'RSI abaixo de {rsi_filter}'},
                        ],
                        'entry_short': [
                            {'campo': 'ema_dist_ticks', 'operador': '>', 'valor': dist, 'descricao': f'Preço {dist}+ ticks acima da EMA({ema_periodo})'},
                            {'campo': 'rsi', 'operador': '>', 'valor': 100 - rsi_filter, 'descricao': f'RSI acima de {100 - rsi_filter}'},
                        ],
                        'exit': {
                            'tipo': 'STOP_TARGET',
                            'max_bars': 20,
                            'descricao': 'Saída por stop loss ou take profit'
                        }
                    }
                })

        # Setup 4: Pivot Zone
        for zona_long in ['ENTRE_S1_PP']:
            for zona_short in ['ENTRE_PP_R1']:
                cond_long = (df['pivot_zona'] == zona_long) & (df['direcao'] == 'ALTA') & (df['rsi'] < 45)
                cond_short = (df['pivot_zona'] == zona_short) & (df['direcao'] == 'BAIXA') & (df['rsi'] > 55)
                setups.append({
                    'name': f'Pivot_{zona_long}_{zona_short}',
                    'long': cond_long,
                    'short': cond_short,
                    'rules': {
                        'timeframe': {'tipo': timeframe_tipo, 'valor': timeframe_valor},
                        'indicators': [
                            {'name': 'Fibonacci Pivots', 'range': pivot_range},
                            {'name': 'RSI', 'periodo': rsi_periodo, 'smooth': rsi_smooth},
                            {'name': 'Candle Direction', 'campo': 'direcao'},
                        ],
                        'entry_long': [
                            {'campo': 'pivot_zona', 'operador': '==', 'valor': zona_long, 'descricao': f'Preço na zona {zona_long}'},
                            {'campo': 'direcao', 'operador': '==', 'valor': 'ALTA', 'descricao': 'Candle de alta'},
                            {'campo': 'rsi', 'operador': '<', 'valor': 45, 'descricao': 'RSI abaixo de 45'},
                        ],
                        'entry_short': [
                            {'campo': 'pivot_zona', 'operador': '==', 'valor': zona_short, 'descricao': f'Preço na zona {zona_short}'},
                            {'campo': 'direcao', 'operador': '==', 'valor': 'BAIXA', 'descricao': 'Candle de baixa'},
                            {'campo': 'rsi', 'operador': '>', 'valor': 55, 'descricao': 'RSI acima de 55'},
                        ],
                        'exit': {
                            'tipo': 'STOP_TARGET',
                            'max_bars': 20,
                            'descricao': 'Saída por stop loss ou take profit'
                        }
                    }
                })

        # Stop/Target configs
        user_stop = config.get('stopTicks', 20)
        user_target = config.get('targetTicks', 40)
        configs_test = [
            {'stop': user_stop, 'target': user_target},
            {'stop': 10, 'target': 20},
            {'stop': 15, 'target': 30},
            {'stop': 20, 'target': 40},
            {'stop': 10, 'target': 30},
            {'stop': 15, 'target': 45},
            {'stop': 8, 'target': 16},
            {'stop': 12, 'target': 24},
        ]
        seen = set()
        unique_configs = []
        for c in configs_test:
            key = (c['stop'], c['target'])
            if key not in seen:
                seen.add(key)
                unique_configs.append(c)
        configs_test = unique_configs

        total_tests = len(setups) * len(configs_test)
        print(f"Total de combinações a testar: {total_tests}")

        for setup in setups:
            for cfg in configs_test:
                trades = []
                stop_ticks = cfg['stop']
                target_ticks = cfg['target']

                long_signals = df[setup['long']].index
                for idx in long_signals:
                    if idx < len(df) - 20:
                        result = self.simulate_trade(df, idx, 'LONG', stop_ticks, target_ticks)
                        if result:
                            trades.append({**result, 'tipo': 'LONG'})

                short_signals = df[setup['short']].index
                for idx in short_signals:
                    if idx < len(df) - 20:
                        result = self.simulate_trade(df, idx, 'SHORT', stop_ticks, target_ticks)
                        if result:
                            trades.append({**result, 'tipo': 'SHORT'})

                if len(trades) >= min_trades:
                    closed = [t for t in trades if t['result'] != 'OPEN']
                    if len(closed) > 0:
                        wins = len([t for t in closed if t['result'] == 'TARGET'])
                        win_rate = (wins / len(closed)) * 100

                        if win_rate >= min_win_rate:
                            total_profit = sum(t['profit'] for t in closed)
                            gross_profit = sum(t['profit'] for t in closed if t['profit'] > 0)
                            gross_loss = abs(sum(t['profit'] for t in closed if t['profit'] < 0))
                            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

                            results.append({
                                'setup_name': setup['name'],
                                'stop_ticks': stop_ticks,
                                'target_ticks': target_ticks,
                                'ratio': f"1:{round(target_ticks / stop_ticks, 1)}",
                                'total_trades': len(closed),
                                'wins': wins,
                                'losses': len(closed) - wins,
                                'win_rate': round(win_rate, 1),
                                'profit_factor': round(profit_factor, 2),
                                'net_profit_ticks': total_profit,
                                'net_profit_usd': round(total_profit * self.tick_value, 2),
                                'avg_profit_per_trade': round((total_profit * self.tick_value) / len(closed), 2),
                                'rules': setup['rules'],
                                'rules_exit': {
                                    **setup['rules']['exit'],
                                    'stop_ticks': stop_ticks,
                                    'target_ticks': target_ticks,
                                    'stop_usd': round(stop_ticks * self.tick_value, 2),
                                    'target_usd': round(target_ticks * self.tick_value, 2),
                                },
                                'sample_trades': [
                                    {
                                        'tipo': t['tipo'],
                                        'result': t['result'],
                                        'profit_ticks': t['profit'],
                                        'entry_price': t.get('entry_price'),
                                        'entry_time': t.get('entry_time'),
                                        'exit_price': t.get('exit_price'),
                                        'exit_time': t.get('exit_time'),
                                        'bars': t['bars'],
                                    }
                                    for t in closed[:10]
                                ],
                            })

        results.sort(key=lambda x: x['win_rate'], reverse=True)

        return {
            'total_setups_tested': total_tests,
            'profitable_setups': len(results),
            'best_setups': results[:20],
            'data_info': {
                'timeframe': f"{timeframe_valor} {timeframe_tipo}",
                'ema_periodo': ema_periodo,
                'rsi_periodo': rsi_periodo,
                'rsi_smooth': rsi_smooth,
                'atr_periodo': atr_periodo,
                'pivot_range': pivot_range,
            },
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
    try:
        print(f"Recebendo arquivo: {file.filename}")
        contents = await file.read()
        content_str = contents.decode('utf-8')
        print(f"Tamanho: {len(content_str)} caracteres")

        try:
            config_dict = json.loads(config)
        except:
            config_dict = {"minWinRate": 70, "minTrades": 10}

        print(f"Config: {config_dict}")

        data_list = analyzer.parse_json_file(content_str)

        if not data_list:
            return {"status": "error", "message": "Nenhum dado JSON válido encontrado"}

        df = analyzer.convert_to_dataframe(data_list)

        if len(df) == 0:
            return {"status": "error", "message": "Nenhum dado válido após conversão"}

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