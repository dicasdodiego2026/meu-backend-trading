# main.py - TradeLog Analyzer API v4
# Estratégias Direção-Neutras com Consistência Diária

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
from itertools import combinations

app = FastAPI(title="TradeLog Analyzer API v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constantes ───
TICK_VALUE = 0.50
TICK_SIZE = 0.25
COMMISSION_RT = 1.04  # round-trip
MAX_DAILY_LOSS_TICKS = 60
MIN_RR_RATIO = 2.0
MAX_BARS_IN_TRADE = 20


# ═══════════════════════════════════════════════════
# PARSER - converte JSON dos logs em DataFrame
# ═══════════════════════════════════════════════════

def parse_json_content(content: str) -> List[dict]:
    """Parse flexível: aceita array JSON, objetos separados por \\n\\n, ou NDJSON."""
    content = re.sub(r'(?<=\d)(,)(?=\d)', '.', content)
    
    # Tenta array JSON direto
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    
    # Tenta blocos separados por \n\n
    data = []
    for block in content.strip().split('\n\n'):
        block = block.strip()
        if block:
            try:
                p = json.loads(block)
                data.extend(p if isinstance(p, list) else [p])
            except:
                continue
    
    if data:
        return data
    
    # Tenta NDJSON (uma linha por objeto)
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


def to_dataframe(entries: List[dict]) -> pd.DataFrame:
    """Converte entradas do log em DataFrame com indicadores derivados."""
    records = []
    for item in entries:
        try:
            ts = item.get('timestamp_barra') or item.get('timestamp')
            if not ts:
                continue

            barra = item.get('barra', {})
            ind = item.get('indicadores', {})

            r = {
                'timestamp': pd.to_datetime(ts),
                'open': float(barra.get('open', 0)),
                'high': float(barra.get('high', 0)),
                'low': float(barra.get('low', 0)),
                'close': float(barra.get('close', 0)),
                'volume': int(barra.get('volume', ind.get('volume', {}).get('valor', 0))),
                'direcao': barra.get('direcao', ''),
            }

            # RSI
            r['rsi'] = float(ind.get('rsi', {}).get('valor', 50))

            # EMA
            ema_data = ind.get('ema', {})
            r['ema'] = float(ema_data.get('valor', 0))
            r['ema_dist'] = int(ema_data.get('distancia_close_ticks', 0))

            # Fibonacci Pivots
            fp = ind.get('fibonacci_pivots', {})
            r['pivot_pp'] = float(fp.get('pp', 0))
            r['pivot_r1'] = float(fp.get('r1', 0))
            r['pivot_s1'] = float(fp.get('s1', 0))
            r['pivot_zona'] = fp.get('zona', '')

            # ATR se disponível
            r['atr'] = float(ind.get('atr', {}).get('valor', 0))

            records.append(r)
        except:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)

    # ─── Indicadores Derivados ───
    # Candle body e range
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = np.where(df['range'] > 0, df['body'] / df['range'], 0)

    # Posição do close dentro do range (0=low, 1=high)
    df['close_position'] = np.where(
        df['range'] > 0,
        (df['close'] - df['low']) / df['range'],
        0.5
    )

    # Variação vs barra anterior
    df['delta_close'] = df['close'].diff().fillna(0)
    df['delta_close_ticks'] = (df['delta_close'] / TICK_SIZE).round().astype(int)

    # RSI zones
    df['rsi_zone'] = pd.cut(
        df['rsi'],
        bins=[0, 30, 40, 60, 70, 100],
        labels=['oversold', 'weak', 'neutral', 'strong', 'overbought'],
        include_lowest=True
    ).astype(str)

    # Preço vs EMA
    df['above_ema'] = (df['close'] > df['ema']).astype(int)

    # Preço vs Pivot
    df['above_pp'] = np.where(df['pivot_pp'] > 0, (df['close'] > df['pivot_pp']).astype(int), -1)

    # Sequência de barras na mesma direção
    df['dir_num'] = np.where(df['direcao'] == 'ALTA', 1, np.where(df['direcao'] == 'BAIXA', -1, 0))
    streak = []
    current = 0
    for d in df['dir_num']:
        if d == 0:
            current = 0
        elif len(streak) == 0:
            current = d
        elif np.sign(d) == np.sign(current):
            current += d
        else:
            current = d
        streak.append(current)
    df['dir_streak'] = streak

    # Hora do dia (para janela operacional)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute

    return df


# ═══════════════════════════════════════════════════
# GERADOR DE SETUPS DIREÇÃO-NEUTROS
# Cada setup gera sinais LONG e SHORT simetricamente
# ═══════════════════════════════════════════════════

def generate_neutral_setups() -> List[dict]:
    """
    Gera setups que operam AMBAS as direções.
    Cada setup define condição de contexto + gatilho direcional.
    """
    setups = []
    idx = 0

    # ─── Condições de Contexto (direção-neutras) ───
    context_conditions = [
        # RSI em zona neutra (mercado sem tendência extrema)
        {"name": "RSI_Neutro", "long": [{"campo": "rsi", "op": ">", "val": 35}, {"campo": "rsi", "op": "<", "val": 65}],
         "short": [{"campo": "rsi", "op": ">", "val": 35}, {"campo": "rsi", "op": "<", "val": 65}]},
        
        # RSI em extremos (reversão)
        {"name": "RSI_Reversao",
         "long": [{"campo": "rsi", "op": "<", "val": 35}],
         "short": [{"campo": "rsi", "op": ">", "val": 65}]},
        
        {"name": "RSI_Reversao_Forte",
         "long": [{"campo": "rsi", "op": "<", "val": 30}],
         "short": [{"campo": "rsi", "op": ">", "val": 70}]},

        # Sem filtro RSI
        {"name": "Sem_RSI", "long": [], "short": []},
    ]

    # ─── Gatilhos Direcionais (simétricos) ───
    directional_triggers = [
        # Barra de reversão: barra anterior na direção oposta
        {"name": "Barra_Reversao",
         "long": [{"campo": "direcao", "op": "==", "val": "ALTA"}],
         "short": [{"campo": "direcao", "op": "==", "val": "BAIXA"}]},

        # Preço vs EMA
        {"name": "Pullback_EMA",
         "long": [{"campo": "above_ema", "op": "==", "val": 1}, {"campo": "ema_dist", "op": "<", "val": 5}],
         "short": [{"campo": "above_ema", "op": "==", "val": 0}, {"campo": "ema_dist", "op": ">", "val": -5}]},

        {"name": "Afastado_EMA",
         "long": [{"campo": "ema_dist", "op": ">", "val": 8}],
         "short": [{"campo": "ema_dist", "op": "<", "val": -8}]},

        # Close position no candle
        {"name": "Close_Forte",
         "long": [{"campo": "close_position", "op": ">", "val": 0.7}],
         "short": [{"campo": "close_position", "op": "<", "val": 0.3}]},

        # Sequência direcional (momentum)
        {"name": "Momentum_2bars",
         "long": [{"campo": "dir_streak", "op": ">=", "val": 2}],
         "short": [{"campo": "dir_streak", "op": "<=", "val": -2}]},

        {"name": "Momentum_3bars",
         "long": [{"campo": "dir_streak", "op": ">=", "val": 3}],
         "short": [{"campo": "dir_streak", "op": "<=", "val": -3}]},

        # Body ratio (candle cheio vs doji)
        {"name": "Candle_Cheio",
         "long": [{"campo": "body_ratio", "op": ">", "val": 0.6}, {"campo": "direcao", "op": "==", "val": "ALTA"}],
         "short": [{"campo": "body_ratio", "op": ">", "val": 0.6}, {"campo": "direcao", "op": "==", "val": "BAIXA"}]},

        # Delta grande (movimento forte)
        {"name": "Delta_Forte",
         "long": [{"campo": "delta_close_ticks", "op": ">", "val": 4}],
         "short": [{"campo": "delta_close_ticks", "op": "<", "val": -4}]},
    ]

    # ─── Filtros de Preço vs Pivot ───
    pivot_filters = [
        {"name": "", "long": [], "short": []},  # sem filtro
        {"name": "+PivotAbove",
         "long": [{"campo": "above_pp", "op": "==", "val": 1}],
         "short": [{"campo": "above_pp", "op": "==", "val": 0}]},
    ]

    # ─── Combinar: contexto × gatilho × pivot ───
    for ctx in context_conditions:
        for trig in directional_triggers:
            for piv in pivot_filters:
                idx += 1
                name = f"{ctx['name']}_{trig['name']}{piv['name']}"

                long_conds = ctx['long'] + trig['long'] + piv['long']
                short_conds = ctx['short'] + trig['short'] + piv['short']

                setups.append({
                    "name": name,
                    "long_conditions": long_conds,
                    "short_conditions": short_conds,
                })

    return setups


# ═══════════════════════════════════════════════════
# BACKTESTER com controle de risco diário
# ═══════════════════════════════════════════════════

def apply_conditions(df: pd.DataFrame, conditions: List[dict]) -> pd.Series:
    """Retorna máscara booleana onde TODAS as condições são atendidas."""
    mask = pd.Series(True, index=df.index)
    for c in conditions:
        campo = c['campo']
        op = c['op']
        val = c['val']
        if campo not in df.columns:
            return pd.Series(False, index=df.index)
        
        col = df[campo]
        if op == '<': mask &= (col < val)
        elif op == '>': mask &= (col > val)
        elif op == '==': mask &= (col == val)
        elif op == '>=': mask &= (col >= val)
        elif op == '<=': mask &= (col <= val)
    return mask


def backtest_neutral(
    df: pd.DataFrame,
    long_conditions: List[dict],
    short_conditions: List[dict],
    stop_ticks: int,
    target_ticks: int,
) -> List[dict]:
    """
    Backtest direção-neutra com max loss diário de 60 ticks.
    Retorna lista de trades com direção, resultado, etc.
    """
    long_mask = apply_conditions(df, long_conditions)
    short_mask = apply_conditions(df, short_conditions)

    trades = []
    next_allowed = 0
    daily_loss_ticks = 0

    for i in range(len(df) - MAX_BARS_IN_TRADE):
        if i < next_allowed:
            continue

        # Checar se estourou loss diário
        if daily_loss_ticks >= MAX_DAILY_LOSS_TICKS:
            break

        direction = None
        if long_mask.iloc[i]:
            direction = 'LONG'
        elif short_mask.iloc[i]:
            direction = 'SHORT'
        else:
            continue

        entry_price = df.iloc[i]['close']
        entry_time = df.iloc[i]['timestamp']

        if direction == 'LONG':
            stop_price = entry_price - (stop_ticks * TICK_SIZE)
            target_price = entry_price + (target_ticks * TICK_SIZE)
        else:
            stop_price = entry_price + (stop_ticks * TICK_SIZE)
            target_price = entry_price - (target_ticks * TICK_SIZE)

        result = None
        exit_price = entry_price
        exit_time = entry_time
        bars = 0

        for j in range(1, MAX_BARS_IN_TRADE + 1):
            bar = df.iloc[i + j]
            if direction == 'LONG':
                if bar['low'] <= stop_price:
                    result = 'STOP'
                    exit_price = stop_price
                    bars = j
                    break
                elif bar['high'] >= target_price:
                    result = 'TARGET'
                    exit_price = target_price
                    bars = j
                    break
            else:
                if bar['high'] >= stop_price:
                    result = 'STOP'
                    exit_price = stop_price
                    bars = j
                    break
                elif bar['low'] <= target_price:
                    result = 'TARGET'
                    exit_price = target_price
                    bars = j
                    break

        if result is None:
            # Timeout - fecha no close da última barra
            result = 'TIMEOUT'
            exit_price = df.iloc[i + MAX_BARS_IN_TRADE]['close']
            bars = MAX_BARS_IN_TRADE
            profit_ticks = int(round((exit_price - entry_price) / TICK_SIZE)) if direction == 'LONG' \
                else int(round((entry_price - exit_price) / TICK_SIZE))
        else:
            profit_ticks = target_ticks if result == 'TARGET' else -stop_ticks

        # Atualizar loss diário
        if profit_ticks < 0:
            daily_loss_ticks += abs(profit_ticks)

        # Checar se próximo trade estouraria o limite
        if daily_loss_ticks >= MAX_DAILY_LOSS_TICKS and profit_ticks < 0:
            # Registra o trade mas para depois
            pass

        profit_usd = (profit_ticks * TICK_VALUE) - COMMISSION_RT

        trades.append({
            'direction': direction,
            'result': result,
            'profit_ticks': profit_ticks,
            'profit_usd': round(profit_usd, 2),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': str(entry_time),
            'exit_time': str(df.iloc[i + bars]['timestamp']),
            'bars': bars,
        })

        next_allowed = i + bars + 1

    return trades


# ═══════════════════════════════════════════════════
# ANÁLISE EM LOTE - Encontra as TOP 3 estratégias
# ═══════════════════════════════════════════════════

@app.post("/api/v1/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    config: str = Form('{}')
):
    try:
        cfg = json.loads(config) if config else {}
    except:
        cfg = {}

    min_wr = cfg.get('minWinRate', 55)

    # 1. Parse todos os dias
    days_data: Dict[str, pd.DataFrame] = {}
    for f in files:
        content = (await f.read()).decode('utf-8')
        entries = parse_json_content(content)
        if entries:
            df = to_dataframe(entries)
            if not df.empty:
                days_data[f.filename] = df

    if not days_data:
        return {"status": "completed", "total_days": 0, "total_setups_tested": 0, "consistent_setups": []}

    total_days = len(days_data)

    # 2. Gerar setups neutros
    setups = generate_neutral_setups()

    # 3. Stops e targets com RR >= 2:1
    stop_target_combos = []
    for stop in [8, 10, 12, 15, 20]:
        for rr in [2.0, 2.5, 3.0]:
            target = int(stop * rr)
            stop_target_combos.append((stop, target))

    # 4. Testar cada setup × stop/target em cada dia
    all_results = []

    for setup in setups:
        for stop, target in stop_target_combos:
            daily_stats = []
            total_longs = 0
            total_shorts = 0
            all_trades_combined = []

            for day_name, df in days_data.items():
                trades = backtest_neutral(
                    df, setup['long_conditions'], setup['short_conditions'],
                    stop, target
                )

                count = len(trades)
                if count == 0:
                    daily_stats.append({
                        "day": day_name,
                        "total_trades": 0,
                        "longs": 0,
                        "shorts": 0,
                        "profit": 0,
                        "win_rate": 0,
                        "profitable": False,
                    })
                    continue

                wins = sum(1 for t in trades if t['profit_ticks'] > 0)
                profit_usd = sum(t['profit_usd'] for t in trades)
                longs = sum(1 for t in trades if t['direction'] == 'LONG')
                shorts = sum(1 for t in trades if t['direction'] == 'SHORT')
                wr = (wins / count * 100) if count > 0 else 0

                total_longs += longs
                total_shorts += shorts
                all_trades_combined.extend(trades)

                daily_stats.append({
                    "day": day_name,
                    "total_trades": count,
                    "longs": longs,
                    "shorts": shorts,
                    "profit": round(profit_usd, 2),
                    "win_rate": round(wr, 1),
                    "profitable": profit_usd > 0,
                })

            # ─── Filtros de Qualidade ───
            active_days = [d for d in daily_stats if d['total_trades'] > 0]

            # REGRA 1: Deve ter trade TODOS os dias
            if len(active_days) < total_days:
                continue

            # REGRA 2: Equilíbrio direcional (mín 30% de cada lado)
            total_trades_all = total_longs + total_shorts
            if total_trades_all == 0:
                continue
            long_ratio = total_longs / total_trades_all
            if long_ratio < 0.30 or long_ratio > 0.70:
                continue

            # REGRA 3: Consistência (dias lucrativos)
            profitable_days = sum(1 for d in active_days if d['profitable'])
            consistency = profitable_days / len(active_days)

            # REGRA 4: Win rate médio
            avg_wr = sum(d['win_rate'] for d in active_days) / len(active_days)
            if avg_wr < min_wr:
                continue

            # Calcular métricas agregadas
            total_profit = sum(d['profit'] for d in active_days)
            gross_profit = sum(t['profit_usd'] for t in all_trades_combined if t['profit_usd'] > 0)
            gross_loss = abs(sum(t['profit_usd'] for t in all_trades_combined if t['profit_usd'] < 0))
            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999

            # Últimos 5 trades como amostra
            sample = all_trades_combined[-5:] if all_trades_combined else []

            all_results.append({
                "setup_name": setup['name'],
                "stop_ticks": stop,
                "target_ticks": target,
                "days_tested": total_days,
                "days_profitable": profitable_days,
                "consistency": round(consistency, 2),
                "avg_win_rate": round(avg_wr, 1),
                "avg_profit_factor": profit_factor,
                "total_profit_usd": round(total_profit, 2),
                "avg_daily_profit_usd": round(total_profit / len(active_days), 2),
                "total_trades": total_trades_all,
                "total_longs": total_longs,
                "total_shorts": total_shorts,
                "long_ratio": round(long_ratio, 2),
                "daily_results": active_days,
                "rules": {
                    "entry_long": setup['long_conditions'],
                    "entry_short": setup['short_conditions'],
                },
                "rules_exit": {
                    "tipo": "stop_target",
                    "stop_ticks": stop,
                    "target_ticks": target,
                    "stop_usd": round(stop * TICK_VALUE + COMMISSION_RT, 2),
                    "target_usd": round(target * TICK_VALUE - COMMISSION_RT, 2),
                    "max_bars": MAX_BARS_IN_TRADE,
                    "max_daily_loss_ticks": MAX_DAILY_LOSS_TICKS,
                    "rr_ratio": round(target / stop, 1),
                    "descricao": f"Stop {stop}t / Target {target}t (RR {round(target/stop,1)}:1) | Max loss diário: {MAX_DAILY_LOSS_TICKS}t"
                },
                "sample_trades": [
                    {
                        "tipo": t['direction'],
                        "result": t['result'],
                        "profit_ticks": t['profit_ticks'],
                        "entry_price": t['entry_price'],
                        "exit_price": t['exit_price'],
                        "entry_time": t['entry_time'],
                        "exit_time": t['exit_time'],
                        "bars": t['bars'],
                    }
                    for t in sample
                ],
            })

    # 5. Ranquear: prioridade = consistência, depois profit
    all_results.sort(key=lambda x: (x['consistency'], x['total_profit_usd']), reverse=True)

    # Top 3 consistentes (≥90%) + all_setups para exploração
    consistent = [r for r in all_results if r['consistency'] >= 0.9][:3]

    return {
        "status": "completed",
        "total_days": total_days,
        "total_setups_tested": len(setups) * len(stop_target_combos),
        "consistent_setups": consistent,
        "all_setups": all_results[:50],  # Top 50 para exploração
    }


# ═══════════════════════════════════════════════════
# ENDPOINTS AUXILIARES
# ═══════════════════════════════════════════════════

@app.get("/")
def home():
    return {"message": "TradeLog Analyzer API v4", "status": "online"}


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/analyze")
async def analyze_file(file: UploadFile = File(...), config: str = Form('{}')):
    """Análise individual - mantida para compatibilidade."""
    try:
        cfg = json.loads(config) if config else {}
    except:
        cfg = {}

    content = (await file.read()).decode('utf-8')
    entries = parse_json_content(content)
    if not entries:
        return {"status": "error", "message": "Nenhum dado encontrado no arquivo"}

    df = to_dataframe(entries)
    if df.empty:
        return {"status": "error", "message": "Não foi possível converter os dados"}

    stop = cfg.get('stopTicks', 20)
    target = cfg.get('targetTicks', 40)
    min_wr = cfg.get('minWinRate', 60)

    setups = generate_neutral_setups()
    results = []

    for setup in setups:
        trades = backtest_neutral(df, setup['long_conditions'], setup['short_conditions'], stop, target)
        if not trades:
            continue

        wins = sum(1 for t in trades if t['profit_ticks'] > 0)
        wr = (wins / len(trades)) * 100
        if wr < min_wr:
            continue

        profit = sum(t['profit_usd'] for t in trades)
        gp = sum(t['profit_usd'] for t in trades if t['profit_usd'] > 0)
        gl = abs(sum(t['profit_usd'] for t in trades if t['profit_usd'] < 0))

        results.append({
            "setup_name": setup['name'],
            "stop_ticks": stop,
            "target_ticks": target,
            "win_rate": round(wr, 1),
            "profit_factor": round(gp / gl, 2) if gl > 0 else 999,
            "net_profit_usd": round(profit, 2),
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "rules": {
                "entry_long": setup['long_conditions'],
                "entry_short": setup['short_conditions'],
            },
            "rules_exit": {
                "tipo": "stop_target",
                "stop_ticks": stop,
                "target_ticks": target,
                "stop_usd": round(stop * TICK_VALUE + COMMISSION_RT, 2),
                "target_usd": round(target * TICK_VALUE - COMMISSION_RT, 2),
                "max_bars": MAX_BARS_IN_TRADE,
                "descricao": f"Stop {stop}t / Target {target}t",
            },
            "sample_trades": [
                {
                    "tipo": t['direction'],
                    "result": t['result'],
                    "profit_ticks": t['profit_ticks'],
                    "entry_price": t['entry_price'],
                    "exit_price": t['exit_price'],
                    "entry_time": t['entry_time'],
                    "exit_time": t['exit_time'],
                    "bars": t['bars'],
                }
                for t in trades[-5:]
            ],
        })

    results.sort(key=lambda x: x['net_profit_usd'], reverse=True)

    return {
        "status": "completed",
        "filename": file.filename,
        "results": {
            "total_setups_tested": len(setups),
            "profitable_setups": len(results),
            "best_setups": results[:20],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
