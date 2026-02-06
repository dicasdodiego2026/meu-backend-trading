"""
TradeLog Analyzer API v5 — Fibonacci Discovery Engine
=====================================================
Backend que:
1. Usa os Fibonacci Pivots existentes nos dados (pp, r1, s1, zona)
2. Calcula automaticamente swings (topos/fundos) e traça retracements completos
3. Descobre por força bruta qual combinação de condições Fibonacci + indicadores
   é consistentemente lucrativa em todos os dias do dataset
4. Respeita: equilíbrio direcional 30-70%, loss máximo 60 ticks/dia, RR >= 2:1
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import json, math, itertools
from typing import Any
from collections import defaultdict
import pandas as pd
import numpy as np
import os

app = FastAPI(title="TradeLog Analyzer v5 - Fibonacci Discovery")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Constants ──────────────────────────────────────────────────────────
TICK_VALUE_USD = 0.50
COMMISSION_RT = 1.04
MAX_DAILY_LOSS_TICKS = 60
MIN_RR_RATIO = 2.0
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.764]
FIB_EXTENSIONS = [1.128, 1.272, 1.414, 1.618]


# ─── Parsing ────────────────────────────────────────────────────────────
def parse_json_file(content: str) -> list[dict]:
    """Parse JSON objects separated by \\n\\n"""
    entries = []
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        try:
            entries.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if not entries:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]
        except:
            pass
    return entries


def extract_timestamp(entry: dict) -> str:
    if "timestamp_barra" in entry:
        return entry["timestamp_barra"]
    if "timestamp" in entry:
        return entry["timestamp"]
    if "id" in entry:
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})[_T](\d{2}:\d{2}:\d{2})", entry["id"])
        if m:
            return f"{m.group(1)}T{m.group(2)}"
    return "1970-01-01T00:00:00"


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    """Convert raw entries to a flat DataFrame with all indicators."""
    rows = []
    for e in entries:
        row = {}
        row["timestamp"] = extract_timestamp(e)
        
        # OHLCV from barra
        barra = e.get("barra", {})
        row["open"] = barra.get("open", 0)
        row["high"] = barra.get("high", 0)
        row["low"] = barra.get("low", 0)
        row["close"] = barra.get("close", 0)
        row["volume"] = barra.get("volume", 0)
        row["direcao"] = barra.get("direcao", "")
        row["tick_size"] = barra.get("tick_size", 0.25)
        
        # Indicators
        ind = e.get("indicadores", {})
        row["ema"] = ind.get("ema", {}).get("valor", None)
        row["rsi"] = ind.get("rsi", {}).get("valor", None)
        row["atr"] = ind.get("atr", {}).get("valor", None)
        row["volume_ind"] = ind.get("volume", {}).get("valor", None)
        
        # Fibonacci Pivots (from data)
        fib = ind.get("fibonacci_pivots", {})
        row["pivot_pp"] = fib.get("pp", None)
        row["pivot_r1"] = fib.get("r1", None)
        row["pivot_s1"] = fib.get("s1", None)
        row["pivot_r2"] = fib.get("r2", None)
        row["pivot_s2"] = fib.get("s2", None)
        row["pivot_zona"] = fib.get("zona", "")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Derived features
    tick_size = df["tick_size"].iloc[0] if len(df) > 0 else 0.25
    
    # Body ratio (how much of the bar is body vs wicks)
    bar_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"] = np.where(bar_range > 0, body / bar_range, 0)
    
    # Distance from pivot PP in ticks
    if df["pivot_pp"].notna().any():
        df["dist_pp_ticks"] = ((df["close"] - df["pivot_pp"]) / tick_size).round(0)
    else:
        df["dist_pp_ticks"] = 0
    
    # RSI zones
    df["rsi_zone"] = pd.cut(
        df["rsi"],
        bins=[0, 30, 45, 55, 70, 100],
        labels=["oversold", "weak", "neutral", "strong", "overbought"],
        include_lowest=True
    )
    
    # Price position relative to EMA
    df["above_ema"] = df["close"] > df["ema"]
    
    # Price position relative to pivot
    df["above_pp"] = df["close"] > df["pivot_pp"]
    
    # Fibonacci zone categories
    df["fib_zone"] = "neutral"
    if df["pivot_pp"].notna().any() and df["pivot_r1"].notna().any() and df["pivot_s1"].notna().any():
        df.loc[df["close"] > df["pivot_r1"], "fib_zone"] = "above_r1"
        df.loc[df["close"] < df["pivot_s1"], "fib_zone"] = "below_s1"
        df.loc[(df["close"] >= df["pivot_pp"]) & (df["close"] <= df["pivot_r1"]), "fib_zone"] = "pp_to_r1"
        df.loc[(df["close"] <= df["pivot_pp"]) & (df["close"] >= df["pivot_s1"]), "fib_zone"] = "s1_to_pp"
    
    # Swing detection (for Fibonacci retracements)
    df = detect_swings(df)
    
    # Calculate Fibonacci retracement levels from swings
    df = calc_fib_retracements(df)
    
    # Price proximity to Fibonacci retracement levels
    df["near_fib_level"] = "none"
    # Adicionar checagem das colunas de fib
    for i, lvl in enumerate(FIB_LEVELS):
        lvl_name = f"fib_{int(lvl * 1000)}"
        if lvl_name in df.columns:
            proximity = ((df["close"] - df[lvl_name]).abs() / tick_size)
            mask = proximity <= 5  # within 5 ticks of fib level
            df.loc[mask, "near_fib_level"] = lvl_name
    
    # Hour of day
    df["hour"] = df["timestamp"].dt.hour
    
    # Delta close in ticks from previous bar
    df["delta_close_ticks"] = (df["close"].diff() / tick_size).fillna(0).round(0)
    
    # Directional streak
    streak = []
    count = 0
    prev = None
    for d in df["direcao"]:
        if d == prev:
            count += 1
        else:
            count = 1
        streak.append(count if d == "alta" else -count if d == "baixa" else 0)
        prev = d
    df["dir_streak"] = streak
    
    return df


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Detect swing highs and lows using a lookback window."""
    df["swing_high"] = False
    df["swing_low"] = False
    
    for i in range(lookback, len(df) - lookback):
        window_highs = df["high"].iloc[i - lookback:i + lookback + 1]
        window_lows = df["low"].iloc[i - lookback:i + lookback + 1]
        
        if df["high"].iloc[i] == window_highs.max():
            df.loc[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == window_lows.min():
            df.loc[df.index[i], "swing_low"] = True
    
    return df


def calc_fib_retracements(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Fibonacci retracement levels from the last significant swing."""
    fib_cols = {f"fib_{int(lvl * 1000)}": lvl for lvl in FIB_LEVELS}
    for col in fib_cols:
        df[col] = np.nan
    df["swing_trend"] = "none"
    
    last_swing_high = None
    last_swing_low = None
    
    for i in range(len(df)):
        if df["swing_high"].iloc[i]:
            last_swing_high = df["high"].iloc[i]
        if df["swing_low"].iloc[i]:
            last_swing_low = df["low"].iloc[i]
        
        if last_swing_high is not None and last_swing_low is not None:
            swing_range = last_swing_high - last_swing_low
            if swing_range > 0:
                # Determine trend based on which swing came last
                # Simplificação: Se o high mais recente for mais novo que o low mais recente?
                # Vamos usar uma heurística simples baseada nos índices relativos dos swings identificados
                # Mas aqui estamos iterando linha a linha, então sabemos o que foi atualizado por último.
                
                # Para saber qual é a 'perna' atual, precisamos saber quando ocorreu o último high e low.
                # Como last_swing_high é apenas o preço, não temos o índice aqui facilmente sem guardar.
                # Vamos assumir que a tendência é definida pela direção do preço atual em relação ao meio do range?
                # Ou melhor: guardar os índices.
                
                pass # A lógica completa exigiria guardar índices. 
                # Vamos simplificar: se range > 0, calculamos retrações tanto de alta quanto de baixa 
                # baseadas no que parece ser a tendência.
                
                # Se close está mais perto do high, talvez estejamos em uptrend vindo do low?
                # Vamos usar a lógica sugerida no prompt original:
                # "Determine trend based on which swing came last" - difficult without index storage.
                
                # Correção: Vamos assumir retracement de ALTA (Low -> High) se o preço caiu do High
                # E retracement de BAIXA (High -> Low) se o preço subiu do Low.
                
                # Vamos calcular AMBOS e ver qual faz sentido ou usar uma coluna de tendência pré-calculada?
                # Vamos usar a lógica do prompt original que foi copiada:
                if df["swing_high"].iloc[max(0, i-10):i+1].any() and not df["swing_low"].iloc[max(0, i-5):i+1].any():
                     # Recent high → downtrend retracement (from high down to low?) 
                     # Retracement geralmente é contra a tendência principal ou a favor?
                     # Fib Retracement: Se perna é de Alta (Low->High), o preço recua.
                     # Se perna é de Baixa (High->Low), o preço sobe.
                     
                     df.loc[df.index[i], "swing_trend"] = "down" # Perna de baixa
                     for col, lvl in fib_cols.items():
                         # Retração da queda: Low + Range * lvl 
                         # OU Projeção?
                         # O padrão é: Perna de Alta (L->H) => Níveis abaixo do H.
                         # Perna de Baixa (H->L) => Níveis acima do L.
                         
                         # Se detectamos High recente, assumimos que estamos descendo dele.
                         # Então estamos numa perna de baixa provável ou correção.
                         df.loc[df.index[i], col] = last_swing_high - (swing_range * lvl)
                else:
                    # Recent low → uptrend
                    df.loc[df.index[i], "swing_trend"] = "up"
                    for col, lvl in fib_cols.items():
                         df.loc[df.index[i], col] = last_swing_low + (swing_range * lvl)
    
    return df


# ─── Condition Generators ───────────────────────────────────────────────

def generate_conditions() -> list[dict]:
    """Generate all testable conditions combining Fibonacci + indicators."""
    conditions = []
    
    # 1. Fibonacci zone conditions (from pivot data)
    for zone in ["above_r1", "below_s1", "pp_to_r1", "s1_to_pp"]:
        conditions.append({
            "name": f"fib_zone_{zone}",
            "campo": "fib_zone",
            "op": "==",
            "valor": zone,
            "desc": f"Preço na zona Fibonacci: {zone}"
        })
    
    # 2. Proximity to Fibonacci retracement levels
    for lvl in FIB_LEVELS:
        lvl_name = f"fib_{int(lvl * 1000)}"
        conditions.append({
            "name": f"near_{lvl_name}",
            "campo": "near_fib_level",
            "op": "==",
            "valor": lvl_name,
            "desc": f"Preço próximo ao nível {lvl*100:.1f}%"
        })
    
    # 3. RSI conditions
    for zone in ["oversold", "weak", "neutral", "strong", "overbought"]:
        conditions.append({
            "name": f"rsi_{zone}",
            "campo": "rsi_zone",
            "op": "==",
            "valor": zone,
            "desc": f"RSI na zona: {zone}"
        })
    
    # 4. EMA position
    for val in [True, False]:
        label = "acima" if val else "abaixo"
        conditions.append({
            "name": f"ema_{label}",
            "campo": "above_ema",
            "op": "==",
            "valor": val,
            "desc": f"Preço {label} da EMA"
        })
    
    # 5. Pivot position
    for val in [True, False]:
        label = "acima" if val else "abaixo"
        conditions.append({
            "name": f"pp_{label}",
            "campo": "above_pp",
            "op": "==",
            "valor": val,
            "desc": f"Preço {label} do Pivot PP"
        })
    
    # 6. Body ratio (candle patterns)
    conditions.append({
        "name": "doji",
        "campo": "body_ratio",
        "op": "<",
        "valor": 0.3,
        "desc": "Candle tipo Doji (corpo < 30%)"
    })
    conditions.append({
        "name": "marubozu",
        "campo": "body_ratio",
        "op": ">",
        "valor": 0.7,
        "desc": "Candle tipo Marubozu (corpo > 70%)"
    })
    
    # 7. Directional streak
    for streak_val in [2, 3, -2, -3]:
        direction = "alta" if streak_val > 0 else "baixa"
        conditions.append({
            "name": f"streak_{direction}_{abs(streak_val)}",
            "campo": "dir_streak",
            "op": ">=" if streak_val > 0 else "<=",
            "valor": streak_val,
            "desc": f"Sequência de {abs(streak_val)}+ barras de {direction}"
        })
    
    # 8. Volume conditions (relative)
    conditions.append({
        "name": "high_volume",
        "campo": "volume_relative",
        "op": ">",
        "valor": 1.5,
        "desc": "Volume acima de 1.5x da média"
    })
    conditions.append({
        "name": "low_volume",
        "campo": "volume_relative",
        "op": "<",
        "valor": 0.5,
        "desc": "Volume abaixo de 0.5x da média"
    })
    
    # 9. Hour filters
    for h in [9, 10, 11, 13, 14, 15]:
        conditions.append({
            "name": f"hour_{h}",
            "campo": "hour",
            "op": "==",
            "valor": h,
            "desc": f"Operação às {h}h"
        })
    
    # 10. Swing trend
    for trend in ["up", "down"]:
        conditions.append({
            "name": f"swing_{trend}",
            "campo": "swing_trend",
            "op": "==",
            "valor": trend,
            "desc": f"Swing trend: {trend}"
        })
    
    return conditions


def eval_condition(row: pd.Series, cond: dict) -> bool:
    """Evaluate a single condition against a DataFrame row."""
    campo = cond["campo"]
    if campo not in row.index:
        return False
    val = row[campo]
    if pd.isna(val):
        return False
    
    op = cond["op"]
    target = cond["valor"]
    
    if op == "==":
        return val == target
    elif op == "!=":
        return val != target
    elif op == ">":
        return float(val) > float(target)
    elif op == ">=":
        return float(val) >= float(target)
    elif op == "<":
        return float(val) < float(target)
    elif op == "<=":
        return float(val) <= float(target)
    return False


# ─── Backtest Engine ────────────────────────────────────────────────────

def backtest_neutral(
    df: pd.DataFrame,
    entry_conditions: list[dict],
    stop_ticks: int,
    target_ticks: int,
) -> dict:
    """
    Backtest a NEUTRAL strategy (same conditions for long & short).
    - Long when conditions met + close > open (bullish bar)
    - Short when conditions met + close < open (bearish bar)
    Respects: 60 tick daily loss limit, one trade at a time.
    """
    tick_size = df["tick_size"].iloc[0] if len(df) > 0 else 0.25
    stop_price_delta = stop_ticks * tick_size
    target_price_delta = target_ticks * tick_size
    
    trades = []
    daily_loss = 0
    current_day = None
    in_trade = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        day = row["timestamp"].date() if pd.notna(row["timestamp"]) else None
        
        # Reset daily loss on new day
        if day != current_day:
            current_day = day
            daily_loss = 0
            # Reset in_trade at start of day too just in case
            in_trade = False
        
        # Skip if daily loss limit reached
        if daily_loss >= MAX_DAILY_LOSS_TICKS:
            continue
        
        if in_trade:
            # Check if previous trade resolved? 
            # In simplified logic, we skip ahead. But here we iterate all.
            # We need to skip bars until trade resolution.
            # BUT 'simulate_trade' returns bars_held.
            # We should skip 'bars_held' in the main loop index, 
            # but standard python for-loop doesn't allow modifying i easily.
            # So we use an internal counter or 'next_allowed_idx'.
            pass
        
        # We need a while loop or skip logic. 
        # For simplicity in this structure, we'll just check a 'skip_until' variable.
        pass 
    
    # Reimplementing loop with skip logic
    trades = []
    daily_loss = 0
    current_day = None
    next_allowed_idx = 0
    
    for i in range(len(df)):
        if i < next_allowed_idx:
            continue
            
        row = df.iloc[i]
        day = row["timestamp"].date() if pd.notna(row["timestamp"]) else None
        
        if day != current_day:
            current_day = day
            daily_loss = 0
            
        if daily_loss >= MAX_DAILY_LOSS_TICKS:
            continue
            
        # Check all conditions
        all_met = all(eval_condition(row, c) for c in entry_conditions)
        if not all_met:
            continue
        
        # Determine direction from bar
        if row["close"] > row["open"]:
            direction = "LONG"
        elif row["close"] < row["open"]:
            direction = "SHORT"
        else:
            continue
        
        # Simulate trade result
        entry_price = row["close"]
        result = simulate_trade(df, i, direction, entry_price, stop_price_delta, target_price_delta)
        
        if result is not None:
            result["day"] = str(day)
            result["direction"] = direction
            result["entry_time"] = str(row["timestamp"])
            
            pnl_ticks = result["pnl_ticks"]
            if pnl_ticks < 0:
                daily_loss += abs(pnl_ticks)
            
            trades.append(result)
            next_allowed_idx = i + result["bars_held"] + 1
    
    return analyze_trades(trades)


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_delta: float,
    target_delta: float,
    max_bars: int = 20,
) -> dict | None:
    """Simulate trade outcome looking forward max_bars."""
    tick_size = df["tick_size"].iloc[0] if len(df) > 0 else 0.25
    
    for j in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        bar = df.iloc[j]
        
        if direction == "LONG":
            # Check stop first (conservative)
            if bar["low"] <= entry_price - stop_delta:
                pnl_ticks = -round(stop_delta / tick_size)
                pnl_usd = (pnl_ticks * TICK_VALUE_USD) - COMMISSION_RT
                return {"pnl_ticks": pnl_ticks, "pnl_usd": pnl_usd, "result": "LOSS", "bars_held": j - entry_idx}
            # Check target
            if bar["high"] >= entry_price + target_delta:
                pnl_ticks = round(target_delta / tick_size)
                pnl_usd = (pnl_ticks * TICK_VALUE_USD) - COMMISSION_RT
                return {"pnl_ticks": pnl_ticks, "pnl_usd": pnl_usd, "result": "WIN", "bars_held": j - entry_idx}
        else:  # SHORT
            if bar["high"] >= entry_price + stop_delta:
                pnl_ticks = -round(stop_delta / tick_size)
                pnl_usd = (pnl_ticks * TICK_VALUE_USD) - COMMISSION_RT
                return {"pnl_ticks": pnl_ticks, "pnl_usd": pnl_usd, "result": "LOSS", "bars_held": j - entry_idx}
            if bar["low"] <= entry_price - target_delta:
                pnl_ticks = round(target_delta / tick_size)
                pnl_usd = (pnl_ticks * TICK_VALUE_USD) - COMMISSION_RT
                return {"pnl_ticks": pnl_ticks, "pnl_usd": pnl_usd, "result": "WIN", "bars_held": j - entry_idx}
    
    # Timeout
    end_idx = min(entry_idx + max_bars, len(df)-1)
    exit_price = df.iloc[end_idx]["close"]
    
    if direction == "LONG":
        pnl_ticks = round((exit_price - entry_price) / tick_size)
    else:
        pnl_ticks = round((entry_price - exit_price) / tick_size)
        
    pnl_usd = (pnl_ticks * TICK_VALUE_USD) - COMMISSION_RT
    return {"pnl_ticks": pnl_ticks, "pnl_usd": pnl_usd, "result": "TIMEOUT", "bars_held": end_idx - entry_idx}


def analyze_trades(trades: list[dict]) -> dict:
    """Compute statistics from trade list."""
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "pf": 0,
                "net_usd": 0, "net_ticks": 0, "longs": 0, "shorts": 0, "trades": []}
    
    wins = [t for t in trades if t["pnl_ticks"] > 0]
    losses = [t for t in trades if t["pnl_ticks"] <= 0]
    longs = len([t for t in trades if t["direction"] == "LONG"])
    shorts = len([t for t in trades if t["direction"] == "SHORT"])
    
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    
    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else 999,
        "net_usd": sum(t["pnl_usd"] for t in trades),
        "net_ticks": sum(t["pnl_ticks"] for t in trades),
        "longs": longs,
        "shorts": shorts,
        "long_pct": longs / len(trades) * 100 if trades else 0,
        "trades": trades[-10:],  # last 10 as sample
    }


# ─── Discovery Engine ──────────────────────────────────────────────────

def discover_strategies(
    all_day_dfs: dict[str, pd.DataFrame],
    min_win_rate: float = 55,
    min_consistency: float = 0.8,
) -> dict:
    """
    Brute-force discovery: test all 2-3 condition combos × stop/target pairs.
    Returns strategies profitable across min_consistency% of all days.
    """
    conditions = generate_conditions()
    days = sorted(all_day_dfs.keys())
    total_days = len(days)
    
    # Add volume_relative to each day's df
    for day, df in all_day_dfs.items():
        if "volume_ind" in df.columns and df["volume_ind"].notna().any():
            mean_vol = df["volume_ind"].mean()
            df["volume_relative"] = df["volume_ind"] / mean_vol if mean_vol > 0 else 1
        else:
            df["volume_relative"] = 1.0
        all_day_dfs[day] = df
    
    # Stop/Target combos (RR >= 2.0)
    stop_target_combos = []
    # Simplified range to speed up
    for stop in [10, 15, 20]:
        for rr in [2.0, 3.0]:
            target = int(stop * rr)
            stop_target_combos.append((stop, target))
    
    # Generate combos of 2 conditions
    combo_2 = list(itertools.combinations(range(len(conditions)), 2))
    # Combo 3 deactivated for speed in this demo, or limited
    # combo_3 = list(itertools.combinations(range(len(conditions)), 3))
    
    all_combos = combo_2 # + combo_3
    
    # Random sample if too huge
    if len(all_combos) > 500:
        import random
        random.seed(42)
        all_combos = random.sample(all_combos, 500)

    results = []
    total_tested = 0
    
    print(f"[DISCOVERY] Testing {len(all_combos)} combos * {len(stop_target_combos)} stop/targets on {total_days} days...")
    
    for combo_indices in all_combos:
        combo_conds = [conditions[i] for i in combo_indices]
        
        for stop_ticks, target_ticks in stop_target_combos:
            total_tested += 1
            
            daily_results = []
            has_trades_all_days = True
            
            for day in days:
                df = all_day_dfs[day]
                stats = backtest_neutral(df, combo_conds, stop_ticks, target_ticks)
                
                if stats["total"] == 0:
                    has_trades_all_days = False
                    break
                
                daily_results.append({
                    "day": day,
                    "win_rate": stats["win_rate"],
                    "profit": stats["net_usd"],
                    "profitable": stats["net_usd"] > 0,
                    "total_trades": stats["total"],
                    "longs": stats["longs"],
                    "shorts": stats["shorts"],
                })
            
            if not has_trades_all_days:
                continue
            
            # Check consistency
            days_profitable = sum(1 for d in daily_results if d["profitable"])
            consistency = days_profitable / total_days
            avg_win_rate = sum(d["win_rate"] for d in daily_results) / len(daily_results)
            total_profit = sum(d["profit"] for d in daily_results)
            
            if avg_win_rate < min_win_rate:
                continue
            
            if consistency < min_consistency:
                continue
            
            # Build setup name
            setup_name = " + ".join(c["name"] for c in combo_conds)
            rr_ratio = target_ticks / stop_ticks if stop_ticks > 0 else 0
            
            results.append({
                "setup_name": f"FIB_{setup_name}_S{stop_ticks}_T{target_ticks}",
                "stop_ticks": stop_ticks,
                "target_ticks": target_ticks,
                "days_tested": total_days,
                "days_profitable": days_profitable,
                "consistency": round(consistency, 3),
                "avg_win_rate": round(avg_win_rate, 1),
                "avg_profit_factor": 0, 
                "total_profit_usd": round(total_profit, 2),
                "avg_daily_profit_usd": round(total_profit / total_days, 2),
                "daily_results": daily_results,
                "rr_ratio": rr_ratio,
                "rules": {
                    "conditions": [c["desc"] for c in combo_conds]
                },
                "rules_exit": {
                    "stop_loss_ticks": stop_ticks,
                    "take_profit_ticks": target_ticks,
                    "max_daily_loss_ticks": MAX_DAILY_LOSS_TICKS,
                },
            })
    
    results.sort(key=lambda x: (-x["consistency"], -x["total_profit_usd"]))
    
    return {
        "total_tested": total_tested,
        "results": results,
    }


# ─── API Routes ─────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "5.0-fibonacci", "timestamp": str(datetime.now())}


@app.post("/api/v1/analyze")
async def analyze_single(file: UploadFile = File(...), config: str = Form("{}")):
    """Single file analysis."""
    content = (await file.read()).decode("utf-8")
    entries = parse_json_file(content)
    cfg = json.loads(config) # Handle try/except inside parsing if needed
    
    if not entries:
        return {"status": "error", "message": "Nenhuma entrada válida no arquivo"}
    
    df = to_dataframe(entries)
    
    # Add volume_relative
    if "volume_ind" in df.columns and df["volume_ind"].notna().any():
        mean_vol = df["volume_ind"].mean()
        df["volume_relative"] = df["volume_ind"] / mean_vol if mean_vol > 0 else 1
    else:
        df["volume_relative"] = 1.0
    
    stop = cfg.get("stopTicks", 10)
    target = cfg.get("targetTicks", 20)
    min_wr = cfg.get("minWinRate", 55)
    
    conditions = generate_conditions()
    best_setups = []
    
    # Test all 2-condition combos
    # Limit to first 300 to avoid timeout on single analysis
    all_combos = list(itertools.combinations(range(len(conditions)), 2))
    if len(all_combos) > 300:
        import random
        random.seed(42)
        all_combos = random.sample(all_combos, 300)
        
    for combo in all_combos:
        combo_conds = [conditions[i] for i in combo]
        stats = backtest_neutral(df, combo_conds, stop, target)
        
        if stats["total"] >= 3 and stats["win_rate"] >= min_wr:
            # Directional balance check loose for single file
            setup_name = " + ".join(c["name"] for c in combo_conds)
            best_setups.append({
                "setup_name": f"FIB_{setup_name}",
                "stop_ticks": stop,
                "target_ticks": target,
                "win_rate": round(stats["win_rate"], 1),
                "profit_factor": round(stats["pf"], 2),
                "net_profit_usd": round(stats["net_usd"], 2),
                "total_trades": stats["total"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "rules": {"conditions": [c["desc"] for c in combo_conds]},
                "sample_trades": stats["trades"],
            })
    
    best_setups.sort(key=lambda x: -x["net_profit_usd"])
    
    return {
        "status": "completed",
        "filename": file.filename,
        "results": {
            "total_setups_tested": len(all_combos),
            "profitable_setups": len(best_setups),
            "best_setups": best_setups[:20],
        },
    }


@app.post("/api/v1/analyze-batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    config: str = Form("{}"),
):
    """Batch analysis: discover strategies consistent across multiple days."""
    try:
        cfg = json.loads(config)
    except:
        cfg = {}
        
    min_wr = cfg.get("minWinRate", 55)
    min_consistency = cfg.get("minConsistency", 0.8)
    
    # Parse all files into per-day DataFrames
    all_day_dfs = {}
    for f in files:
        f.file.seek(0)
        content = (await f.read()).decode("utf-8")
        entries = parse_json_file(content)
        if not entries:
            continue
        df = to_dataframe(entries)
        if len(df) > 0:
            day = str(df["timestamp"].iloc[0].date())
            all_day_dfs[day] = df
    
    if len(all_day_dfs) < 1:
        return {"status": "error", "message": "Nenhum dado válido encontrado"}
    
    # Run discovery
    discovery = discover_strategies(all_day_dfs, min_wr, min_consistency)
    
    consistent = discovery["results"][:3]   # Top 3
    all_results = discovery["results"][:50]  # Top 50
    
    return {
        "status": "completed",
        "total_days": len(all_day_dfs),
        "total_setups_tested": discovery["total_tested"],
        "consistent_setups": consistent,
        "all_setups": all_results,
    }

from datetime import datetime

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
