"""
Moving Average Crossover Backtester
=====================================
Testet eine einfache Trendfolge-Strategie (MA-Crossover, optional mit RSI-Filter)
an historischen Kursdaten - OHNE echtes Geld zu riskieren.

Strategie-Logik:
- KAUF-Signal:  kurzer MA (z.B. 20 Tage) kreuzt von unten über langen MA (z.B. 50 Tage)
- VERKAUF-Signal: kurzer MA kreuzt von oben unter langen MA
- Optionaler RSI-Filter: kein Kauf, wenn RSI > 70 (überkauft)

So benutzt du es:
1. `python ma_crossover_backtest.py` ausführen
2. Ergebnisse (Kennzahlen + Chart) prüfen
3. Parameter unten in der `CONFIG`-Sektion anpassen und erneut testen
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG - hier Parameter anpassen und Strategie durchspielen
# ============================================================
TICKER = "BTC-USD"        # z.B. "BTC-USD", "ETH-USD", "AAPL", "EURUSD=X"
START_DATE = "2021-01-01"
END_DATE = None           # None = bis heute
SHORT_WINDOW = 20         # kurzer gleitender Durchschnitt (Tage)
LONG_WINDOW = 50          # langer gleitender Durchschnitt (Tage)
USE_RSI_FILTER = True     # RSI-Filter an/aus
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
STARTING_CAPITAL = 10_000 # fiktives Startkapital in EUR/USD
TRADING_FEE_PCT = 0.001   # 0.1% Gebühr pro Trade (realistisch für Exchanges)


def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker} gefunden. Ticker-Symbol prüfen.")
    # yfinance liefert manchmal MultiIndex-Spalten -> vereinheitlichen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_signals(df):
    df["MA_short"] = df["Close"].rolling(SHORT_WINDOW).mean()
    df["MA_long"] = df["Close"].rolling(LONG_WINDOW).mean()
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    df["signal"] = 0
    crossover_up = (df["MA_short"] > df["MA_long"]) & (df["MA_short"].shift(1) <= df["MA_long"].shift(1))
    crossover_down = (df["MA_short"] < df["MA_long"]) & (df["MA_short"].shift(1) >= df["MA_long"].shift(1))

    if USE_RSI_FILTER:
        crossover_up = crossover_up & (df["RSI"] < RSI_OVERBOUGHT)

    df.loc[crossover_up, "signal"] = 1    # Kaufsignal
    df.loc[crossover_down, "signal"] = -1  # Verkaufssignal
    return df


def run_backtest(df):
    capital = STARTING_CAPITAL
    position = 0  # Anzahl gehaltener Einheiten
    equity_curve = []
    trades = []
    entry_price = None

    for date, row in df.iterrows():
        price = row["Close"]

        if row["signal"] == 1 and position == 0:
            # Kaufen: gesamtes Kapital investieren
            fee = capital * TRADING_FEE_PCT
            position = (capital - fee) / price
            entry_price = price
            capital = 0
            trades.append({"date": date, "type": "BUY", "price": price})

        elif row["signal"] == -1 and position > 0:
            # Verkaufen: alles liquidieren
            proceeds = position * price
            fee = proceeds * TRADING_FEE_PCT
            capital = proceeds - fee
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({"date": date, "type": "SELL", "price": price, "pnl_pct": pnl_pct})
            position = 0

        current_equity = capital + (position * price)
        equity_curve.append(current_equity)

    df["equity"] = equity_curve
    return df, trades


def print_report(df, trades):
    final_equity = df["equity"].iloc[-1]
    total_return_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    buy_hold_return_pct = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    wins = [t for t in sell_trades if t["pnl_pct"] > 0]
    win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0

    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    print("=" * 55)
    print(f" BACKTEST REPORT: {TICKER}")
    print(f" Zeitraum: {df.index[0].date()} bis {df.index[-1].date()}")
    print(f" Strategie: MA({SHORT_WINDOW}/{LONG_WINDOW})"
          f"{' + RSI-Filter' if USE_RSI_FILTER else ''}")
    print("=" * 55)
    print(f" Startkapital:        {STARTING_CAPITAL:>12,.2f}")
    print(f" Endkapital:          {final_equity:>12,.2f}")
    print(f" Strategie-Rendite:   {total_return_pct:>11.2f}%")
    print(f" Buy & Hold Rendite:  {buy_hold_return_pct:>11.2f}%")
    print(f" Anzahl Trades:       {len(sell_trades):>12}")
    print(f" Trefferquote:        {win_rate:>11.2f}%")
    print(f" Max. Drawdown:       {max_drawdown:>11.2f}%")
    print("=" * 55)

    if total_return_pct < buy_hold_return_pct:
        print(" Hinweis: Diese Strategie schlägt Buy & Hold NICHT in diesem Zeitraum.")
    else:
        print(" Diese Strategie schlägt Buy & Hold in diesem Zeitraum.")
    print(" WICHTIG: Vergangene Performance garantiert keine zukünftigen Ergebnisse.")


def plot_results(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(df.index, df["Close"], label="Kurs", alpha=0.5, color="gray")
    axes[0].plot(df.index, df["MA_short"], label=f"MA{SHORT_WINDOW}", linewidth=1.2)
    axes[0].plot(df.index, df["MA_long"], label=f"MA{LONG_WINDOW}", linewidth=1.2)

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    axes[0].scatter(buys.index, buys["Close"], marker="^", color="green", s=80, label="Kauf", zorder=5)
    axes[0].scatter(sells.index, sells["Close"], marker="v", color="red", s=80, label="Verkauf", zorder=5)

    axes[0].set_title(f"{TICKER} - MA Crossover Strategie")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["equity"], label="Strategie-Equity", color="blue")
    buy_hold_equity = STARTING_CAPITAL * (df["Close"] / df["Close"].iloc[0])
    axes[1].plot(df.index, buy_hold_equity, label="Buy & Hold", color="orange", linestyle="--")
    axes[1].set_title("Portfolio-Entwicklung")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_ergebnis.png")
    plt.savefig(output_path, dpi=120)
    print(f"\nChart gespeichert: {output_path}")


if __name__ == "__main__":
    print(f"Lade Daten für {TICKER}...")
    data = fetch_data(TICKER, START_DATE, END_DATE)
    data = generate_signals(data)
    data, trade_log = run_backtest(data)
    print_report(data, trade_log)
    plot_results(data)
