"""
Multi-Asset Backtester: Struktur + Modell + Saisonalität
============================================================
Kombiniert drei Signal-Ebenen zu einer Entry-Entscheidung:

1. REGELBASIERT (Struktur):  Long-Einstieg bei bullischem Strukturbruch
                              (Schlusskurs > letztes Swing-High)
2. MODELLBASIERT:            Logistische Regression muss Aufwärtswahrscheinlichkeit
                              über einem Schwellwert bestätigen
3. SAISONALITÄT:             Aktuelle Stunde/Wochentag muss historisch
                              mindestens neutral bis positiv performt haben

Stop-Loss wird knapp UNTER dem letzten Swing-Low platziert (das "Ziel-Level
unteres Tief" aus der Anforderung - hier als Risiko-Invalidierung verwendet).
Take-Profit als Risk/Reward-Vielfaches des Stop-Abstands.

Läuft über MEHRERE Assets gleichzeitig, mit gleich gewichtetem Kapital-Anteil
pro Asset, und gibt am Ende einen konsolidierten Gewinn-Report aus.

WICHTIG: Dies ist ein Backtester (historische Simulation, kein echtes Geld).
Bevor diese Strategie live läuft, sollte sie hier ausführlich getestet und
verstanden werden.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]   # gleichzeitig gehandelte Assets
INTERVAL = "1h"                    # yfinance: '1h' nur für die letzten ~730 Tage verfügbar

# WICHTIG: yfinance begrenzt Stundendaten auf ca. 730 Tage Historie.
# START_DATE wird automatisch daran angepasst, damit der Backtest nicht
# durch ein zu weit zurückliegendes Datum leer zurückkommt.
import datetime
if INTERVAL in ("1h", "60m", "30m", "15m", "5m"):
    START_DATE = (datetime.datetime.now() - datetime.timedelta(days=700)).strftime("%Y-%m-%d")
else:
    START_DATE = "2023-01-01"      # bei Tages-/Wochenkerzen ('1d','1wk') ist deutlich mehr Historie möglich
END_DATE = None

SWING_WINDOW = 3                   # Kerzen links/rechts für Swing-Erkennung
MODEL_HORIZON = 5                  # Kerzen in die Zukunft, die das Modell vorhersagt
MODEL_MIN_PROBA = 0.55             # Modell muss mind. X% Aufwärtswahrscheinlichkeit zeigen
MODEL_RETRAIN_EVERY = 200          # alle X Kerzen wird das Modell neu trainiert (rollierend)
MODEL_TRAIN_WINDOW = 500           # wie viele historische Kerzen fürs Training genutzt werden

SEASONALITY_MIN_SCORE = -0.0005    # Mindest-Score, sonst wird das Signal gefiltert
                                    # (leicht negativ erlaubt, um nicht zu viele Signale wegzufiltern)

RISK_REWARD = 2.0                  # Take-Profit = RISK_REWARD * Stop-Abstand
STARTING_CAPITAL_TOTAL = 10_000    # wird gleichmäßig auf alle Assets aufgeteilt
TRADING_FEE_PCT = 0.001


# ============================================================
# BAUSTEIN 1: STRUKTUR
# ============================================================
def find_swing_points(df, window=SWING_WINDOW):
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        wh = highs[i - window:i + window + 1]
        wl = lows[i - window:i + window + 1]
        if highs[i] == wh.max() and np.argmax(wh) == window:
            is_swing_high[i] = True
        if lows[i] == wl.min() and np.argmin(wl) == window:
            is_swing_low[i] = True

    df = df.copy()
    df["swing_high"] = is_swing_high
    df["swing_low"] = is_swing_low
    return df


def detect_structure_breaks(df):
    df = df.copy()
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_highs, last_lows = [], []
    bos_up = np.zeros(len(df), dtype=bool)
    already_broken = False

    for i in range(len(df)):
        if not np.isnan(last_swing_high) and df["Close"].iloc[i] > last_swing_high and not already_broken:
            bos_up[i] = True
            already_broken = True
        if df["swing_high"].iloc[i]:
            last_swing_high = df["High"].iloc[i]
            already_broken = False
        if df["swing_low"].iloc[i]:
            last_swing_low = df["Low"].iloc[i]
        last_highs.append(last_swing_high)
        last_lows.append(last_swing_low)

    df["last_swing_high"] = last_highs
    df["last_swing_low"] = last_lows
    df["bos_up"] = bos_up
    return df


# ============================================================
# BAUSTEIN 2: MODELL
# ============================================================
FEATURE_COLUMNS = ["ret_1", "ret_3", "ret_10", "volatility_10", "dist_ma20", "vol_change", "rsi"]


def build_features(df):
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_10"] = df["Close"].pct_change(10)
    df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
    df["dist_ma20"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    df["vol_change"] = df["Volume"].pct_change(5) if "Volume" in df.columns else 0.0
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # Division durch Null (z.B. Volumen=0 in ruhigen Marktphasen) kann +/-inf erzeugen.
    # dropna() erfasst kein inf, deshalb hier explizit zu NaN konvertieren -
    # betroffene Zeilen werden dann sauber rausgefiltert statt das Modell zu crashen.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return df


def build_labels(df, horizon=MODEL_HORIZON):
    df = df.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["label_up"] = (df["future_return"] > 0).astype(int)
    return df


class SignalModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, df_slice):
        data = build_labels(df_slice, horizon=MODEL_HORIZON)
        data = data.replace([np.inf, -np.inf], np.nan)  # Sicherheitsnetz, falls inf durchrutscht
        data = data.dropna(subset=FEATURE_COLUMNS + ["label_up"])
        if len(data) < 50 or data["label_up"].nunique() < 2:
            self.is_trained = False
            return False
        X = self.scaler.fit_transform(data[FEATURE_COLUMNS].values)
        self.model.fit(X, data["label_up"].values)
        self.is_trained = True
        return True

    def predict_proba_up(self, feature_row):
        if not self.is_trained:
            return 0.5
        X = feature_row[FEATURE_COLUMNS].astype(float).values.reshape(1, -1)
        if np.isnan(X).any():
            return 0.5
        return self.model.predict_proba(self.scaler.transform(X))[0][1]


# ============================================================
# BAUSTEIN 3: SAISONALITÄT
# ============================================================
def compute_seasonality(df, horizon=MODEL_HORIZON):
    data = df.copy()
    data["future_return"] = data["Close"].shift(-horizon) / data["Close"] - 1
    data["weekday"] = data.index.dayofweek
    data["hour"] = data.index.hour
    weekday_perf = data.groupby("weekday")["future_return"].mean()
    hour_perf = data.groupby("hour")["future_return"].mean()
    return weekday_perf, hour_perf


def seasonality_score(timestamp, weekday_perf, hour_perf):
    wd_score = weekday_perf.get(timestamp.dayofweek, 0.0)
    hr_score = hour_perf.get(timestamp.hour, 0.0)
    return (wd_score + hr_score) / 2


# ============================================================
# BACKTEST-ENGINE (pro Asset)
# ============================================================
def run_single_asset_backtest(ticker, capital_allocated):
    raw = yf.download(ticker, start=START_DATE, end=END_DATE, interval=INTERVAL, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = find_swing_points(raw)
    df = detect_structure_breaks(df)
    df = build_features(df)

    weekday_perf, hour_perf = compute_seasonality(df)

    model = SignalModel()
    capital = capital_allocated
    position = 0
    entry_price = None
    stop_loss = None
    take_profit = None
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["Close"]

        # Modell periodisch neu trainieren (nur auf Daten VOR dem aktuellen Punkt - kein Lookahead)
        if i > 0 and i % MODEL_RETRAIN_EVERY == 0 and i >= MODEL_TRAIN_WINDOW:
            train_slice = df.iloc[max(0, i - MODEL_TRAIN_WINDOW):i]
            model.train(train_slice)

        # Offene Position verwalten: Stop-Loss / Take-Profit prüfen
        if position > 0:
            if price <= stop_loss:
                proceeds = position * stop_loss * (1 - TRADING_FEE_PCT)
                pnl_pct = (stop_loss - entry_price) / entry_price * 100
                trades.append({"ticker": ticker, "exit_time": df.index[i], "type": "STOP_LOSS",
                                "entry": entry_price, "exit": stop_loss, "pnl_pct": pnl_pct})
                capital = proceeds
                position = 0
            elif price >= take_profit:
                proceeds = position * take_profit * (1 - TRADING_FEE_PCT)
                pnl_pct = (take_profit - entry_price) / entry_price * 100
                trades.append({"ticker": ticker, "exit_time": df.index[i], "type": "TAKE_PROFIT",
                                "entry": entry_price, "exit": take_profit, "pnl_pct": pnl_pct})
                capital = proceeds
                position = 0

        # Neuen Einstieg prüfen (nur wenn aktuell flat)
        elif row["bos_up"] and not np.isnan(row["last_swing_low"]):
            proba_up = model.predict_proba_up(row)
            season_score = seasonality_score(df.index[i], weekday_perf, hour_perf)

            if proba_up >= MODEL_MIN_PROBA and season_score >= SEASONALITY_MIN_SCORE:
                stop_candidate = row["last_swing_low"] * 0.999  # kleiner Puffer unter dem Tief
                risk = price - stop_candidate
                if risk > 0:
                    entry_price = price
                    stop_loss = stop_candidate
                    take_profit = price + risk * RISK_REWARD
                    fee = capital * TRADING_FEE_PCT
                    position = (capital - fee) / price
                    capital = 0
                    trades.append({"ticker": ticker, "entry_time": df.index[i], "type": "ENTRY",
                                    "entry": entry_price, "stop": stop_loss, "target": take_profit,
                                    "model_proba": proba_up, "season_score": season_score})

        current_equity = capital + (position * price if position > 0 else 0)
        equity_curve.append(current_equity)

    df["equity"] = equity_curve
    final_equity = df["equity"].iloc[-1]
    buy_hold_equity = capital_allocated * (df["Close"] / df["Close"].iloc[0])

    return {
        "ticker": ticker,
        "df": df,
        "trades": trades,
        "final_equity": final_equity,
        "starting_capital": capital_allocated,
        "buy_hold_final": buy_hold_equity.iloc[-1],
    }


# ============================================================
# MULTI-ASSET REPORT
# ============================================================
def run_portfolio_backtest():
    capital_per_asset = STARTING_CAPITAL_TOTAL / len(TICKERS)
    results = []

    for ticker in TICKERS:
        print(f"Backteste {ticker}...")
        result = run_single_asset_backtest(ticker, capital_per_asset)
        if result is not None:
            results.append(result)
        else:
            print(f"  -> Keine Daten für {ticker}, übersprungen.")

    print_portfolio_report(results)
    if results:
        plot_portfolio(results)
    else:
        print("\nKein Chart erstellt (keine Daten vorhanden).")
    return results


def print_portfolio_report(results):
    print("\n" + "=" * 70)
    print(" MULTI-ASSET PORTFOLIO REPORT")
    print(" Strategie: Strukturbruch + Modell-Bestätigung + Saisonalitäts-Filter")
    print("=" * 70)

    if not results:
        print("\n FEHLER: Für keinen der Ticker konnten Daten geladen werden.")
        print(" Mögliche Ursachen:")
        print("  - START_DATE liegt außerhalb des von yfinance erlaubten Zeitraums für dieses INTERVAL")
        print("  - Ticker-Symbol falsch geschrieben (z.B. 'BTC-USD' statt 'BTC/USD')")
        print("  - Kein Internetzugang / yfinance-Dienst gerade nicht erreichbar")
        print("=" * 70)
        return

    total_start = 0
    total_end = 0
    total_bh_end = 0

    for r in results:
        entries = [t for t in r["trades"] if t["type"] == "ENTRY"]
        exits = [t for t in r["trades"] if t["type"] in ("STOP_LOSS", "TAKE_PROFIT")]
        wins = [t for t in exits if t["pnl_pct"] > 0]
        win_rate = (len(wins) / len(exits) * 100) if exits else 0
        ret_pct = (r["final_equity"] - r["starting_capital"]) / r["starting_capital"] * 100
        bh_ret_pct = (r["buy_hold_final"] - r["starting_capital"]) / r["starting_capital"] * 100

        print(f"\n{r['ticker']}")
        print(f"  Trades: {len(entries)} Einstiege, {len(exits)} Ausstiege | Trefferquote: {win_rate:.1f}%")
        print(f"  Strategie-Rendite: {ret_pct:+.2f}%  |  Buy&Hold: {bh_ret_pct:+.2f}%")
        print(f"  Kapital: {r['starting_capital']:,.0f} -> {r['final_equity']:,.0f}")

        total_start += r["starting_capital"]
        total_end += r["final_equity"]
        total_bh_end += r["buy_hold_final"]

    print("\n" + "-" * 70)
    total_ret = (total_end - total_start) / total_start * 100
    total_bh_ret = (total_bh_end - total_start) / total_start * 100
    print(f" GESAMT-PORTFOLIO ({len(results)} Assets)")
    print(f"  Kapital: {total_start:,.0f} -> {total_end:,.0f}  ({total_ret:+.2f}%)")
    print(f"  Buy & Hold Vergleich: {total_bh_ret:+.2f}%")
    print("=" * 70)
    if total_ret < total_bh_ret:
        print(" Hinweis: Portfolio schlägt einfaches Buy&Hold NICHT in diesem Zeitraum.")
    else:
        print(" Portfolio schlägt einfaches Buy&Hold in diesem Zeitraum.")


def plot_portfolio(results):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        df = r["df"]
        ax.plot(df.index, df["equity"], label=f"{r['ticker']} Strategie", color="#3FB950")
        bh = r["starting_capital"] * (df["Close"] / df["Close"].iloc[0])
        ax.plot(df.index, bh, label="Buy & Hold", color="#6E7A8A", linestyle="--")

        entries = [t for t in r["trades"] if t["type"] == "ENTRY"]
        if entries:
            entry_times = [t["entry_time"] for t in entries if t["entry_time"] in df.index]
            entry_equity = [df.loc[t, "equity"] for t in entry_times]
            ax.scatter(entry_times, entry_equity, marker="^", color="#3FB950", s=60, zorder=5, label="Einstieg")

        ax.set_title(f"{r['ticker']} - Struktur+Modell+Saisonalität Strategie")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("multi_asset_backtest.png", dpi=110)
    print("\nChart gespeichert: multi_asset_backtest.png")


if __name__ == "__main__":
    run_portfolio_backtest()
