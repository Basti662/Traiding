"""
Exchange-Trading-Bot: Struktur + Modell + Saisonalität (Binance Testnet)
==========================================================================
Live-Version der Strategie aus multi_strategy_backtest.py, gebaut auf demselben
Grundgerüst wie exchange_bot.py (ccxt, Flask-Dashboard, State-Persistenz).

Anders als eine einfache "ein Markt zur Zeit"-Version handelt dieser Bot ALLE
in SYMBOLS gelisteten Märkte GLEICHZEITIG - jeder mit eigenem Modell, eigener
Position, eigenem Stop-Loss/Take-Profit. Das Dashboard zeigt echte Candlestick-
Charts (kein Ersatz-Linienchart) sowie eine Übersichtstabelle über alle Märkte,
und erlaubt es, im laufenden Betrieb Märkte hinzuzufügen oder zu entfernen.

STRATEGIE (siehe multi_strategy_backtest.py für die Backtest-Herleitung):
1. STRUKTUR:      Long-Einstieg nur bei bullischem Strukturbruch
                   (Schlusskurs > letztes Swing-High)
2. MODELL:        Logistische Regression muss Aufwärtswahrscheinlichkeit
                   über MODEL_MIN_PROBA bestätigen
3. SAISONALITÄT:  Aktuelle Stunde/Wochentag muss historisch mind. neutral
                   bis positiv performt haben

Stop-Loss = knapp unter letztem Swing-Low. Take-Profit = Entry + RISK_REWARD *
Stop-Abstand. Das sind FESTE Preis-Levels, die bei Entry berechnet und bis zum
Exit gehalten werden (kein Nachziehen).

WICHTIGE SICHERHEITSHINWEISE (identisch zu exchange_bot.py):
- USE_TESTNET = True ist der Standard. Erst auf False stellen, wenn du die
  Strategie wochenlang auf Testnet beobachtet hast UND die Ergebnisse überzeugen.
- API Keys NIEMALS im Code hardcoden. Sie kommen aus einer .env-Datei.
- Erstelle Keys NUR mit Handelsrechten, KEINEN Abhebe-Rechten (Withdrawal).
- MAX_POSITION_SIZE_PCT begrenzt, wie viel % des FREIEN Guthabens pro Trade
  riskiert wird. Bei mehreren gleichzeitigen Märkten wird das Guthaben bei
  jedem Kauf frisch abgefragt, d.h. spätere Käufe in derselben Runde bekommen
  automatisch weniger Kapital zugeteilt als frühere. Trotzdem gilt: je mehr
  Märkte gleichzeitig aktiv sind, desto mehr Positionen können theoretisch
  gleichzeitig offen sein - plane dein Guthaben entsprechend.
- Das Modell wird NUR auf Vergangenheitsdaten trainiert (kein Lookahead), aber
  eine Logistische Regression auf ~7 einfachen Features ist kein Freifahrtschein -
  behandle proba_up als schwachen Zusatzfilter, nicht als Vorhersage.

SETUP:
1. pip install ccxt python-dotenv pandas numpy scikit-learn flask
2. Datei ".env" im selben Ordner erstellen mit:
     BINANCE_API_KEY=dein_testnet_api_key
     BINANCE_API_SECRET=dein_testnet_secret
   (kann dieselbe .env wie exchange_bot.py sein, Variablennamen sind identisch)
3. python structure_model_bot.py
   -> Dashboard läuft auf einem ANDEREN Port als exchange_bot.py (5001 statt 5000),
      damit beide Bots gleichzeitig laufen können.
"""

import os
import time
import json
import logging
import threading
from collections import deque
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

USE_TESTNET = True                 # NIEMALS ohne wochenlange Testnet-Erfahrung auf False stellen

SYMBOLS = ["DOGE/USDT", "BTC/USDT", "ETH/USDT"]   # Märkte, die GLEICHZEITIG gehandelt werden
QUOTE_CURRENCY = "USDT"            # muss für alle SYMBOLS identisch sein (Guthabens-Tracking geht davon aus)
TIMEFRAME = "1h"                   # gilt für alle Märkte gemeinsam (wie im Backtest, INTERVAL = "1h")

SWING_WINDOW = 3                   # Kerzen links/rechts für Swing-Erkennung
MODEL_HORIZON = 5                  # Kerzen in die Zukunft, die das Modell "gelernt" hat vorherzusagen
MODEL_MIN_PROBA = 0.55             # Modell muss mind. X% Aufwärtswahrscheinlichkeit zeigen
MODEL_TRAIN_WINDOW = 500           # wie viele historische Kerzen fürs Training genutzt werden
MODEL_RETRAIN_EVERY_CHECKS = 30    # alle X Loop-Durchläufe wird das Modell (pro Symbol!) neu trainiert

SEASONALITY_MIN_SCORE = -0.0005    # Mindest-Score, sonst wird das Signal gefiltert

RISK_REWARD = 2.0                  # Take-Profit = RISK_REWARD * Stop-Abstand
SWING_LOW_BUFFER_PCT = 0.001       # Stop-Loss 0.1% unter dem Swing-Low (wie im Backtest)

OHLCV_FETCH_LIMIT = 700            # genug für Trainingsfenster + Swing-Lookback + Rolling-Features
MAX_POSITION_SIZE_PCT = 0.10       # max. 10% des FREIEN Guthabens pro Trade (siehe Hinweis oben)
CHECK_INTERVAL_SECONDS = 60        # wie oft der Bot pro Runde ALLE Märkte durchgeht

STATE_FILE = "bot_state_structure_model.json"    # State für ALLE Märkte, als {symbol: {...}}
LOG_FILE = "bot_log_structure_model.txt"

DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = 5001               # ANDERER Port als exchange_bot.py (5000), damit beide parallel laufen
HISTORY_MAXLEN = 300
CANDLE_MAXLEN = 150
LOG_TAIL_MAXLEN = 80
TRADES_MAXLEN = 150

MIN_CANDLES_REQUIRED = MODEL_TRAIN_WINDOW + 2 * SWING_WINDOW + 20

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("structure_model_bot")


# ============================================================
# SHARED STATE FÜR DAS DASHBOARD (thread-sicher)
# ============================================================
def _empty_market_entry(symbol):
    return {
        "symbol": symbol,
        "price": None,
        "bos_up": False,
        "last_swing_high": None,
        "last_swing_low": None,
        "model_proba": None,
        "model_trained": False,
        "season_score": None,
        "signal": None,
        "in_position": False,
        "entry_price": None,
        "position_amount": None,
        "stop_loss": None,
        "take_profit": None,
        "pnl_pct": None,
        "base_currency": symbol.split("/")[0],
        "base_free": None,
        "base_total": None,
        "last_update": None,
        "candles": deque(maxlen=CANDLE_MAXLEN),
        "error": None,
    }


class DashboardState:
    """Hält alle Daten, die das Web-Dashboard anzeigt. Wird vom Bot-Thread
    beschrieben und vom Flask-Thread gelesen. Ein Eintrag pro gehandeltem Markt."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "testnet": USE_TESTNET,
            "strategy": "Struktur + Modell + Saisonalität",
            "timeframe": TIMEFRAME,
            "quote_currency": QUOTE_CURRENCY,
            "quote_free": None,
            "quote_total": None,
            "last_update": None,
            "markets": {},   # symbol -> market entry dict (siehe _empty_market_entry)
            "trades": deque(maxlen=TRADES_MAXLEN),
            "log_tail": deque(maxlen=LOG_TAIL_MAXLEN),
        }

    def ensure_market(self, symbol):
        with self.lock:
            if symbol not in self.data["markets"]:
                self.data["markets"][symbol] = _empty_market_entry(symbol)

    def remove_market(self, symbol):
        with self.lock:
            self.data["markets"].pop(symbol, None)

    def set_market_error(self, symbol, message):
        with self.lock:
            if symbol in self.data["markets"]:
                self.data["markets"][symbol]["error"] = message

    def update_market_tick(self, symbol, price, row, model_proba, model_trained, season_score, signal, state):
        with self.lock:
            m = self.data["markets"].get(symbol)
            if m is None:
                return
            m["price"] = price_round(price)
            m["bos_up"] = bool(row.get("bos_up", False))
            lsh, lsl = row.get("last_swing_high"), row.get("last_swing_low")
            m["last_swing_high"] = price_round(lsh) if (lsh is not None and lsh == lsh) else None
            m["last_swing_low"] = price_round(lsl) if (lsl is not None and lsl == lsl) else None
            m["model_proba"] = round(float(model_proba), 3) if model_proba is not None else None
            m["model_trained"] = model_trained
            m["season_score"] = round(float(season_score), 6) if season_score is not None else None
            m["signal"] = signal
            m["in_position"] = state["in_position"]
            m["entry_price"] = state.get("entry_price")
            m["position_amount"] = state.get("amount")
            m["stop_loss"] = state.get("stop_loss")
            m["take_profit"] = state.get("take_profit")
            if state["in_position"] and state.get("entry_price"):
                m["pnl_pct"] = round((price - state["entry_price"]) / state["entry_price"] * 100, 2)
            else:
                m["pnl_pct"] = None
            m["last_update"] = datetime.now().strftime("%H:%M:%S")
            m["error"] = None
            self.data["last_update"] = m["last_update"]

    def update_market_candles(self, symbol, df):
        with self.lock:
            m = self.data["markets"].get(symbol)
            if m is None:
                return
            recent = df.tail(CANDLE_MAXLEN)
            m["candles"] = deque(
                (
                    {
                        "t": int(ts.timestamp() * 1000),
                        "o": price_round(row["Open"]),
                        "h": price_round(row["High"]),
                        "l": price_round(row["Low"]),
                        "c": price_round(row["Close"]),
                    }
                    for ts, row in recent.iterrows()
                ),
                maxlen=CANDLE_MAXLEN,
            )

    def update_quote_balance(self, quote_bal):
        with self.lock:
            self.data["quote_free"] = round(quote_bal.get("free", 0), 2) if quote_bal else None
            self.data["quote_total"] = round(quote_bal.get("total", 0), 2) if quote_bal else None

    def update_market_balance(self, symbol, base_bal):
        with self.lock:
            m = self.data["markets"].get(symbol)
            if m is None:
                return
            m["base_free"] = round(base_bal.get("free", 0), 8) if base_bal else None
            m["base_total"] = round(base_bal.get("total", 0), 8) if base_bal else None

    def add_trade(self, symbol, trade_type, price, pnl_pct, reason):
        with self.lock:
            self.data["trades"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "symbol": symbol,
                "type": trade_type,
                "price": price_round(price),
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                "reason": reason,
            })

    def add_log(self, message):
        with self.lock:
            self.data["log_tail"].append(message)

    def snapshot(self):
        with self.lock:
            markets_out = {}
            for sym, m in self.data["markets"].items():
                markets_out[sym] = {**{k: v for k, v in m.items() if k != "candles"},
                                     "candles": list(m["candles"])}
            return {
                **{k: v for k, v in self.data.items() if k not in ("markets", "trades", "log_tail")},
                "markets": markets_out,
                "trades": list(self.data["trades"]),
                "log_tail": list(self.data["log_tail"]),
            }


dashboard = DashboardState()


class DashboardLogHandler(logging.Handler):
    """Spiegelt jede Log-Zeile zusätzlich ins Dashboard."""
    def emit(self, record):
        dashboard.add_log(self.format(record))


dashboard_handler = DashboardLogHandler()
dashboard_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
log.addHandler(dashboard_handler)


# ============================================================
# SYMBOL-REGISTRY - erlaubt Märkte im laufenden Betrieb über das Dashboard
# hinzuzufügen/zu entfernen, ohne den Prozess neu zu starten.
# ============================================================
MARKET_PRESETS = ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "SOL/USDT", "XRP/USDT",
                   "ADA/USDT", "BNB/USDT", "TRX/USDT", "LTC/USDT", "MATIC/USDT"]


class SymbolRegistry:
    def __init__(self, symbols):
        self.lock = threading.Lock()
        self.symbols = list(dict.fromkeys(symbols))  # Reihenfolge erhalten, Duplikate raus

    def get_all(self):
        with self.lock:
            return list(self.symbols)

    def add(self, symbol):
        with self.lock:
            if symbol in self.symbols:
                return False
            self.symbols.append(symbol)
            return True

    def remove(self, symbol):
        with self.lock:
            if symbol not in self.symbols:
                return False
            self.symbols.remove(symbol)
            return True


registry = SymbolRegistry(SYMBOLS)
_exchange_ref = None  # wird in run_bot() gesetzt, damit API-Routen Märkte validieren können


def price_round(value):
    """Rundet abhängig von der Preisgröße - verhindert, dass Coins unter 1 USD
    (z.B. DOGE bei 0.07) auf 0.07 == 0.07 == 0.07 kollabieren und Kerzen verschwinden."""
    value = float(value)
    if value == 0:
        return 0.0
    magnitude = abs(value)
    if magnitude >= 100:
        return round(value, 2)
    elif magnitude >= 1:
        return round(value, 4)
    else:
        return round(value, 8)


# ============================================================
# FLASK DASHBOARD-SERVER
# ============================================================
app = Flask(__name__)


@app.after_request
def disable_caching(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template("index.html", timeframe=TIMEFRAME, symbols=registry.get_all())


@app.route("/api/data")
def api_data():
    return jsonify(dashboard.snapshot())


@app.route("/api/presets")
def api_presets():
    active = registry.get_all()
    return jsonify({
        "markets": [m for m in MARKET_PRESETS if m not in active],
        "timeframe": TIMEFRAME,
    })


@app.route("/api/add_market", methods=["POST"])
def api_add_market():
    payload = request.get_json(silent=True) or {}
    new_symbol = (payload.get("symbol") or "").strip().upper()

    if not new_symbol or "/" not in new_symbol:
        return jsonify({"ok": False, "error": "Ungültiges Symbol-Format (erwartet z.B. BTC/USDT)."}), 400
    if not new_symbol.endswith("/" + QUOTE_CURRENCY):
        return jsonify({"ok": False, "error": f"Nur {QUOTE_CURRENCY}-Paare werden unterstützt (Guthabens-Tracking geht von einer gemeinsamen Quote-Währung aus)."}), 400

    if _exchange_ref is not None:
        try:
            markets = _exchange_ref.load_markets()
            if new_symbol not in markets:
                return jsonify({"ok": False, "error": f"'{new_symbol}' ist auf dieser Exchange nicht verfügbar."}), 400
        except Exception as e:
            log.warning(f"Marktvalidierung beim Hinzufügen fehlgeschlagen (wird trotzdem versucht): {e}")

    added = registry.add(new_symbol)
    if not added:
        return jsonify({"ok": False, "error": f"{new_symbol} wird bereits gehandelt."}), 409

    dashboard.ensure_market(new_symbol)
    log.info(f"Markt hinzugefügt: {new_symbol} (Timeframe: {TIMEFRAME}) - wird ab der nächsten Runde gehandelt.")
    return jsonify({"ok": True, "symbol": new_symbol})


@app.route("/api/remove_market", methods=["POST"])
def api_remove_market():
    payload = request.get_json(silent=True) or {}
    symbol = (payload.get("symbol") or "").strip().upper()

    snapshot = dashboard.snapshot()
    market = snapshot["markets"].get(symbol)
    if market and market.get("in_position"):
        return jsonify({"ok": False, "error": f"{symbol} hat aktuell eine offene Position - erst schließen, bevor der Markt entfernt wird."}), 409

    removed = registry.remove(symbol)
    if not removed:
        return jsonify({"ok": False, "error": f"{symbol} wird aktuell nicht gehandelt."}), 404

    dashboard.remove_market(symbol)
    state_by_symbol.pop(symbol, None)
    model_by_symbol.pop(symbol, None)
    save_state(state_by_symbol)
    log.info(f"Markt entfernt: {symbol}")
    return jsonify({"ok": True, "symbol": symbol})


def run_dashboard():
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)


# ============================================================
# EXCHANGE VERBINDUNG
# ============================================================
def create_exchange():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise EnvironmentError(
            "BINANCE_API_KEY / BINANCE_API_SECRET nicht gefunden. "
            "Prüfe deine .env-Datei im selben Ordner wie dieses Script."
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "recvWindow": 10000,
        },
    })

    if USE_TESTNET:
        exchange.set_sandbox_mode(True)
        log.info("TESTNET-Modus aktiv - es wird mit Fake-Geld gehandelt.")
    else:
        log.warning("!!! LIVE-MODUS AKTIV - ECHTES GELD WIRD GEHANDELT !!!")

    try:
        exchange.load_time_difference()
        log.info(f"Zeit-Offset zu Binance-Servern synchronisiert: {exchange.options.get('timeDifference', 0)} ms")
    except Exception as e:
        log.warning(f"Zeit-Synchronisation fehlgeschlagen (nicht kritisch, wird bei jedem Request neu versucht): {e}")

    return exchange


def fetch_ohlcv_df(exchange, symbol, timeframe, limit=OHLCV_FETCH_LIMIT):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ============================================================
# STRATEGIE-BAUSTEIN 1: STRUKTUR (unverändert aus multi_strategy_backtest.py)
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
# STRATEGIE-BAUSTEIN 2: MODELL (unverändert aus multi_strategy_backtest.py)
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
        data = data.replace([np.inf, -np.inf], np.nan)
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


class ModelState:
    """Hält das trainierte Modell + Retrain-Zähler EINES Symbols über Loop-
    Durchläufe hinweg (im Backtest wurde pro verarbeiteter Kerze gezählt, live
    zählen wir Loop-Checks). Jedes Symbol bekommt eine eigene Instanz, weil sich
    Muster zwischen Assets stark unterscheiden."""

    def __init__(self):
        self.model = SignalModel()
        self.checks_since_train = MODEL_RETRAIN_EVERY_CHECKS  # sofort beim ersten Lauf trainieren

    def maybe_retrain(self, df, symbol=""):
        self.checks_since_train += 1
        if self.checks_since_train < MODEL_RETRAIN_EVERY_CHECKS:
            return
        self.checks_since_train = 0
        train_slice = df.tail(MODEL_TRAIN_WINDOW)
        ok = self.model.train(train_slice)
        if ok:
            log.info(f"[{symbol}] Modell neu trainiert auf den letzten {len(train_slice)} Kerzen.")
        else:
            log.warning(f"[{symbol}] Modell-Training übersprungen (zu wenig/zu einseitige Daten).")


model_by_symbol = {}   # symbol -> ModelState()


def get_or_create_model(symbol):
    if symbol not in model_by_symbol:
        model_by_symbol[symbol] = ModelState()
    return model_by_symbol[symbol]


# ============================================================
# STRATEGIE-BAUSTEIN 3: SAISONALITÄT (unverändert aus multi_strategy_backtest.py)
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
# STATE (Position + Stop/Target werden lokal gespeichert, übersteht Neustarts)
# state_by_symbol hat die Form {symbol: {"in_position":..., "entry_price":..., ...}}
# ============================================================
def default_symbol_state():
    return {"in_position": False, "entry_price": None, "amount": None,
            "stop_loss": None, "take_profit": None}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(all_state):
    with open(STATE_FILE, "w") as f:
        json.dump(all_state, f, indent=2)


state_by_symbol = {}   # wird in run_bot() aus STATE_FILE befüllt


def ensure_state(symbol):
    if symbol not in state_by_symbol:
        state_by_symbol[symbol] = default_symbol_state()
    return state_by_symbol[symbol]


# ============================================================
# ORDER-AUSFÜHRUNG
# ============================================================
def get_quote_balance(exchange, quote_currency=QUOTE_CURRENCY):
    balance = exchange.fetch_balance()
    return balance.get(quote_currency, {}).get("free", 0)


def place_buy_order(exchange, symbol, price, stop_loss, take_profit, model_proba, season_score, state):
    quote_balance = get_quote_balance(exchange)
    spend_amount = quote_balance * MAX_POSITION_SIZE_PCT

    if spend_amount < 10:  # Binance Mindestordergröße grob berücksichtigen
        log.warning(f"[{symbol}] Zu wenig Guthaben für sinnvollen Trade ({spend_amount:.2f} {QUOTE_CURRENCY}). Übersprungen.")
        return state

    base_amount = spend_amount / price
    try:
        order = exchange.create_market_buy_order(symbol, base_amount)
        log.info(
            f"[{symbol}] KAUF ausgeführt: {base_amount:.6f} @ ~{price_round(price)} (Order-ID: {order['id']}) | "
            f"Stop: {price_round(stop_loss)} | Target: {price_round(take_profit)} | "
            f"Modell-Proba: {model_proba:.2f} | Saison-Score: {season_score:.5f}"
        )
        state.update({
            "in_position": True, "entry_price": price, "amount": base_amount,
            "stop_loss": stop_loss, "take_profit": take_profit,
        })
        dashboard.add_trade(symbol, "BUY", price, None, f"BOS+Modell({model_proba:.2f})+Saison")
    except Exception as e:
        log.error(f"[{symbol}] Kauf fehlgeschlagen: {e}")
    return state


def place_sell_order(exchange, symbol, price, state, reason="Signal"):
    amount = state.get("amount")
    if not amount:
        return state
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        pnl_pct = (price - state["entry_price"]) / state["entry_price"] * 100
        log.info(f"[{symbol}] VERKAUF ausgeführt ({reason}): {amount:.6f} @ ~{price_round(price)} | PnL: {pnl_pct:.2f}%")
        dashboard.add_trade(symbol, "SELL", price, pnl_pct, reason)
        state.update({"in_position": False, "entry_price": None, "amount": None,
                       "stop_loss": None, "take_profit": None})
    except Exception as e:
        log.error(f"[{symbol}] Verkauf fehlgeschlagen: {e}")
    return state


# ============================================================
# SIGNAL-BERECHNUNG (ein Durchlauf = Struktur + Modell + Saisonalität, pro Symbol)
# ============================================================
def compute_signal(df, state, model_state, symbol=""):
    """Gibt (signal, last_row, model_proba, season_score, df) zurück.
    signal ist 'BUY' nur wenn aktuell FLAT und alle drei Ebenen zustimmen,
    sonst 'HOLD' (Exit-Logik läuft separat über Stop-Loss/Take-Profit)."""
    df = find_swing_points(df)
    df = detect_structure_breaks(df)
    df = build_features(df)

    model_state.maybe_retrain(df, symbol=symbol)

    weekday_perf, hour_perf = compute_seasonality(df)
    last = df.iloc[-1]
    model_proba = model_state.model.predict_proba_up(last)
    season_score = seasonality_score(df.index[-1], weekday_perf, hour_perf)

    signal = "HOLD"
    if not state["in_position"] and last["bos_up"] and not np.isnan(last["last_swing_low"]):
        if model_proba >= MODEL_MIN_PROBA and season_score >= SEASONALITY_MIN_SCORE:
            signal = "BUY"

    return signal, last, model_proba, season_score, df


def compute_entry_levels(price, last_row):
    stop_candidate = last_row["last_swing_low"] * (1 - SWING_LOW_BUFFER_PCT)
    risk = price - stop_candidate
    if risk <= 0:
        return None, None
    take_profit = price + risk * RISK_REWARD
    return stop_candidate, take_profit


# ============================================================
# EINEN EINZELNEN MARKT VERARBEITEN (wird pro Symbol pro Runde aufgerufen)
# ============================================================
def process_symbol(exchange, symbol, balance):
    ensure_state(symbol)
    dashboard.ensure_market(symbol)
    model_state = get_or_create_model(symbol)
    state = state_by_symbol[symbol]

    raw_df = fetch_ohlcv_df(exchange, symbol, TIMEFRAME)
    if len(raw_df) < MIN_CANDLES_REQUIRED:
        msg = f"Zu wenig Historie geladen ({len(raw_df)} Kerzen) - warte auf mehr Daten."
        log.warning(f"[{symbol}] {msg}")
        dashboard.set_market_error(symbol, msg)
        return

    signal, last_row, model_proba, season_score, full_df = compute_signal(raw_df, state, model_state, symbol=symbol)
    price = last_row["Close"]

    dashboard.update_market_tick(symbol, price, last_row, model_proba, model_state.model.is_trained,
                                  season_score, signal, state)
    dashboard.update_market_candles(symbol, full_df)

    if balance is not None:
        base_currency = symbol.split("/")[0]
        dashboard.update_market_balance(symbol, balance.get(base_currency, {}))

    # Stop-Loss / Take-Profit haben IMMER Vorrang vor neuen Entry-Signalen
    if state["in_position"] and state.get("stop_loss") and state.get("take_profit"):
        if price <= state["stop_loss"]:
            log.warning(f"[{symbol}] STOP-LOSS ausgelöst bei {price_round(price)} (Level: {price_round(state['stop_loss'])})")
            state_by_symbol[symbol] = place_sell_order(exchange, symbol, price, state, reason="STOP-LOSS")
            return
        elif price >= state["take_profit"]:
            log.info(f"[{symbol}] TAKE-PROFIT erreicht bei {price_round(price)} (Level: {price_round(state['take_profit'])})")
            state_by_symbol[symbol] = place_sell_order(exchange, symbol, price, state, reason="TAKE-PROFIT")
            return

    if signal == "BUY" and not state["in_position"]:
        stop_loss, take_profit = compute_entry_levels(price, last_row)
        if stop_loss is None:
            log.warning(f"[{symbol}] Strukturbruch erkannt, aber Stop-Abstand <= 0 (Swing-Low zu nah am Preis) - übersprungen.")
        else:
            state_by_symbol[symbol] = place_buy_order(exchange, symbol, price, stop_loss, take_profit,
                                                        model_proba, season_score, state)
    else:
        bos_txt = "BOS_UP" if last_row["bos_up"] else "kein Strukturbruch"
        trained_txt = "trainiert" if model_state.model.is_trained else "noch nicht trainiert"
        log.info(f"[{symbol}] Kein Trade | Preis: {price_round(price)} | {bos_txt} | Modell: {trained_txt} "
                 f"(Proba: {model_proba:.2f}) | Saison-Score: {season_score:+.5f} | In Position: {state['in_position']}")


# ============================================================
# HAUPTSCHLEIFE - geht pro Runde ALLE aktiven Symbole durch
# ============================================================
def run_bot():
    global _exchange_ref, state_by_symbol
    exchange = create_exchange()
    _exchange_ref = exchange
    state_by_symbol = load_state()

    try:
        markets = exchange.load_markets()
        for symbol in registry.get_all():
            if symbol not in markets:
                log.error(f"'{symbol}' ist auf {'Testnet' if USE_TESTNET else 'Binance'} nicht verfügbar! "
                          f"Wird aus der Handelsliste entfernt - bitte im Dashboard einen gültigen Markt hinzufügen.")
                registry.remove(symbol)
    except Exception as e:
        log.warning(f"Marktliste konnte nicht geprüft werden (wird trotzdem versucht): {e}")

    for symbol in registry.get_all():
        ensure_state(symbol)
        dashboard.ensure_market(symbol)

    log.info(f"Bot gestartet | Strategie: Struktur+Modell+Saisonalität | Märkte: {registry.get_all()} | "
             f"Timeframe: {TIMEFRAME} | Testnet: {USE_TESTNET}")
    log.info(f"Dashboard verfügbar unter: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    while True:
        active_symbols = registry.get_all()
        if not active_symbols:
            log.warning("Keine Märkte aktiv - füge im Dashboard mindestens einen Markt hinzu.")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        try:
            balance = exchange.fetch_balance()
            dashboard.update_quote_balance(balance.get(QUOTE_CURRENCY, {}))
        except Exception as e:
            log.warning(f"Guthaben konnte nicht geladen werden: {e}")
            balance = None

        for symbol in active_symbols:
            try:
                process_symbol(exchange, symbol, balance)
            except ccxt.NetworkError as e:
                log.error(f"[{symbol}] Netzwerkfehler, versuche es nächste Runde erneut: {e}")
                dashboard.set_market_error(symbol, "Netzwerkfehler")
            except Exception as e:
                log.error(f"[{symbol}] Unerwarteter Fehler: {e}")
                dashboard.set_market_error(symbol, str(e))

        save_state(state_by_symbol)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    time.sleep(1)
    print(f"\n  ➜  Dashboard läuft unter: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}\n")

    run_bot()
