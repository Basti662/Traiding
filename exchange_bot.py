"""
Exchange-Trading-Bot (Binance Testnet)
========================================
Verbindet die MA-Crossover-Strategie mit einer echten Exchange (Testnet = Fake-Geld).

WICHTIGE SICHERHEITSHINWEISE:
- USE_TESTNET = True ist der Standard. Erst auf False stellen, wenn du die Strategie
  wochenlang auf Testnet beobachtet hast UND die Ergebnisse überzeugen.
- API Keys NIEMALS im Code hardcoden oder auf GitHub hochladen. Sie kommen aus einer
  .env-Datei (siehe Anleitung unten).
- Erstelle Testnet-Keys NUR mit Handelsrechten, KEINEN Abhebe-Rechten (Withdrawal).
  Auf Live-Exchanges gilt das genauso: Abheben nie über API erlauben.
- MAX_POSITION_SIZE_PCT begrenzt, wie viel % deines Kapitals pro Trade riskiert wird.

SETUP:
1. pip install ccxt python-dotenv pandas numpy
2. Datei ".env" im selben Ordner erstellen mit:
     BINANCE_API_KEY=dein_testnet_api_key
     BINANCE_API_SECRET=dein_testnet_secret
3. python exchange_bot.py
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

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

USE_TESTNET = True                 # NIEMALS ohne wochenlange Testnet-Erfahrung auf False stellen
SYMBOL = "SOL/USDT"               # volatiler als BTC - gut zum Stresstesten der Strategie
TIMEFRAME = "15m"                  # Kerzengröße: '1m','5m','15m','1h','4h','1d'
SHORT_WINDOW = 20
LONG_WINDOW = 50
USE_RSI_FILTER = True
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

MAX_POSITION_SIZE_PCT = 0.10       # max. 10% des verfügbaren Guthabens pro Trade
STOP_LOSS_PCT = 0.08               # Notverkauf bei -8% seit Kauf (bei volatilen Coins etwas großzügiger,
                                    # sonst löst er bei normalem Rauschen zu oft aus)
CHECK_INTERVAL_SECONDS = 60        # wie oft der Bot nach neuen Kerzen schaut

STATE_FILE = "bot_state.json"      # merkt sich Position über Neustarts hinweg
LOG_FILE = "bot_log.txt"

DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")  # auf dem Pi in .env auf 0.0.0.0 setzen,
                                                             # um vom Laptop/Handy aus zuzugreifen
DASHBOARD_PORT = 5000
HISTORY_MAXLEN = 300                # wie viele Chart-Punkte im UI gezeigt werden
CANDLE_MAXLEN = 120                 # wie viele Kerzen im Candlestick-Chart gezeigt werden
LOG_TAIL_MAXLEN = 60                # wie viele Log-Zeilen im UI gezeigt werden

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("bot")


# ============================================================
# SHARED STATE FÜR DAS DASHBOARD (thread-sicher)
# ============================================================
class DashboardState:
    """Hält alle Daten, die das Web-Dashboard anzeigt. Wird vom Bot-Thread
    beschrieben und vom Flask-Thread gelesen."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "testnet": USE_TESTNET,
            "price": None,
            "ma_short": None,
            "ma_long": None,
            "rsi": None,
            "signal": None,
            "in_position": False,
            "entry_price": None,
            "position_amount": None,
            "pnl_pct": None,
            "last_update": None,
            "history": deque(maxlen=HISTORY_MAXLEN),
            "candles": deque(maxlen=CANDLE_MAXLEN),
            "trades": deque(maxlen=100),
            "log_tail": deque(maxlen=LOG_TAIL_MAXLEN),
            "balance": {
                "quote_currency": None, "quote_free": None, "quote_total": None,
                "base_currency": None, "base_free": None, "base_total": None,
            },
        }

    def update_tick(self, price, ma_short, ma_long, rsi, signal, state):
        with self.lock:
            self.data["price"] = price_round(price)
            self.data["ma_short"] = price_round(ma_short) if ma_short == ma_short else None
            self.data["ma_long"] = price_round(ma_long) if ma_long == ma_long else None
            self.data["rsi"] = round(float(rsi), 1) if rsi == rsi else None
            self.data["signal"] = signal
            self.data["in_position"] = state["in_position"]
            self.data["entry_price"] = state.get("entry_price")
            self.data["position_amount"] = state.get("amount")
            if state["in_position"] and state.get("entry_price"):
                self.data["pnl_pct"] = round((price - state["entry_price"]) / state["entry_price"] * 100, 2)
            else:
                self.data["pnl_pct"] = None
            self.data["last_update"] = datetime.now().strftime("%H:%M:%S")
            self.data["history"].append({
                "t": datetime.now().strftime("%H:%M:%S"),
                "price": price_round(price),
                "ma_short": price_round(ma_short) if ma_short == ma_short else None,
                "ma_long": price_round(ma_long) if ma_long == ma_long else None,
            })

    def update_candles(self, df):
        with self.lock:
            recent = df.tail(CANDLE_MAXLEN)
            self.data["candles"] = deque(
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

    def update_balance(self, quote_currency, base_currency, quote_bal, base_bal):
        with self.lock:
            self.data["balance"] = {
                "quote_currency": quote_currency,
                "quote_free": round(quote_bal.get("free", 0), 2) if quote_bal else None,
                "quote_total": round(quote_bal.get("total", 0), 2) if quote_bal else None,
                "base_currency": base_currency,
                "base_free": round(base_bal.get("free", 0), 8) if base_bal else None,
                "base_total": round(base_bal.get("total", 0), 8) if base_bal else None,
            }

    def switch_market(self, symbol, timeframe):
        with self.lock:
            self.data["symbol"] = symbol
            self.data["timeframe"] = timeframe
            self.data["price"] = None
            self.data["ma_short"] = None
            self.data["ma_long"] = None
            self.data["rsi"] = None
            self.data["signal"] = None
            self.data["history"].clear()
            self.data["candles"].clear()

    def add_trade(self, trade_type, price, pnl_pct, reason):
        with self.lock:
            self.data["trades"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
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
            return {
                **{k: v for k, v in self.data.items() if k not in ("history", "candles", "trades", "log_tail")},
                "history": list(self.data["history"]),
                "candles": list(self.data["candles"]),
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
# RUNTIME CONFIG - erlaubt Marktwechsel im laufenden Betrieb über das Dashboard
# ============================================================
MARKET_PRESETS = ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "SOL/USDT", "XRP/USDT",
                   "ADA/USDT", "BNB/USDT", "TRX/USDT", "LTC/USDT", "MATIC/USDT"]
TIMEFRAME_PRESETS = ["1m", "5m", "15m", "1h", "4h", "1d"]


class RuntimeConfig:
    """Hält Symbol/Timeframe veränderbar, damit man den Markt im laufenden Bot
    über das Dashboard wechseln kann, ohne den Prozess neu zu starten."""

    def __init__(self, symbol, timeframe):
        self.lock = threading.Lock()
        self.symbol = symbol
        self.timeframe = timeframe
        self.switch_requested = False

    def get(self):
        with self.lock:
            return self.symbol, self.timeframe

    def request_switch(self, symbol, timeframe):
        with self.lock:
            self.symbol = symbol
            self.timeframe = timeframe
            self.switch_requested = True

    def consume_switch_flag(self):
        with self.lock:
            if self.switch_requested:
                self.switch_requested = False
                return True
            return False


runtime_config = RuntimeConfig(SYMBOL, TIMEFRAME)
_exchange_ref = None  # wird in run_bot() gesetzt, damit die API-Route Märkte validieren kann


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
    """Verhindert, dass der Browser eine alte Version des Dashboards zwischenspeichert.
    Wichtig, weil sich Symbol/Timeframe/Layout während der Entwicklung ändern können."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template("index.html", symbol=SYMBOL, timeframe=TIMEFRAME)


@app.route("/api/data")
def api_data():
    return jsonify(dashboard.snapshot())


@app.route("/api/presets")
def api_presets():
    return jsonify({"markets": MARKET_PRESETS, "timeframes": TIMEFRAME_PRESETS})


@app.route("/api/switch_market", methods=["POST"])
def api_switch_market():
    payload = request.get_json(silent=True) or {}
    new_symbol = (payload.get("symbol") or "").strip().upper()
    new_timeframe = (payload.get("timeframe") or "").strip()

    if not new_symbol or "/" not in new_symbol:
        return jsonify({"ok": False, "error": "Ungültiges Symbol-Format (erwartet z.B. BTC/USDT)."}), 400
    if new_timeframe not in TIMEFRAME_PRESETS:
        return jsonify({"ok": False, "error": f"Ungültiger Timeframe. Erlaubt: {TIMEFRAME_PRESETS}"}), 400

    snapshot = dashboard.snapshot()
    if snapshot["in_position"]:
        return jsonify({"ok": False, "error": "Aktuell in offener Position - erst schließen, bevor der Markt gewechselt wird."}), 409

    if _exchange_ref is not None:
        try:
            markets = _exchange_ref.load_markets()
            if new_symbol not in markets:
                return jsonify({"ok": False, "error": f"'{new_symbol}' ist auf dieser Exchange nicht verfügbar."}), 400
        except Exception as e:
            log.warning(f"Marktvalidierung beim Wechsel fehlgeschlagen (wird trotzdem versucht): {e}")

    old_symbol, old_timeframe = runtime_config.get()
    runtime_config.request_switch(new_symbol, new_timeframe)
    dashboard.switch_market(new_symbol, new_timeframe)
    log.info(f"Markt gewechselt: {old_symbol}/{old_timeframe} → {new_symbol}/{new_timeframe}")

    return jsonify({"ok": True, "symbol": new_symbol, "timeframe": new_timeframe})


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
            "adjustForTimeDifference": True,  # gleicht Abweichungen zur Binance-Serverzeit automatisch aus
            "recvWindow": 10000,              # großzügigeres Zeitfenster gegen "outside of recvWindow"-Fehler
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


# ============================================================
# STRATEGIE (gleiche Logik wie im Backtester)
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def compute_signal(df):
    df["MA_short"] = df["Close"].rolling(SHORT_WINDOW).mean()
    df["MA_long"] = df["Close"].rolling(LONG_WINDOW).mean()
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossover_up = (last["MA_short"] > last["MA_long"]) and (prev["MA_short"] <= prev["MA_long"])
    crossover_down = (last["MA_short"] < last["MA_long"]) and (prev["MA_short"] >= prev["MA_long"])

    if USE_RSI_FILTER and crossover_up and last["RSI"] >= RSI_OVERBOUGHT:
        crossover_up = False

    if crossover_up:
        return "BUY", last
    elif crossover_down:
        return "SELL", last
    return "HOLD", last


# ============================================================
# STATE (Position wird lokal gespeichert, übersteht Neustarts)
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"in_position": False, "entry_price": None, "amount": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================
# ORDER-AUSFÜHRUNG
# ============================================================
def get_quote_balance(exchange, quote_currency="USDT"):
    balance = exchange.fetch_balance()
    return balance.get(quote_currency, {}).get("free", 0)


def get_full_balance(exchange, symbol):
    """Holt Guthaben für beide Seiten des Handelspaars, z.B. USDT und BTC bei BTC/USDT."""
    base_currency, quote_currency = symbol.split("/")
    balance = exchange.fetch_balance()
    quote_bal = balance.get(quote_currency, {})
    base_bal = balance.get(base_currency, {})
    return quote_currency, base_currency, quote_bal, base_bal


def place_buy_order(exchange, symbol, price, state):
    quote_balance = get_quote_balance(exchange)
    spend_amount = quote_balance * MAX_POSITION_SIZE_PCT

    if spend_amount < 10:  # Binance Mindestordergröße grob berücksichtigen
        log.warning(f"Zu wenig Guthaben für sinnvollen Trade ({spend_amount:.2f} USDT). Übersprungen.")
        return state

    base_amount = spend_amount / price
    try:
        order = exchange.create_market_buy_order(symbol, base_amount)
        log.info(f"KAUF ausgeführt: {base_amount:.6f} @ ~{price_round(price)} (Order-ID: {order['id']})")
        state.update({"in_position": True, "entry_price": price, "amount": base_amount})
        dashboard.add_trade("BUY", price, None, "Signal")
    except Exception as e:
        log.error(f"Kauf fehlgeschlagen: {e}")
    return state


def place_sell_order(exchange, symbol, price, state, reason="Signal"):
    amount = state.get("amount")
    if not amount:
        return state
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        pnl_pct = (price - state["entry_price"]) / state["entry_price"] * 100
        log.info(f"VERKAUF ausgeführt ({reason}): {amount:.6f} @ ~{price_round(price)} | PnL: {pnl_pct:.2f}%")
        dashboard.add_trade("SELL", price, pnl_pct, reason)
        state.update({"in_position": False, "entry_price": None, "amount": None})
    except Exception as e:
        log.error(f"Verkauf fehlgeschlagen: {e}")
    return state


# ============================================================
# HAUPTSCHLEIFE
# ============================================================
def run_bot():
    global _exchange_ref
    exchange = create_exchange()
    _exchange_ref = exchange
    state = load_state()

    current_symbol, current_timeframe = runtime_config.get()

    try:
        markets = exchange.load_markets()
        if current_symbol not in markets:
            volatile_candidates = ["DOGE/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "TRX/USDT", "ETH/USDT", "BNB/USDT"]
            available = [s for s in volatile_candidates if s in markets]
            log.error(f"'{current_symbol}' ist auf {'Testnet' if USE_TESTNET else 'Binance'} nicht verfügbar! "
                      f"Verfügbare Alternativen aus unserer Liste: {available or 'siehe exchange.load_markets()'}. "
                      f"Bitte im Dashboard einen anderen Markt wählen oder SYMBOL in der CONFIG anpassen.")
            return
    except Exception as e:
        log.warning(f"Marktliste konnte nicht geprüft werden (wird trotzdem versucht): {e}")

    log.info(f"Bot gestartet | Symbol: {current_symbol} | Timeframe: {current_timeframe} | "
             f"Testnet: {USE_TESTNET} | In Position: {state['in_position']}")
    log.info(f"Dashboard verfügbar unter: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    while True:
        try:
            # Prüfen, ob über das Dashboard ein Marktwechsel angefordert wurde
            if runtime_config.consume_switch_flag():
                current_symbol, current_timeframe = runtime_config.get()
                state = {"in_position": False, "entry_price": None, "amount": None}
                save_state(state)
                log.info(f"Wechsel übernommen - handle jetzt {current_symbol} auf {current_timeframe}")

            df = fetch_ohlcv_df(exchange, current_symbol, current_timeframe)
            signal, last_row = compute_signal(df)
            price = last_row["Close"]

            dashboard.update_tick(price, last_row["MA_short"], last_row["MA_long"],
                                   last_row["RSI"], signal, state)
            dashboard.update_candles(df)

            try:
                quote_ccy, base_ccy, quote_bal, base_bal = get_full_balance(exchange, current_symbol)
                dashboard.update_balance(quote_ccy, base_ccy, quote_bal, base_bal)
            except Exception as e:
                log.warning(f"Guthaben konnte nicht geladen werden: {e}")

            # Stop-Loss-Check hat IMMER Vorrang vor Strategie-Signalen
            if state["in_position"] and state["entry_price"]:
                loss_pct = (price - state["entry_price"]) / state["entry_price"]
                if loss_pct <= -STOP_LOSS_PCT:
                    log.warning(f"STOP-LOSS ausgelöst bei {loss_pct*100:.2f}%")
                    state = place_sell_order(exchange, current_symbol, price, state, reason="STOP-LOSS")
                    save_state(state)
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

            if signal == "BUY" and not state["in_position"]:
                state = place_buy_order(exchange, current_symbol, price, state)
            elif signal == "SELL" and state["in_position"]:
                state = place_sell_order(exchange, current_symbol, price, state, reason="Signal")
            else:
                ma_short = last_row["MA_short"]
                ma_long = last_row["MA_long"]
                gap_pct = (ma_short - ma_long) / ma_long * 100
                trend = "MA20 über MA50 (bullisch)" if ma_short > ma_long else "MA20 unter MA50 (bärisch)"
                log.info(f"Kein Trade | Preis: {price_round(price)} | MA20: {price_round(ma_short)} | "
                         f"MA50: {price_round(ma_long)} | Abstand: {gap_pct:+.3f}% ({trend}) "
                         f"| RSI: {last_row['RSI']:.1f} | In Position: {state['in_position']}")

            save_state(state)

        except ccxt.NetworkError as e:
            log.error(f"Netzwerkfehler, versuche es erneut: {e}")
        except Exception as e:
            log.error(f"Unerwarteter Fehler: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Dashboard-Server im Hintergrund starten, Bot-Loop läuft im Hauptthread
    # (damit Strg+C weiterhin sauber funktioniert)
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    time.sleep(1)  # Flask kurz Zeit zum Hochfahren geben
    print(f"\n  ➜  Dashboard läuft unter: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}\n")

    run_bot()
