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
from flask import Flask, jsonify, render_template

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

USE_TESTNET = True                 # NIEMALS ohne wochenlange Testnet-Erfahrung auf False stellen
SYMBOL = "DOGE/USDT"               # volatiler als BTC - gut zum Stresstesten der Strategie
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
            self.data["price"] = round(float(price), 2)
            self.data["ma_short"] = round(float(ma_short), 2) if ma_short == ma_short else None
            self.data["ma_long"] = round(float(ma_long), 2) if ma_long == ma_long else None
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
                "price": round(float(price), 2),
                "ma_short": round(float(ma_short), 2) if ma_short == ma_short else None,
                "ma_long": round(float(ma_long), 2) if ma_long == ma_long else None,
            })

    def update_candles(self, df):
        with self.lock:
            recent = df.tail(CANDLE_MAXLEN)
            self.data["candles"] = deque(
                (
                    {
                        "t": int(ts.timestamp() * 1000),
                        "o": round(float(row["Open"]), 2),
                        "h": round(float(row["High"]), 2),
                        "l": round(float(row["Low"]), 2),
                        "c": round(float(row["Close"]), 2),
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

    def add_trade(self, trade_type, price, pnl_pct, reason):
        with self.lock:
            self.data["trades"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": trade_type,
                "price": round(float(price), 2),
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
        log.info(f"KAUF ausgeführt: {base_amount:.6f} @ ~{price:.2f} (Order-ID: {order['id']})")
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
        log.info(f"VERKAUF ausgeführt ({reason}): {amount:.6f} @ ~{price:.2f} | PnL: {pnl_pct:.2f}%")
        dashboard.add_trade("SELL", price, pnl_pct, reason)
        state.update({"in_position": False, "entry_price": None, "amount": None})
    except Exception as e:
        log.error(f"Verkauf fehlgeschlagen: {e}")
    return state


# ============================================================
# HAUPTSCHLEIFE
# ============================================================
def run_bot():
    exchange = create_exchange()
    state = load_state()

    try:
        markets = exchange.load_markets()
        if SYMBOL not in markets:
            volatile_candidates = ["DOGE/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "TRX/USDT", "ETH/USDT", "BNB/USDT"]
            available = [s for s in volatile_candidates if s in markets]
            log.error(f"'{SYMBOL}' ist auf {'Testnet' if USE_TESTNET else 'Binance'} nicht verfügbar! "
                      f"Verfügbare Alternativen aus unserer Liste: {available or 'siehe exchange.load_markets()'}. "
                      f"Bitte SYMBOL in der CONFIG anpassen und Bot neu starten.")
            return
    except Exception as e:
        log.warning(f"Marktliste konnte nicht geprüft werden (wird trotzdem versucht): {e}")

    log.info(f"Bot gestartet | Symbol: {SYMBOL} | Timeframe: {TIMEFRAME} | "
             f"Testnet: {USE_TESTNET} | In Position: {state['in_position']}")
    log.info(f"Dashboard verfügbar unter: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    while True:
        try:
            df = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME)
            signal, last_row = compute_signal(df)
            price = last_row["Close"]

            dashboard.update_tick(price, last_row["MA_short"], last_row["MA_long"],
                                   last_row["RSI"], signal, state)
            dashboard.update_candles(df)

            try:
                quote_ccy, base_ccy, quote_bal, base_bal = get_full_balance(exchange, SYMBOL)
                dashboard.update_balance(quote_ccy, base_ccy, quote_bal, base_bal)
            except Exception as e:
                log.warning(f"Guthaben konnte nicht geladen werden: {e}")

            # Stop-Loss-Check hat IMMER Vorrang vor Strategie-Signalen
            if state["in_position"] and state["entry_price"]:
                loss_pct = (price - state["entry_price"]) / state["entry_price"]
                if loss_pct <= -STOP_LOSS_PCT:
                    log.warning(f"STOP-LOSS ausgelöst bei {loss_pct*100:.2f}%")
                    state = place_sell_order(exchange, SYMBOL, price, state, reason="STOP-LOSS")
                    save_state(state)
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue

            if signal == "BUY" and not state["in_position"]:
                state = place_buy_order(exchange, SYMBOL, price, state)
            elif signal == "SELL" and state["in_position"]:
                state = place_sell_order(exchange, SYMBOL, price, state, reason="Signal")
            else:
                ma_short = last_row["MA_short"]
                ma_long = last_row["MA_long"]
                gap_pct = (ma_short - ma_long) / ma_long * 100
                trend = "MA20 über MA50 (bullisch)" if ma_short > ma_long else "MA20 unter MA50 (bärisch)"
                log.info(f"Kein Trade | Preis: {price:.2f} | MA20: {ma_short:.2f} | MA50: {ma_long:.2f} "
                         f"| Abstand: {gap_pct:+.3f}% ({trend}) | RSI: {last_row['RSI']:.1f} "
                         f"| In Position: {state['in_position']}")

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
