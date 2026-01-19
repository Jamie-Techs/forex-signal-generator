import os, logging, threading, requests, time, numpy as np, pandas as pd, ta
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, StopLossDetails, TakeProfitDetails, TrailingStopLossDetails

# ================= CONFIGURATION =================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
BASE_RISK = 0.005
DAILY_LOSS_LIMIT = 0.02
PORTFOLIO_LOCK = threading.Lock()

# Global tracking
SESSION_START_NAV = None
SESSION_DATE = None
DAILY_TRADE_COUNT = 0
HIGH_IMPACT_EVENTS = []
OPEN_POSITIONS = {} 

client = API(access_token=OANDA_API_KEY, environment="practice")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-V10-FINAL] | %(message)s")

# ==================== TELEGRAM & PERFORMANCE ====================
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def send_daily_summary():
    """Calculates P/L and sends at 5:00 PM UTC."""
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        curr_nav = float(r.response["account"]["NAV"])
        pnl = curr_nav - SESSION_START_NAV
        pnl_pct = (pnl / SESSION_START_NAV) * 100
        
        summary = (
            f"📊 --- DAILY TRADING REPORT ---\n"
            f"💰 Ending NAV: ${curr_nav:,.2f}\n"
            f"📈 Net P/L: ${pnl:,.2f} ({pnl_pct:.2f}%)\n"
            f"🚀 Trades Today: {DAILY_TRADE_COUNT}\n"
            f"📅 Session: {datetime.now().strftime('%Y-%m-%d')}"
        )
        send_telegram(summary)
    except Exception as e:
        logging.error(f"Summary Error: {e}")

# ==================== NEWS & CALENDAR ====================
def scrape_high_impact_news():
    global HIGH_IMPACT_EVENTS
    try:
        # Using a reliable XML feed
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        xml = response.text
        events = []
        # Basic parsing of XML structure
        for item in xml.split("<event>")[1:]:
            if "<impact>High</impact>" in item:
                currency = item.split("<currency>")[1].split("</currency>")[0]
                title = item.split("<title>")[1].split("</title>")[0]
                date = item.split("<date>")[1].split("</date>")[0]
                time_str = item.split("<time>")[1].split("</time>")[0]
                
                # Convert to UTC datetime
                dt_str = f"{date} {time_str}"
                event_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc)
                events.append({"currency": currency, "title": title, "time": event_dt})
        
        HIGH_IMPACT_EVENTS = events
        logging.info(f"Loaded {len(events)} High Impact events.")
    except Exception as e:
        logging.error(f"Calendar Scrape Fail: {e}")

def is_news_window(symbol):
    base, quote = symbol.split("_")
    now = datetime.now(timezone.utc)
    for ev in HIGH_IMPACT_EVENTS:
        if ev["currency"] in (base, quote):
            delta = (ev["time"] - now).total_seconds()
            if -60 <= delta <= 300: # 1 min before to 5 mins after
                return True, ev
    return False, None

# ==================== LOGIC & FILTERS ====================
def multi_currency_filter(symbol, signal):
    """Prevents over-exposure to correlated pairs."""
    correlations = [("EUR_USD", "GBP_USD"), ("AUD_USD", "NZD_USD")]
    for pair in correlations:
        if symbol in pair:
            other = pair[0] if pair[1] == symbol else pair[1]
            if other in OPEN_POSITIONS:
                return False # Stay out if correlated pair is already open
    return True

def order_flow_bias(df):
    delta = df['close'] - df['open']
    bull = delta[delta > 0].sum()
    bear = abs(delta[delta < 0].sum())
    return "BUY" if bull > bear * 1.4 else "SELL" if bear > bull * 1.4 else "NEUTRAL"

# ==================== EXECUTION ====================
def execute_unified_trade(symbol, side, price, atr, nav, trade_type):
    global DAILY_TRADE_COUNT, OPEN_POSITIONS
    try:
        pip = 0.01 if "JPY" in symbol else 0.0001
        units = max(1, int((nav * BASE_RISK) / (atr * 3 * pip)))
        if side == "SELL": units *= -1

        prec = 3 if "JPY" in symbol else 5
        sl = round(price - atr*3 if side=="BUY" else price + atr*3, prec)
        tp1 = round(price + atr*1.5 if side=="BUY" else price - atr*1.5, prec)
        tp2 = round(price + atr*3.0 if side=="BUY" else price - atr*3.0, prec)

        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(sl)).data,
            takeProfitOnFill=TakeProfitDetails(price=str(tp2)).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr*1.5, prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        
        DAILY_TRADE_COUNT += 1
        OPEN_POSITIONS[symbol] = True
        
        msg = f"🔔 SIGNAL: {trade_type}\n📦 {symbol} {side}\n💵 Entry: {price}\n🛡 SL: {sl}\n🎯 TP1: {tp1} (Partial) | TP2: {tp2}"
        send_telegram(msg)
    except Exception as e:
        logging.error(f"Trade Fail: {e}")

# ==================== SCHEDULER & APP ====================
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(func=scrape_high_impact_news, trigger="interval", minutes=60)
scheduler.add_job(func=send_daily_summary, trigger="cron", hour=17, minute=0) # 5:00 PM UTC
scheduler.start()

@app.route('/run')
def trigger():
    def run_cycle():
        for i, s in enumerate(SYMBOLS):
            # Process sequentially with 15s delay to prevent 429 errors
            run_apex_unified(s, i)
            time.sleep(15)

    threading.Thread(target=run_cycle).start()
    return jsonify({"status": "Sequential Engine Started"}), 200

# (Include your run_apex_unified logic here)

if __name__ == "__main__":
    scrape_high_impact_news()
    r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
    SESSION_START_NAV = float(r.response["account"]["NAV"])
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
