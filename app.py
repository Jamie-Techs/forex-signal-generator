import os, logging, threading, requests, time, numpy as np, pandas as pd, ta, openai
from datetime import datetime, timezone
import pytz
from flask import Flask, jsonify
from bayes_opt import BayesianOptimization
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import (
    MarketOrderRequest, StopLossDetails, 
    TakeProfitDetails, TrailingStopLossDetails
)

# ================== CONFIGURATION ==================
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
RISK_PER_TRADE = 0.005  # 0.5% risk
DAILY_LOSS_LIMIT = 0.02 # 2% Daily Drawdown Guard
PORTFOLIO_LOCK = threading.Lock()

# Global tracking for the Drawdown Guard
SESSION_START_NAV = None
SESSION_DATE = None

client = API(access_token=OANDA_API_KEY, environment="practice")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [APEX-PRO] | %(message)s")

# ================== LAYER 1: UTILS & RISK GUARD ==================

def is_market_open():
    """Checks NY Market Hours (Sun 17:00 - Fri 17:00 EST)."""
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    if now.weekday() == 5: return False 
    if now.weekday() == 4 and now.hour >= 17: return False
    if now.weekday() == 6 and now.hour < 17: return False
    return True

def drawdown_guard_passed():
    """Strategic Update 1: Daily Max Drawdown Limit."""
    global SESSION_START_NAV, SESSION_DATE
    try:
        r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(r)
        curr_nav = float(r.response["account"]["NAV"])
        today = datetime.now(timezone.utc).date()

        if SESSION_DATE != today or SESSION_START_NAV is None:
            SESSION_START_NAV = curr_nav
            SESSION_DATE = today
            logging.info(f"New Session Started. Starting NAV: {SESSION_START_NAV}")
            return True

        drawdown = (SESSION_START_NAV - curr_nav) / SESSION_START_NAV
        if drawdown >= DAILY_LOSS_LIMIT:
            logging.critical(f"🚨 STOP: Daily loss limit reached ({drawdown:.2%}).")
            return False
        return True
    except Exception as e:
        logging.error(f"Drawdown Guard Error: {e}")
        return True

def get_h1_trend(symbol):
    """Strategic Update 2: Multi-Timeframe Analysis (H1 Filter)."""
    try:
        r = instruments.InstrumentsCandles(symbol, {"granularity": "H1", "count": 20})
        client.request(r)
        closes = [float(c["mid"]["c"]) for c in r.response["candles"]]
        sma = sum(closes) / len(closes)
        return "UP" if closes[-1] > sma else "DOWN"
    except: return "NEUTRAL"

# ================== LAYER 2: OPTIMIZER & AI ==================

def fitness_function(df, rsi_l, rsi_h, adx_t):
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], 14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    pnl = []
    for i in range(1, len(df)):
        if df['adx'].iloc[i] < adx_t:
            if df['rsi'].iloc[i] < rsi_l: pnl.append(df['close'].iloc[i] - df['close'].iloc[i-1])
            if df['rsi'].iloc[i] > rsi_h: pnl.append(df['close'].iloc[i-1] - df['close'].iloc[i])
    return np.mean(pnl) / (np.std([p for p in pnl if p < 0]) + 1e-6) if len(pnl) > 5 else -10

def get_optimized_params(df):
    opt = BayesianOptimization(
        f=lambda rsi_l, rsi_h, adx_t: fitness_function(df, rsi_l, rsi_h, adx_t),
        pbounds={'rsi_l': (20, 35), 'rsi_h': (65, 80), 'adx_t': (20, 30)},
        verbose=0, random_state=42
    )
    opt.maximize(init_points=2, n_iter=3)
    return opt.max['params']

def get_macro_sentiment(symbol):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            news_data = requests.get(url).json().get("feed", [])[:5]
            headlines = [n['title'] for n in news_data]
            prompt = (f"Analyze {symbol} headlines: {headlines}. Score -1 to 1. "
                      "If 'NFP', 'CPI', or 'Fed' news is imminent, return 'DANGER'.")
            response = ai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=15
            )
            return response.choices[0].message.content.strip().upper()
        except openai.RateLimitError:
            time.sleep((2 ** attempt) * 8)
        except Exception: break
    return "0"

# ================== LAYER 3: EXECUTION ==================

def execute_pro_trade(symbol, side, price, atr, nav):
    """Strategic Update 3: Universal Scaling (Handles $10 to $100k+)."""
    try:
        pip = 0.01 if "JPY" in symbol else 0.0001
        # Minimum 1 unit to prevent errors on tiny accounts
        units = max(1, int((nav * RISK_PER_TRADE) / (atr * 3 * pip)))
        if side == "SELL": units *= -1
        
        prec = 3 if "JPY" in symbol else 5
        order = MarketOrderRequest(
            instrument=symbol, units=units,
            stopLossOnFill=StopLossDetails(price=str(round(price - atr*3 if side=="BUY" else price + atr*3, prec))).data,
            takeProfitOnFill=TakeProfitDetails(price=str(round(price + atr*2 if side=="BUY" else price - atr*2, prec))).data,
            trailingStopLossOnFill=TrailingStopLossDetails(distance=str(round(atr * 1.5, prec))).data
        )
        client.request(orders.OrderCreate(OANDA_ACCOUNT_ID, data=order.data))
        logging.info(f"🚀 {side} {symbol} Placed | Units: {abs(units)}")
    except Exception as e:
        logging.error(f"Execution Error for {symbol}: {e}")

# ================== ENGINE ==================

def run_apex_pro(symbol, index):
    time.sleep(index * 5)
    
    # 1. Market Hours Check
    if not is_market_open():
        logging.info(f"Market Closed for {symbol}. Standing by.")
        return

    # 2. Risk Guard Check
    if not drawdown_guard_passed():
        return

    with PORTFOLIO_LOCK:
        try:
            # 3. Multi-Timeframe Filter
            htf_trend = get_h1_trend(symbol)
            
            # 4. News Guard
            sentiment = get_macro_sentiment(symbol)
            if "DANGER" in sentiment:
                logging.warning(f"🚫 {symbol} News Guard: DANGER detected.")
                return

            # 5. Data & Optimization
            r = instruments.InstrumentsCandles(symbol, {"granularity": "M15", "count": 200})
            client.request(r)
            df = pd.DataFrame([{"close": float(c["mid"]["c"]), "open": float(c["mid"]["o"]), 
                                "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"])} for c in r.response["candles"]])
            
            params = get_optimized_params(df)
            rsi = ta.momentum.rsi(df['close'], 14).iloc[-1]
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            
            # 6. Confluence Decision
            is_bullish_pat = df.iloc[-1]['close'] > df.iloc[-2]['open']
            signal = "NEUTRAL"
            
            if rsi < params['rsi_l'] and is_bullish_pat and htf_trend == "UP": signal = "BUY"
            if rsi > params['rsi_h'] and not is_bullish_pat and htf_trend == "DOWN": signal = "SELL"
            
            # 7. Final Execution
            if signal != "NEUTRAL":
                ai_score = float(sentiment) if any(char.isdigit() for char in sentiment) else 0
                if (signal == "BUY" and ai_score >= -0.1) or (signal == "SELL" and ai_score <= 0.1):
                    # Fetch NAV one last time for precise sizing
                    acc_r = accounts.AccountSummary(OANDA_ACCOUNT_ID); client.request(acc_r)
                    execute_pro_trade(symbol, signal, df.iloc[-1]['close'], atr, float(acc_r.response["account"]["NAV"]))
        
        except Exception as e:
            logging.error(f"Logic Error for {symbol}: {e}")

@app.route('/')
def home():
    return jsonify({"status": "Active", "engine": "APEX-PRO-v2026", "guards": "MTFA + Drawdown"}), 200

@app.route("/run")
def trigger_cycle():
    now_hour = datetime.now(timezone.utc).hour
    if 21 <= now_hour <= 23: return jsonify({"status": "Paused (Rollover)"})

    threads = [threading.Thread(target=run_apex_pro, args=(s, i)) for i, s in enumerate(SYMBOLS)]
    for t in threads: t.start()
    return jsonify({"status": "Cycle Launched", "time": str(datetime.now())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
