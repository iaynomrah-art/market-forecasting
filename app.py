import os
import json
import requests
import feedparser
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tradingview_ta import TA_Handler, Interval

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Timeframe Mapping Engine ---
TIMEFRAME_MAP = {
    "5m": {"tv_interval": Interval.INTERVAL_5_MINUTES, "text": "5 minutes"},
    "15m": {"tv_interval": Interval.INTERVAL_15_MINUTES, "text": "15 minutes"},
    "1h": {"tv_interval": Interval.INTERVAL_1_HOUR, "text": "1 hour"},
    "4h": {"tv_interval": Interval.INTERVAL_4_HOURS, "text": "4 hours"},
    "1d": {"tv_interval": Interval.INTERVAL_1_DAY, "text": "1 day"},
    "1W": {"tv_interval": Interval.INTERVAL_1_WEEK, "text": "1 week"},
    "1M": {"tv_interval": Interval.INTERVAL_1_MONTH, "text": "1 month"}
}

MACRO_PAIRINGS = {
    "5m": "1h",   
    "15m": "4h",  
    "1h": "1d",   
    "4h": "1W",   
    "1d": "1M"    
}

def get_latest_headlines():
    try:
        feed = feedparser.parse("https://www.forexlive.com/feed/news")
        headlines_with_time = []
        for entry in feed.entries[:3]:
            title = entry.title
            published_time = entry.get('published', 'Unknown time')
            headlines_with_time.append(f"[{published_time}] {title}")
        return " | ".join(headlines_with_time)
    except Exception:
        return "No significant news data available."

def get_exchange_config(symbol):
    symbol = symbol.upper()
    if symbol in ["XAUUSD", "GOLD"]:
        return {"tv_symbol": "GOLD", "screener": "cfd", "exchange": "TVC"} 
    elif symbol == "BTCUSD":
        return {"tv_symbol": "BTCUSD", "screener": "crypto", "exchange": "COINBASE"}
    else:
        return {"tv_symbol": symbol, "screener": "crypto", "exchange": "BINANCE"}

def fetch_technical_data(config, tf_data):
    handler = TA_Handler(
        symbol=config["tv_symbol"],   
        screener=config["screener"],      
        exchange=config["exchange"],     
        interval=tf_data["tv_interval"]  
    )
    analysis = handler.get_analysis()
    
    raw_summary = analysis.summary["RECOMMENDATION"]
    buy_votes = analysis.summary["BUY"]
    sell_votes = analysis.summary["SELL"]
    neutral_votes = analysis.summary["NEUTRAL"]
    total = buy_votes + sell_votes + neutral_votes
    
    decision = "BUY" if buy_votes > sell_votes else "SELL"
    if "BUY" in raw_summary: decision = "BUY"
    elif "SELL" in raw_summary: decision = "SELL"

    math_process = f"Analyzed {total} indicators. Results: {buy_votes} BUY, {sell_votes} SELL, {neutral_votes} NEUTRAL. Verdict: {decision}."

    ind = analysis.indicators
    price = round(ind.get("close", 0), 2)
    rsi = round(ind.get("RSI", 50), 2) # Default to 50 if missing
    macd = round(ind.get("MACD.macd", 0), 2)
    ema_20 = round(ind.get("EMA20", 0), 2)
    
    sma_200 = round(ind.get("SMA200", 0), 2)
    sma_50 = round(ind.get("SMA50", 0), 2)
    
    # NEW: ATR with a mathematical safety fallback to prevent 0-value crashes
    raw_atr = ind.get("ATR")
    if raw_atr is None or raw_atr == 0:
        atr = round(price * 0.002, 2) # Fallback to 0.2% of current price
    else:
        atr = round(raw_atr, 2)

    rsi_txt = f"RSI {rsi} (Overbought)" if rsi > 70 else f"RSI {rsi} (Oversold)" if rsi < 30 else f"RSI {rsi} (Neutral)"
    macd_txt = f"MACD Bearish ({macd})" if macd < 0 else f"MACD Bullish ({macd})"
    trend_txt = f"Price vs 200 SMA: {'Above' if price > sma_200 else 'Below'}. Price vs 50 SMA: {'Above' if price > sma_50 else 'Below'}"
    
    return {
        "decision": decision,
        "process": math_process,
        "basis": f"{trend_txt}. {macd_txt}. {rsi_txt}.",
        "price": price,
        "raw_rsi": rsi,
        "atr": atr,
        "sma_200": sma_200
    }

def ask_ollama(symbol, micro_tf, macro_tf, micro_data, macro_data, headlines, current_utc_time):
    current_price = micro_data['price']
    atr = micro_data['atr']
    
    # Pre-calculate ideal risk/reward parameters to prevent LLM math errors
    buy_stop = round(current_price - (1.5 * atr), 2)
    buy_target = round(current_price + (2.0 * atr), 2)
    sell_stop = round(current_price + (1.5 * atr), 2)
    sell_target = round(current_price - (2.0 * atr), 2)
    
    system_prompt = f"""You are a strict quantitative trading AI analyzing {symbol} for a short-term ({micro_tf}) execution bot.
    
    CURRENT SYSTEM TIME: {current_utc_time}
    CURRENT PRICE: {current_price}
    200 SMA (Long Term Trend): {micro_data['sma_200']}
    ATR (Volatility): {atr}
    
    [1] MACRO TREND ({macro_tf}): {macro_data['decision']}
    [2] MICRO TREND ({micro_tf}): {micro_data['decision']}
    
    DECISION LOGIC (Execute strictly for short-term {micro_tf} horizon):
    1. TREND ALIGNMENT: Prioritize the Micro Trend ({micro_tf}) if the current price is above the 200 SMA (for BUY) or below the 200 SMA (for SELL). 
    2. RISK/REWARD SELECTION: 
       - If you output BUY: Set stop_loss to {buy_stop} and target_price to {buy_target}.
       - If you output SELL: Set stop_loss to {sell_stop} and target_price to {sell_target}.
    3. NEWS CUES: Only reference the provided headlines if they explicitly invalidate the technical setup. Otherwise, ignore them.
    
    Respond strictly in JSON format matching this exact schema. Do not output markdown code blocks, just the raw JSON:
    {{
      "verdict": "BUY" or "SELL",
      "entry_price": {current_price},
      "target_price": <number>,
      "stop_loss": <number>,
      "confidence_score": <number 1-100 based on indicator confluence>,
      "reasoning": "Short 2-sentence explanation citing the 200 SMA, current momentum, and calculated risk parameters."
    }}
    """

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": "gemma3:4b",   
        "prompt": system_prompt,
        "stream": False,
        "format": "json",       
        "options": {"temperature": 0.0, "num_thread": 4} 
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        raw_response = data.get("response", "{}")
        ai_data = json.loads(raw_response)
        
        # Fallback safety checks
        if ai_data.get("verdict", "").strip().upper() not in ["BUY", "SELL"]:
            ai_data["verdict"] = micro_data["decision"]
            
        return ai_data
    except Exception as e:
        print(f"Ollama error: {e}")
        # Return a structurally sound fallback if Ollama crashes
        fallback_verdict = micro_data["decision"]
        return {
            "verdict": fallback_verdict, 
            "entry_price": current_price,
            "target_price": buy_target if fallback_verdict == "BUY" else sell_target,
            "stop_loss": buy_stop if fallback_verdict == "BUY" else sell_stop,
            "confidence_score": 50,
            "reasoning": "AI offline. Defaulting to standard technicals."
        }

@app.get("/api/signal")
def get_market_signal(symbol: str = "XAUUSD", timeframe: str = "1h"):
    try:
        target_symbol = symbol.upper()
        config = get_exchange_config(target_symbol)
        
        # 1. Identify Timeframes
        micro_tf_key = timeframe.lower()
        macro_tf_key = MACRO_PAIRINGS.get(micro_tf_key, "1d") 
        
        micro_tf_data = TIMEFRAME_MAP.get(micro_tf_key, TIMEFRAME_MAP["1h"])
        macro_tf_data = TIMEFRAME_MAP.get(macro_tf_key, TIMEFRAME_MAP["1d"])

        # 2. Fetch Math
        micro_data = fetch_technical_data(config, micro_tf_data)
        macro_data = fetch_technical_data(config, macro_tf_data)

        # 3. Fetch News & Current Time
        headlines = get_latest_headlines()
        current_utc_time = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

        # 4. Feed everything directly to the AI
        ai_response = ask_ollama(
            target_symbol, 
            micro_tf_data["text"], 
            macro_tf_data["text"], 
            micro_data, 
            macro_data, 
            headlines,
            current_utc_time
        )
        
        return {
            "symbol": target_symbol,
            "timeframe": micro_tf_data["text"],
            "signal": {
                "action": ai_response.get("verdict"),
                "entry": ai_response.get("entry_price", micro_data["price"]),
                "target": ai_response.get("target_price"),
                "stop_loss": ai_response.get("stop_loss"),
                "confidence": ai_response.get("confidence_score")
            },
            "technical_context": {
                "current_price": micro_data["price"],
                "sma_200": micro_data["sma_200"],
                "volatility_atr": micro_data["atr"],
                "micro_verdict": micro_data["decision"],
                "macro_verdict": macro_data["decision"]
            },
            "ai_reasoning": ai_response.get("reasoning")
        }
        
    except Exception as e:
        return {"error": str(e)}
