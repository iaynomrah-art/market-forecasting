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
    rsi = round(ind.get("RSI", 50), 2) # Default to 50 if missing
    macd = round(ind.get("MACD.macd", 0), 2)
    ema_20 = round(ind.get("EMA20", 0), 2)
    price = round(ind.get("close", 0), 2)

    rsi_txt = f"RSI {rsi} (Overbought)" if rsi > 70 else f"RSI {rsi} (Oversold)" if rsi < 30 else f"RSI {rsi} (Neutral)"
    macd_txt = f"MACD Bearish ({macd})" if macd < 0 else f"MACD Bullish ({macd})"
    trend_txt = f"Price below 20 EMA" if price < ema_20 else f"Price above 20 EMA"
    
    return {
        "decision": decision,
        "process": math_process,
        "basis": f"{trend_txt}. {macd_txt}. {rsi_txt}.",
        "price": price,
        "raw_rsi": rsi 
    }

def ask_ollama(symbol, micro_tf, macro_tf, micro_data, macro_data, headlines, current_utc_time):
    system_prompt = f"""You are a quantitative trading AI analyzing {symbol}.
    
    CURRENT SYSTEM TIME: {current_utc_time}
    
    [1] MACRO TREND (Primary Direction - {macro_tf}):
    - Verdict: {macro_data['decision']}
    
    [2] MICRO TREND (Entry Momentum - {micro_tf}):
    - Verdict: {micro_data['decision']}
    
    [3] CATALYSTS (News): 
    {headlines}
    
    DECISION LOGIC (Follow Strictly):
    1. CONFLUENCE: If Macro is BUY and Micro is BUY, output "BUY". If Macro is SELL and Micro is SELL, output "SELL".
    2. CONFLICT: If Macro and Micro disagree, YOU MUST CHOOSE THE MACRO TREND. The Macro trend ({macro_tf}) is the absolute authority.
    3. RSI OVERRIDE: Ignore the trend ONLY if the Micro RSI is extreme. If Micro RSI > 70, output "SELL". If Micro RSI < 30, output "BUY".
    4. NEWS RECENCY: Compare the News timestamps to the CURRENT SYSTEM TIME. If the news is more than 4 hours old, consider it "priced in" and ignore it. Only factor in fresh news if it heavily aligns with the Macro trend.
    
    Respond strictly in JSON format containing exactly two keys:
    {{
      "verdict": "BUY" or "SELL",
      "reasoning": "Strict 2-sentence explanation citing the Macro trend, RSI, and relevant fresh news."
    }}
    """

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": "gemma3:4b",   
        "prompt": system_prompt,
        "stream": False,
        "format": "json",       
        "options": {"temperature": 0.0, "num_thread": 4} # Temp 0.0 forces logic over creativity
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        raw_response = data.get("response", "{}")
        ai_data = json.loads(raw_response)
        
        verdict = ai_data.get("verdict", "").strip().upper()
        
        # Strict fallback to prevent hallucinations
        if verdict not in ["BUY", "SELL"]:
            verdict = macro_data["decision"] 
            
        return verdict, ai_data.get("reasoning", "AI generated standard signal based on strict technical rules.")
    except Exception as e:
        print(f"Ollama error: {e}")
        return macro_data["decision"], "AI offline or failed. Defaulting to Macro technical math."

@app.get("/api/signal")
def get_market_signal(symbol: str = "BTCUSD", timeframe: str = "1h"):
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

        # ---------------------------------------------------------
        # VOLATILITY GATE: Protect the Hedge from Chop
        # ---------------------------------------------------------
        # If trends conflict AND momentum is dead (RSI between 45 and 55)
        if micro_data["decision"] != macro_data["decision"] and (45 <= micro_data["raw_rsi"] <= 55):
            return {
                "symbol": target_symbol, 
                "current_price": micro_data["price"],
                "final_ai_verdict": "RANGING", # Frontend should catch this and disable the execution button
                "ai_reasoning": "VOLATILITY WARNING: Macro and Micro trends are conflicting, and RSI is entirely neutral. Hedging in this environment is high risk as price will likely chop and fail to hit Take Profit on either side."
            }

        # 4. Feed everything to the AI if it passes the Volatility Gate
        final_verdict, ai_reasoning = ask_ollama(
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
            "current_price": micro_data["price"],
            "analysis_timeframes": {
                "micro_entry": micro_tf_data["text"],
                "macro_trend": macro_tf_data["text"]
            },
            "macro_technical_state": {
                "decision": macro_data["decision"],
                "basis": macro_data["basis"],
                "process": macro_data["process"]
            },
            "micro_technical_state": {
                "decision": micro_data["decision"],
                "basis": micro_data["basis"],
                "process": micro_data["process"]
            },
            "news_context": headlines,
            "ai_reasoning": ai_reasoning,
            "final_ai_verdict": final_verdict
        }
        
    except Exception as e:
        return {"error": str(e)}