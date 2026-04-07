import os
import json
import requests
import feedparser
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
        "raw_rsi": rsi # Extracted for the Python Volatility Gate
    }

def ask_ollama(symbol, micro_tf, macro_tf, micro_data, macro_data, headlines):
    system_prompt = f"""You are a quantitative trading AI analyzing {symbol}.
    Asset Profile: {symbol} 
    
    [1] MACRO TREND (Primary Direction - {macro_tf}):
    - Verdict: {macro_data['decision']}
    - Basis: {macro_data['basis']}
    
    [2] MICRO TREND (Entry Momentum - {micro_tf}):
    - Verdict: {micro_data['decision']}
    - Basis: {micro_data['basis']}
    
    [3] CATALYSTS (News): 
    {headlines}
    
    TASK: You must output ONLY "BUY" or "SELL". 
    
    DECISION LOGIC (Follow Strictly):
    1. CONFLUENCE: If Macro and Micro both say BUY, output BUY. If both say SELL, output SELL.
    2. THE TIE-BREAKER: If Macro and Micro disagree, the MACRO TREND ({macro_tf}) always wins unless there is breaking news in the opposite direction. Do not trade against the Macro trend on a whim.
    3. RSI EXTREMES: If the Micro RSI is Overbought (>70), heavily lean towards SELL. If Oversold (<30), heavily lean towards BUY.
    
    Respond strictly in JSON format containing exactly two keys:
    {{
      "verdict": "BUY" or "SELL",
      "reasoning": "Strict 2-sentence explanation citing the Macro trend and RSI."
    }}
    """

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": "gemma3:4b",   
        "prompt": system_prompt,
        "stream": False,
        "format": "json",       
        "options": {"temperature": 0.0, "num_thread": 4} # Temp 0.0 ensures highly logical, repeatable outputs
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        raw_response = data.get("response", "{}")
        ai_data = json.loads(raw_response)
        
        verdict = ai_data.get("verdict", "").strip().upper()
        
        # Strict fallback to prevent hallucinations
        if verdict not in ["BUY", "SELL"]:
            verdict = macro_data["decision"] # Default to MACRO trend if AI fails
            
        return verdict, ai_data.get("reasoning", "AI generated standard signal based on technicals.")
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

        # 3. Fetch News
        headlines = get_latest_headlines()

        # ---------------------------------------------------------
        # VOLATILITY GATE: Protect the Hedge from Chop
        # ---------------------------------------------------------
        # If trends conflict AND momentum is dead (RSI between 45 and 55)
        if micro_data["decision"] != macro_data["decision"] and (45 <= micro_data["raw_rsi"] <= 55):
            return {
                "symbol": target_symbol, 
                "current_price": micro_data["price"],
                "final_ai_verdict": "RANGING", # Frontend should catch this and disable the button
                "ai_reasoning": "VOLATILITY WARNING: Macro and Micro trends are conflicting, and RSI is entirely neutral. Hedging in this environment is high risk as price will likely chop and fail to hit Take Profit on either side."
            }

        # 4. Feed everything to the AI if it passes the Volatility Gate
        final_verdict, ai_reasoning = ask_ollama(
            target_symbol, 
            micro_tf_data["text"], 
            macro_tf_data["text"], 
            micro_data, 
            macro_data, 
            headlines
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