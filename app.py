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

# --- NEW: Timeframe Mapping Engine ---
# This translates URL parameters into TradingView logic and human-readable text for the AI.
TIMEFRAME_MAP = {
    "5m": {"tv_interval": Interval.INTERVAL_5_MINUTES, "text": "5 minutes"},
    "15m": {"tv_interval": Interval.INTERVAL_15_MINUTES, "text": "15 minutes"},
    "1h": {"tv_interval": Interval.INTERVAL_1_HOUR, "text": "1 hour"},
    "4h": {"tv_interval": Interval.INTERVAL_4_HOURS, "text": "4 hours"},
    "1d": {"tv_interval": Interval.INTERVAL_1_DAY, "text": "1 day"}
}

def get_latest_headlines():
    try:
        feed = feedparser.parse("https://www.forexlive.com/feed/news")
        headlines = [entry.title for entry in feed.entries[:3]]
        return " | ".join(headlines)
    except Exception:
        return "No significant news data available."

def get_exchange_config(symbol):
    if symbol == "XAUUSD":
        return {"screener": "forex", "exchange": "FX_IDC"}
    elif symbol == "BTCUSD":
        return {"screener": "crypto", "exchange": "BINANCE"}
    else:
        return {"screener": "crypto", "exchange": "BINANCE"}

def ask_ollama(symbol, timeframe_text, historical_decision, ta_basis, headlines):
    """Now forces Gemma 3 to explicitly state the timeframe of its prediction."""
    system_prompt = f"""You are a strict, data-driven {symbol} trading algorithmic judge.
    Asset: {symbol}
    Time Horizon: The next {timeframe_text}
    Historical Technical Decision: {historical_decision}
    Technical Basis: {ta_basis}
    Latest Breaking News: {headlines}
    
    You must choose ONLY "BUY" or "SELL".
    
    Rule 1 (The Override): If the news is highly volatile and strongly contradicts the Historical Decision, output the direction the NEWS implies.
    Rule 2 (The Agreement): If the news supports the Historical Decision, output the Historical Decision.
    Rule 3 (The Default): If the news is boring or irrelevant, output the Historical Decision.
    
    You must respond in pure JSON format containing exactly two keys:
    1. "verdict": Exactly one word ("BUY" or "SELL").
    2. "reasoning": A strict 1 to 2 sentence explanation. 
    
    CRITICAL INSTRUCTIONS FOR YOUR REASONING:
    - You MUST explicitly name the asset ({symbol}).
    - You MUST explicitly state the time horizon (e.g., "Over the next {timeframe_text}...").
    - You MUST quote at least one specific keyword or phrase from the Breaking News.
    - You MUST cite at least one specific number from the Technical Basis.
    """

    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "gemma3:4b",   
        "prompt": system_prompt,
        "stream": False,
        "format": "json",       
        "options": {
            "temperature": 0.0, 
            "num_thread": 4     
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        data = response.json()
        raw_response = data.get("response", "{}")
        
        ai_data = json.loads(raw_response)
        
        verdict = ai_data.get("verdict", "").strip().upper()
        reasoning = ai_data.get("reasoning", "No reasoning provided.")
        
        if verdict not in ["BUY", "SELL"]:
            verdict = historical_decision
            
        return verdict, reasoning
        
    except Exception as e:
        print(f"Ollama error: {e}")
        return historical_decision, f"AI offline. Defaulting to historical math for {symbol} over the next {timeframe_text}."

# --- NEW: Endpoint now accepts a 'timeframe' variable ---
@app.get("/api/signal")
def get_market_signal(symbol: str = "BTCUSD", timeframe: str = "1h"):
    try:
        target_symbol = symbol.upper()
        config = get_exchange_config(target_symbol)
        
        # Grab the correct TradingView interval and text mapping. Fallback to 5m if invalid.
        tf_data = TIMEFRAME_MAP.get(timeframe.lower(), TIMEFRAME_MAP["1h"])

        handler = TA_Handler(
            symbol=target_symbol,        
            screener=config["screener"],      
            exchange=config["exchange"],     
            interval=tf_data["tv_interval"]  # Dynamically changing the math timeframe
        )
        
        analysis = handler.get_analysis()
        
        raw_summary = analysis.summary["RECOMMENDATION"]
        buy_votes = analysis.summary["BUY"]
        sell_votes = analysis.summary["SELL"]
        
        if "BUY" in raw_summary:
            historical_decision = "BUY"
        elif "SELL" in raw_summary:
            historical_decision = "SELL"
        else:
            historical_decision = "BUY" if buy_votes > sell_votes else "SELL"

        indicators = analysis.indicators
        rsi_value = round(indicators.get("RSI", 0), 2)
        macd_value = round(indicators.get("MACD.macd", 0), 2)
        ema_20 = round(indicators.get("EMA20", 0), 2)
        current_price = round(indicators.get("close", 0), 2)

        if rsi_value > 70:
            rsi_reason = f"RSI is {rsi_value} (Overbought)."
        elif rsi_value < 30:
            rsi_reason = f"RSI is {rsi_value} (Oversold)."
        else:
            rsi_reason = f"RSI is {rsi_value} (Neutral)."

        macd_reason = f"MACD Bearish ({macd_value})." if macd_value < 0 else f"MACD Bullish ({macd_value})."
        trend_reason = f"Price below 20 EMA." if current_price < ema_20 else f"Price above 20 EMA."
        
        detailed_basis = f"{trend_reason} {macd_reason} {rsi_reason}"

        headlines = get_latest_headlines()

        # Pass the human-readable timeframe text (e.g., "1 hour") to the AI
        final_verdict, ai_reasoning = ask_ollama(target_symbol, tf_data["text"], historical_decision, detailed_basis, headlines)
        
        return {
            "symbol": target_symbol, 
            "timeframe_analyzed": tf_data["text"],
            "current_price": current_price,
            "historical_decision": historical_decision, 
            "technical_basis": detailed_basis,
            "news_context": headlines,
            "ai_reasoning": ai_reasoning,
            "final_ai_verdict": final_verdict
        }
        
    except Exception as e:
        return {"error": str(e)}