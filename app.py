from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tradingview_ta import TA_Handler, Interval
import feedparser

app = FastAPI()

# This allows your Next.js frontend to request data without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_high_impact_news():
    # Grabs the latest headlines from a free forex news feed
    feed = feedparser.parse("https://www.forexlive.com/feed/news")
    
    # Keywords that cause massive, unpredictable XAUUSD spikes
    danger_words = ["cpi", "fed", "powell", "inflation", "nfp", "rate", "interest"]
    
    # Check the top 5 most recent headlines
    for entry in feed.entries[:5]:
        title = entry.title.lower()
        if any(word in title for word in danger_words):
            return True
            
    return False

@app.get("/api/signal")
def get_market_signal():
    try:
        # Step 1: Safety Check (Temporarily disabled for testing)
        # if check_high_impact_news():
        #     return {
        #         "symbol": "XAUUSD", 
        #         "recommendation": "WAIT", 
        #         "reason": "High impact news detected"
        #     }

        # Step 2: Technical Consensus
        handler = TA_Handler(
            symbol="XAUUSD",
            screener="cfd",     # <--- Change this to cfd
            exchange="OANDA",   
            interval=Interval.INTERVAL_1_HOUR
        )
        
        analysis = handler.get_analysis()
        
        # This will return "BUY", "SELL", "STRONG_BUY", "STRONG_SELL", or "NEUTRAL"
        summary = analysis.summary["RECOMMENDATION"]
        
        return {
            "symbol": "XAUUSD", 
            "recommendation": summary, 
            "reason": "Technical consensus"
        }
        
    except Exception as e:
        return {"error": str(e)}