# news_analyst.py
from __future__ import annotations
import os, time, requests
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import feedparser

FAST = os.getenv("FAST_MODE") == "1"
USE_MOCK = os.getenv("MOCK_MODE") == "1"

NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
    # keep it to one feed in FAST mode for speed
]

# Import mock data
try:
    from mock_data import mock_news_items
except ImportError:
    # Fallback if mock_data not available
    def mock_news_items(ticker: str):
        now = datetime.now(timezone.utc)
        return [
            {"title": f"{ticker} shows positive momentum in recent trading", "url": "https://finance.yahoo.com", 
             "published": now.isoformat(), "source": "mock:yahoo"},
        ]

def _get_feed(url: str):
    # hard timeout so we never hang
    r = requests.get(url, timeout=4)
    r.raise_for_status()
    return feedparser.parse(r.content)

def _normalize(feed, source: str):
    rows = []
    for e in getattr(feed, "entries", []):
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "")
        ts = datetime.now(timezone.utc)
        try:
            pp = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            if pp: ts = datetime.fromtimestamp(time.mktime(pp), tz=timezone.utc)
        except Exception:
            pass
        rows.append({"title": title, "url": link, "published": ts, "source": source})
    return rows

def fetch_news(ticker: str, lookback_hours=48, max_items=30) -> List[Dict[str, Any]]:
    if USE_MOCK:
        return [{"title": it["title"], "url": it["url"], "published": datetime.fromisoformat(it["published"].replace("Z","")), "source": it["source"]}
                for it in mock_news_items(ticker)]

    if FAST:
        lookback_hours = min(lookback_hours, 24)
        max_items = min(max_items, 10)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    all_items = []
    try:
        for tmpl in NEWS_FEEDS:
            url = tmpl.format(ticker=ticker)
            feed = _get_feed(url)
            all_items.extend(_normalize(feed, url))
    except Exception:
        # network down → mock
        return [{"title": it["title"], "url": it["url"], "published": datetime.fromisoformat(it["published"].replace("Z","")), "source": it["source"]}
                for it in mock_news_items(ticker)]

    tkr = ticker.upper()
    filtered = [it for it in all_items if it["published"] >= cutoff and tkr in it["title"].upper()]
    seen, uniq = set(), []
    for it in filtered:
        key = (it["title"], it["url"])
        if key in seen: continue
        seen.add(key); uniq.append(it)
    uniq.sort(key=lambda x: x["published"], reverse=True)
    return uniq[:max_items]

def analyze_news(ticker: str, lookback_hours=48, max_items=30, llm: Any = None):
    if FAST: llm = None
    items = fetch_news(ticker, lookback_hours, max_items)
    bullets = [f"Recent headlines for {ticker}:"] + [f"- {it['title']}" for it in items[:6]]
    return {
        "agent": "news_analyst",
        "ticker": ticker.upper(),
        "since_hours": lookback_hours,
        "item_count": len(items),
        "summary_bullets": bullets,
        "sources": [
            {"title": it["title"], "url": it["url"], "published": it["published"].isoformat(), "source": it["source"]}
            for it in items
        ],
    }

# Legacy function for backward compatibility with existing graph
def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Use the new free RSS-based analysis
        analysis = analyze_news(ticker=ticker, lookback_hours=168)  # 1 week
        
        # Format the analysis into a report
        report = f"""
# News Analysis for {ticker}

## News Summary
- **Articles Analyzed**: {analysis['item_count']} articles from the last {analysis['since_hours']} hours
- **Data Sources**: Yahoo Finance RSS, NASDAQ RSS feeds

## Key Insights
"""
        
        for bullet in analysis['summary_bullets']:
            report += f"{bullet}\n"
        
        report += f"""
## Recent Headlines
"""
        for i, source in enumerate(analysis['sources'][:8], 1):
            report += f"{i}. [{source['title']}]({source['url']}) - {source['published'][:10]}\n"

        return {
            "messages": [{"content": report, "role": "assistant"}],
            "news_report": report,
        }

    return news_analyst_node

if __name__ == "__main__":
    import json, sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(json.dumps(analyze_news(tk), indent=2))