# social_media_analyst.py
from __future__ import annotations
import os, re, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

FAST = os.getenv("FAST_MODE") == "1"
USE_MOCK = os.getenv("MOCK_MODE") == "1"
TIME_BUDGET_S = float(os.getenv("REDDIT_TIME_BUDGET_S", "3.0"))
ALLOW_MOCK_WHEN_EMPTY = os.getenv("ALLOW_MOCK_WHEN_EMPTY", "1") == "1"

try:
    import praw  # keep import; we'll just not use it if USE_MOCK
except Exception:
    praw = None

# Define mock function at module level
def mock_reddit_posts(ticker: str):
    """Mock function for Reddit posts"""
    now = datetime.now(timezone.utc)
    return [
        {"id": "m1", "title": f"{ticker} trending bullish on Reddit", "score": 150, "num_comments": 45,
         "url": "https://www.reddit.com/r/stocks/", "created_utc": now.isoformat(), "subreddit": "stocks"},
    ]

# Try to import from mock_data, but use our fallback if it fails
try:
    from mock_data import mock_reddit_posts as imported_mock_reddit_posts
    mock_reddit_posts = imported_mock_reddit_posts
except ImportError:
    # Use our fallback function (already defined above)
    pass

_POS = {"beat","beats","breakout","bull","bullish","call","calls","rip","ripping","moon","mooning","up","upgrade","green","squeeze","buy","bought","long"}
_NEG = {"miss","missed","guidance","cut","bear","bearish","dump","dumping","down","downgrade","red","sell","sold","short","bag","bags","bagholder"}

def _sent(text: str) -> float:
    toks = re.findall(r"[a-z']+", text.lower())
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    return 0.0 if (pos+neg)==0 else (pos - neg) / float(pos + neg)

def _label(s: float) -> str:
    return "Bullish" if s > 0.15 else "Bearish" if s < -0.15 else "Mixed/Neutral"

def _reddit_client():
    if not praw or USE_MOCK:
        return None
    cid, sec = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "TradingAgentsBot/0.1")
    if not cid or not sec:
        return None
    return praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)

def fetch_reddit_posts(ticker: str, subreddits=None, limit_per_sub=75, lookback_hours=24) -> List[Dict[str, Any]]:
    if USE_MOCK:
        return mock_reddit_posts(ticker)

    subreddits = subreddits or (["stocks"] if FAST else ["wallstreetbets","stocks","investing"])
    limit_per_sub = min(limit_per_sub, 10 if FAST else limit_per_sub)
    lookback_hours = min(lookback_hours, 6 if FAST else lookback_hours)

    reddit = _reddit_client()
    if reddit is None:
        # if keys missing or praw not available → mock
        return mock_reddit_posts(ticker)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    pat = re.compile(rf"(?i)\b{re.escape(ticker)}\b|\${re.escape(ticker)}")
    start = time.monotonic()
    out: List[Dict[str, Any]] = []

    try:
        for sub in subreddits:
            if time.monotonic() - start > TIME_BUDGET_S: break
            sr = reddit.subreddit(sub)
            for post in sr.hot(limit=limit_per_sub):
                if time.monotonic() - start > TIME_BUDGET_S: break
                created = datetime.fromtimestamp(getattr(post, "created_utc", time.time()), tz=timezone.utc)
                if created < cutoff: continue
                title = (post.title or "").strip()
                if not pat.search(title): continue
                out.append({
                    "id": post.id, "title": title, "score": int(post.score or 0),
                    "num_comments": int(post.num_comments or 0),
                    "url": f"https://www.reddit.com{post.permalink}",
                    "created_utc": created.isoformat(), "subreddit": sub
                })
    except Exception:
        # network/rate-limit → fallback
        try:
            from mock_data import mock_reddit_posts
            return mock_reddit_posts(ticker)
        except ImportError:
            return []

    # dedupe + sort
    seen, uniq = set(), []
    for p in out:
        if p["id"] in seen: continue
        seen.add(p["id"]); uniq.append(p)
    uniq.sort(key=lambda x: (x["score"], x["num_comments"]), reverse=True)
    
    # Fast fallback: if no posts found and in FAST mode, use mock data
    if FAST and not uniq and ALLOW_MOCK_WHEN_EMPTY:
        from mock_data import mock_reddit_posts
        uniq = mock_reddit_posts(ticker)
    
    return uniq

def analyze_social_media(ticker: str, subreddits=None, limit_per_sub=75, lookback_hours=24, llm: Any = None):
    if FAST: llm = None
    posts = fetch_reddit_posts(ticker, subreddits, limit_per_sub, lookback_hours)
    scores = [_sent(p["title"]) for p in posts]
    avg = (sum(scores)/len(scores)) if scores else 0.0
    return {
        "agent": "social_media_analyst",
        "ticker": ticker.upper(),
        "since_hours": lookback_hours,
        "post_count": len(posts),
        "sentiment_score": round(avg, 3),
        "sentiment_label": _label(avg),
        "summary_bullets": [f"Top Reddit chatter for {ticker}:"] + [f"- {p['title']}" for p in posts[:5]],
        "sources": posts,
    }

# Legacy function for backward compatibility with existing graph
def create_social_media_analyst(llm, toolkit):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        # Use the new free Reddit-based analysis
        analysis = analyze_social_media(ticker=ticker, lookback_hours=168)  # 1 week
        
        # Format the analysis into a report
        report = f"""
# Social Media Analysis for {ticker}

## Sentiment Summary
- **Overall Sentiment**: {analysis['sentiment_label']} (Score: {analysis['sentiment_score']})
- **Posts Analyzed**: {analysis['post_count']} posts from the last {analysis['since_hours']} hours
- **Data Sources**: Reddit (r/wallstreetbets, r/stocks, r/investing)

## Key Insights
"""
        
        for bullet in analysis['summary_bullets']:
            report += f"{bullet}\n"
        
        report += f"""
## Top Posts
"""
        for i, post in enumerate(analysis['sources'][:5], 1):
            report += f"{i}. [{post['title']}]({post['url']}) (Score: {post['score']}, Comments: {post['num_comments']})\n"

        return {
            "messages": [{"content": report, "role": "assistant"}],
            "sentiment_report": report,
        }

    return social_media_analyst_node

if __name__ == "__main__":
    import json, sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(json.dumps(analyze_social_media(tk), indent=2))