import feedparser
import json
from urllib.parse import quote


def fetch_google_news_india(stock_name):
    """Fetches news articles for a stock from Google News RSS."""
    encoded_stock_name = quote(f"{stock_name} NSE Stocks")
    url = f"https://news.google.com/rss/search?q={encoded_stock_name}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    
    return [{"title": entry.title, "date": entry.published} for entry in feed.entries]

def save_to_json(data, filename):
    """Saves extracted articles to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Example Usage
    news_data = []                                  
    stocks = ["RELIANCE", "TATA STEEL","HDFC"]
    for stock in stocks:
        news_data = fetch_google_news_india(stock)
        # Save to JSON without link
        save_to_json(news_data, f"news/{stock.replace(' ', '_')}_news.json")
