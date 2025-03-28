import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor

CHROME_DRIVER_PATH = "/Users/ginger/Developer/techkriti/chromedriver-mac-arm64/chromedriver"

def get_driver():
    """Creates and returns a headless Selenium WebDriver instance."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920x1080")
    return webdriver.Chrome(service=Service(CHROME_DRIVER_PATH), options=options)

def scrape_yahoo_finance_news(url):
    """Scrapes Yahoo Finance news article links and titles."""
    driver = get_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "li.stream-item"))
        )
        for _ in range(10):  # Improved scrolling logic
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight / 2);")
            time.sleep(2)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        news_items = soup.find_all('li', class_='story-item')
        articles = []
        seen_titles = set()
        
        for item in news_items:
            title_elem = item.find('h3')
            link_elem = item.find('a')
            if title_elem and link_elem and link_elem.has_attr('href'):
                title = title_elem.get_text(strip=True)
                if title not in seen_titles:
                    seen_titles.add(title)
                    articles.append({'title': title, 'link': link_elem['href']})
        
        return articles
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []
    finally:
        driver.quit()

def scrape_article(article):
    """Scrapes full article text and date from a given URL."""
    try:
        response = requests.get(article['link'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        article_text = " ".join(p.get_text() for p in paragraphs if p.get_text()!="Oops, something went wrong")
        
        # Extract date
        date_elem = soup.find('time', class_='byline-attr-meta-time')
        article_date = date_elem['datetime'] if date_elem else None
        
        return {
            "title": article["title"], 
            "article": article_text,
            "date": article_date
        }
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {article['link']}: {e}")
        return None

def scrape_articles_parallel(articles, max_threads=5):
    """Uses multithreading to scrape articles in parallel."""
    scraped_data = []
    with ThreadPoolExecutor(max_threads) as executor:
        results = executor.map(scrape_article, articles)
        for result in results:
            if result:
                scraped_data.append(result)
    return scraped_data

def save_to_json(data, filename):
    """Saves extracted articles to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_ticker(ticker):
    """Processes a ticker by scraping and saving its news articles."""
    url = f'https://finance.yahoo.com/quote/{ticker}/news/'
    articles = scrape_yahoo_finance_news(url)
    
    if articles:
        full_articles = scrape_articles_parallel(articles, max_threads=10)
        save_to_json(full_articles, f"{ticker}_news.json")
        print(f"Saved {len(full_articles)} articles for {ticker}.")
    else:
        print(f"No articles found for {ticker}.")

if __name__ == "__main__":
    # Example usage
    ticker = "IBM"
    process_ticker(ticker)