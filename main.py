import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def download_tickers_tidy(tickers, d):
    start_date = datetime.now() - timedelta(days=d)
    end_date = datetime.now()
    
    # Download all tickers at once, keep Adj Close
    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False
    )
    
    # Flatten MultiIndex from yfinance
    df = df.stack(level=0).reset_index()
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    return df

sp500_tickers = get_sp500_tickers()
print("First 10 tickers:", sp500_tickers[:10])

data_tidy = download_tickers_tidy(sp500_tickers[:10], 365)
print(data_tidy.head())
