import yfinance as yf
msft = yf.Ticker("MSFT")
msft.info
data = yf.download("^TNX SPY QQQ KWEB", start="2020-01-01", end="2021-10-14")
data.to_csv("/Users/guoqiong/life/invest/stock/yahoo_data/etf.txt")
