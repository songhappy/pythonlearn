import yfinance as yf
msft = yf.Ticker("MSFT")
msft.info
import datetime

today = datetime.date.today()
tomorrow = datetime.date.today() + datetime.timedelta(days=1)

data = yf.download("^SPX ^IXIC ^DJI ^TNX SPY QQQ IWM SOXX", start="2020-01-01", end=tomorrow)
data.to_csv("/Users/guoqiong/life/invest/stock/yahoo_data/etf.txt")
