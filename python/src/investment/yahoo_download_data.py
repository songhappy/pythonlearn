import yfinance as yf
msft = yf.Ticker("MSFT")
msft.info
import datetime
# move the following code into main function and add logging

if __name__ == "__main__":
    today = datetime.date.today()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tickers = ["^SPX", "^IXIC", "^DJI", "^TNX", "SPY", "QQQ", "IWM","SOXX", "SPY", "IVV", "VOO", "QQQ", "IWM", "VTI", "XLF", "XLC", "XLB", "XLE", "XLY", "XLP", "EEM", "EFA", "VEA", "VWO", "AGG", "BND", "LQD", "GLD", "SLV", "DBC", "ARKK", "TAN", "BOTZ"]
    data_file ="/home/arda/intelWork/data/yahoo_stock/yahoo_data.csv"
    data = yf.download(tickers, start="2020-01-01", end=tomorrow)
    data.to_csv(data_file)

