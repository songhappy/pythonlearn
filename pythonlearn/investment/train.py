import pandas as pd

yahoo_etf_input = "/Users/guoqiong/life/invest/stock/yahoo_data/etf.txt"
df = pd.read_csv(yahoo_etf_input)

# fillna with mean
# scale
#
df.show()