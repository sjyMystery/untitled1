import datetime
import pandas as pd

from myalgo.bar import Bar
from myalgo.feed import BarFeed

url = 'http://candledata.fxcorporate.com/'  ##This is the base url
periodicity = 'm1'  ##periodicity, can be m1, H1, D1
url_suffix = '.csv.gz'  ##Extension of the file name
symbol = 'EURUSD'  ##symbol we want to get tick data
start_dt = datetime.date(2014, 1, 1)  ##random start date
end_dt = datetime.date(2014, 1, 7)  ##random end date


def fetch_data(symbol='EURUSD', start_dt=datetime.date(2014, 1, 1), end_dt=datetime.date(2014, 1, 7)):
    start_wk = start_dt.isocalendar()[1]  ##find the week of the year for the start
    end_wk = end_dt.isocalendar()[1]  ##find the week of the year for the end
    start_year = (start_dt.isocalendar()[0])  ##pull out the year of the start
    end_year = (end_dt.isocalendar()[0])  ##pull out the year of the start

    data = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        for i in range(start_wk, end_wk):
            url_data = url + periodicity + '/' + symbol + '/' + str(year) + '/' + str(i) + url_suffix
            tempdata = pd.read_csv(url_data, compression='gzip')
            data = pd.concat([data, tempdata])

    i = 0

    bars = []

    for k, row in data.iterrows():
        start_date = datetime.datetime.strptime(row["DateTime"], '%d/%m/%Y %H:%M:%S.000')
        end_date = start_date + datetime.timedelta(seconds=1)

        bar = Bar(ask_low=row["AskLow"], start_date=start_date, end_date=end_date,
                  ask_high=row["AskHigh"], ask_open=row["AskOpen"], ask_close=row["AskClose"],
                  bid_high=row["BidHigh"], bid_open=row["BidOpen"], bid_close=row["BidClose"], bid_low=row["BidLow"],
                  volume=1
                  )
        bars.append({symbol: bar})

        feed = BarFeed(bars=bars)

        return feed
