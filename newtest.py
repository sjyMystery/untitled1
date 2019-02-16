from myalgo.feed.sqlitefeed import SQLiteFeed
import datetime
from pandas import DataFrame
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.preprocessing import sequence

bar_feed = SQLiteFeed('bins', './sqlite')
bar_feed.load_data(['USDJPY'], datetime.datetime(2013, 1, 1, 0, 0, 0), datetime.datetime(2018, 6, 1, 0, 0, 0))
df = DataFrame([bar['USDJPY'].dict for bar in bar_feed.bars])
generator = sequence.TimeseriesGenerator(
    data=np.array(df[["ask_close", "bid_close", "ask_high", "ask_low", "ask_open", "bid_high", "bid_open", "bid_low"]]),
    targets=np.array(df[["bid_close"]]), length=360, batch_size=128)

model = Sequential()

model.add(LSTM(32, dropout=0.1))
model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.build(input_shape=(None, 360, 8))
model.compile('adam', loss='mape', metrics=['mape'])
model.fit_generator(generator, epochs=10, use_multiprocessing=True, workers=8)
