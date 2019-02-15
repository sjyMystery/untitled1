import tensorflow as tf
from A3C import Master
from TradeEnv import Env
import datetime
import numpy as np

tf.flags.DEFINE_integer('workers', 4, """co-workers num""", 1)
tf.flags.DEFINE_string('instrument', "USDJPY", """name of instruments""")


def action_convert(action):
    should_sell = np.random.choice([0, 1], 1, p=[action[0], 1 - action[0]])[0] == 1
    should_buy = np.random.choice([0, 1], 1, p=[action[1], 1 - action[1]])[0] == 1
    return [should_sell, should_buy]


env = Env(tf.flags.FLAGS.instrument, 100000, action_convert=action_convert, start_date=datetime.datetime(2012, 1, 1, 0, 0, 0),
          end_date=datetime.datetime(2012, 2, 1, 0, 0, 0))
a3c = Master(env.clone, n_workers=tf.flags.FLAGS.workers)

a3c.run()
