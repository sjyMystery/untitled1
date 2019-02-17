import tensorflow as tf
from A3C import Master
from TradeEnv import Env
import datetime
import numpy as np
import pandas as pd

tf.flags.DEFINE_integer('workers', 4, """co-workers num""", 1)
tf.flags.DEFINE_string('instrument', "USDJPY", """name of instruments""")
tf.flags.DEFINE_integer('per_bin', 1000, """log state per bin""", 1000)
tf.flags.DEFINE_integer('per_trade', 100, """log state per trade""", 100)
tf.flags.DEFINE_integer('time_length', 60*24*7, """time step length""", 1)
tf.flags.DEFINE_integer('eps', 100, """""")
tf.flags.DEFINE_integer('update_steps', 5, """""""")
tf.flags.DEFINE_string('start_date', '2012-01-01', """train begin date""")
tf.flags.DEFINE_string('end_date', '2013-01-01', """train end date""")

start_date = datetime.datetime.strptime(tf.flags.FLAGS.start_date, "%Y-%m-%d")
end_date = datetime.datetime.strptime(tf.flags.FLAGS.end_date, "%Y-%m-%d")

update_steps = tf.flags.FLAGS.update_steps
eps = tf.flags.FLAGS.eps


def action_convert(action):
    action = np.clip(np.random.normal(action, 0.9), 0, 1)
    should_sell = np.random.choice([0, 1], 1, p=[action[0], 1 - action[0]])[0] == 1
    should_buy = np.random.choice([0, 1], 1, p=[action[1], 1 - action[1]])[0] == 1
    return [should_sell, should_buy]


env = Env(tf.flags.FLAGS.instrument, 100000, action_convert=action_convert,
          log_per_bin=tf.flags.FLAGS.per_bin,
          log_per_trade=tf.flags.FLAGS.per_trade,
          start_date=start_date,
          end_date=end_date)
a3c = Master(env.clone, n_workers=tf.flags.FLAGS.workers, time_length=tf.flags.FLAGS.time_length, train_ep=eps,
             update_global_iter=update_steps)
a3c.run()
