import DDPG
import numpy as np
from TradeEnv import Env
import tensorflow as tf
import datetime

tf.flags.DEFINE_integer('workers', 4, """co-workers num""", 1)
tf.flags.DEFINE_string('instrument', "USDJPY", """name of instruments""")
tf.flags.DEFINE_integer('per_bin', 1000, """log state per bin""", 1000)
tf.flags.DEFINE_integer('per_trade', 100, """log state per trade""", 100)
tf.flags.DEFINE_integer('time_length', 60 * 24 * 30, """time step length""", 1)
tf.flags.DEFINE_integer('eps', 100, """""")
tf.flags.DEFINE_integer('update_steps', 1000, """""""")
tf.flags.DEFINE_string('start_date', '2012-01-01', """train begin date""")
tf.flags.DEFINE_string('end_date', '2013-01-01', """train end date""")

start_date = datetime.datetime.strptime(tf.flags.FLAGS.start_date, "%Y-%m-%d")
end_date = datetime.datetime.strptime(tf.flags.FLAGS.end_date, "%Y-%m-%d")
instrument = tf.flags.FLAGS.instrument
update_steps = tf.flags.FLAGS.update_steps
max_eps = tf.flags.FLAGS.eps
time_length = tf.flags.FLAGS.time_length

rate = 0.01


def action_convert(action_):
    action_ = np.clip(np.random.normal(action_, rate), 0, 1)
    should_sell = np.random.choice([0, 1], 1, p=[action_[0], 1 - action_[0]])[0] == 1
    should_buy = np.random.choice([0, 1], 1, p=[action_[1], 1 - action_[1]])[0] == 1
    return [should_sell, should_buy]


env = Env(instrument, 100000, start_date=start_date, end_date=end_date)
ddpg = DDPG.DDPG(env.action_space.shape[0], env.observation_space.shape[0], 1, 1.0)

eps = 0

while eps < max_eps:
    state = env.reset()
    done = False
    states = np.empty((time_length, env.observation_space.shape[0]), dtype=np.float32)
    states = np.append(states, [state])
    while not done:
        action = ddpg.choose_action(np.array([state]))

        state_, rewards, done = env.step(action_convert(action))

        if states.shape[0] >= 2:
            ddpg.store_transition([state_], action, rewards, [state])
        if ddpg.can_train:
            ddpg.learn()
        state = state_

    rate *= 0.9
    eps += 1
