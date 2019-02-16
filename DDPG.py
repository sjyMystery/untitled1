import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)
config = tf.ConfigProto(allow_soft_placement=True, device_count={"cpu":48})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0


class DDPG(object):
    def __init__(self, a_dim, s_dim, s_length, a_bound, tau=0.01, memory_cap=320, batch_size=64, gamma=0.9,
                 lr_a=0.001,
                 lr_c=0.003):
        self.memory_a = np.zeros((memory_cap, a_dim), dtype=np.float32)
        self.memory_s_ = np.zeros((memory_cap, s_length, s_dim), dtype=np.float32)
        self.memory_s = np.zeros((memory_cap, s_length, s_dim), dtype=np.float32)
        self.memory_r = np.zeros((memory_cap, 1), dtype=np.float32)

        self.pointer = 0
        self.sess = tf.Session(config=config)
        self.__can_train = False

        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.memory_cap = memory_cap
        self.tau = tau

        self.s_length = s_length
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        self.S = tf.placeholder(tf.float32, [None, s_length, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_length, s_dim], 's_')

        self.F_S = self.__build_state(self.S)
        self.F_S_ = self.__build_state(self.S_)

        self.s_dim_ = self.F_S.shape[1]

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = self._build_a(self.F_S, )

        q = self._build_c(self.F_S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.F_S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.F_S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + self.gamma * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_cap, size=self.batch_size)
        bs = self.memory_s[indices, :]
        ba = self.memory_a[indices, :]
        br = self.memory_r[indices, :]
        bs_ = self.memory_s_[indices, :]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    @property
    def can_train(self):
        return self.__can_train

    def store_transition(self, s, a, r, s_):
        index = self.pointer % self.memory_cap  # replace the old memory with new memory

        self.memory_a[index, :] = a

        self.memory_s[index, :, :] = s
        self.memory_s_[index, :, :] = s_
        self.memory_r[index] = r

        self.pointer += 1

        if self.pointer >= self.memory_cap:
            self.__can_train = True

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            d1 = tf.layers.dense(s, 32, activation=tf.nn.relu, name='d1', trainable=trainable)
            dr1 = tf.layers.dropout(d1, 0.5, name="actor_dropout_1")
            d2 = tf.layers.dense(dr1, 64, activation=tf.nn.relu, name='actor_dense_2', trainable=trainable)
            dr2 = tf.layers.dropout(d2, 0.5, name="actor_dropout_2")
            d3 = tf.layers.dense(dr2, 32, activation=tf.nn.relu, name='actor_dense_3', trainable=trainable)
            dr3 = tf.layers.dropout(d3, 0.5, name="actor_dropout_3")
            b = tf.layers.dense(dr3, self.a_dim, activation=tf.nn.sigmoid, name='b', trainable=trainable)
            return tf.multiply(b, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 16

            w1_s = tf.get_variable('w1_s', [self.s_dim_, n_l1], trainable=trainable, dtype=tf.float32)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable, dtype=tf.float32)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable, dtype=tf.float32)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # dense_1 = tf.layers.dense(net, 128, trainable=trainable)
            # dp1 = tf.layers.dropout(dense_1, 0.25)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    #
    # def lstm(self, X, input_dim, time_step, rnn_unit=128):
    #     with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
    #         weights = {
    #             'in': tf.Variable(tf.random_normal([input_dim, rnn_unit])),
    #             'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    #         }
    #         biases = {
    #             'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    #             'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    #         }
    #
    #         w_in = weights['in']
    #         b_in = biases['in']
    #         input = tf.reshape(X, [-1, input_dim])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    #         input_rnn = tf.matmul(input, w_in) + b_in
    #         input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    #         cell = tf.nn.rnn_cell.LSTMCell(rnn_unit)
    #         # cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
    #         output_rnn, final_states = tf.keras.layers.(cell, input_rnn,
    #                                                      dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    #         output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    #         w_out = weights['out']
    #         b_out = biases['out']
    #         pred = tf.matmul(output, w_out) + b_out
    #     return pred, final_states

    def __build_state(self, s, trainable=True):
        with tf.variable_scope('STATE', reuse=tf.AUTO_REUSE):

            flatten = tf.layers.flatten(s, name='state_flatten')
            dense1 = tf.layers.dense(flatten, units=128, activation=tf.nn.relu, name='lstm_dense1_s')
            dense2 = tf.layers.dense(dense1, units=32, name='lstm_dense2_s')
        return dense2
