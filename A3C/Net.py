import numpy as np
import tensorflow as tf
from A3C.str import GLOBAL_NET_SCOPE


class Net:
    def __init__(self, scope, n_state, n_action, a_range, sess, op_actor, op_critic, time_length=64, beta=0.2,
                 global_ac=None):
        self.__n_state = n_state
        self.__cell_size = time_length
        self.__n_action = n_action
        self.__a_range = a_range
        self.__sess = sess
        self.__beta = beta
        self.__op_actor = op_actor
        self.__op_critic = op_critic
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.__n_state], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.__n_state], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, self.__n_action], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * self.__a_range[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = self.__beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), self.__a_range[0],
                                              self.__a_range[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_ac.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_ac.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.__op_actor.apply_gradients(zip(self.a_grads, global_ac.a_params))
                    self.update_c_op = self.__op_critic.apply_gradients(zip(self.c_grads, global_ac.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):  # only critic controls the rnn update
            cell_size = self.__cell_size
            s = tf.expand_dims(self.s, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_c = tf.layers.dense(cell_out, 50, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('actor'):  # state representation is based on critic
            l_a = tf.layers.dense(cell_out, 80, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, self.__n_action, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.__n_action, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.__sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.__sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):  # run by a local
        s = s[np.newaxis, :]
        a, cell_state = self.__sess.run([self.A, self.final_state], {self.s: s, self.init_state: cell_state})
        return a, cell_state
