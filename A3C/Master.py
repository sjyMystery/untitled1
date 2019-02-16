import tensorflow as tf
from A3C.Net import Net
from A3C.Worker import Worker
from A3C.str import GLOBAL_NET_SCOPE

import threading


class Master:
    def __init__(self, make_env, lr_actor=0.0001, lr_critic=0.001, beta=0.01, gamma=0.9,
                 train_ep=5, time_length=24 * 60 * 7,
                 update_global_iter=5, n_workers=8):
        self.__eps = []
        self.__sess = tf.Session()

        self.__train_ep = train_ep

        env = make_env()
        n_state = env.observation_space.shape[0]
        n_action = env.action_space.shape[0]
        a_range = [env.action_space.low, env.action_space.high]

        self.__opt_a = tf.train.RMSPropOptimizer(lr_actor, name='RMSPropA')
        self.__opt_c = tf.train.RMSPropOptimizer(lr_critic, name='RMSPropC')
        self.__global_ac = Net(GLOBAL_NET_SCOPE, n_state=n_state, n_action=n_action, a_range=a_range,
                               sess=self.__sess, op_actor=self.__opt_a, time_length=time_length,
                               op_critic=self.__opt_c, beta=beta)  # we only need its params
        self.__workers = []
        # Create worker
        for i in range(n_workers):
            i_name = 'A3C_Worker_%i' % i  # worker name
            self.__workers.append(
                Worker(master=self, name=i_name, make_env=make_env, gamma=gamma,
                       op_actor=self.__opt_a,
                       op_critic=self.__opt_c, beta=beta, update_global_iter=update_global_iter,
                       time_length=time_length,
                       global_ac=self.__global_ac))

        self.__coord = tf.train.Coordinator()
        self.__sess.run(tf.global_variables_initializer())

        self.__worker_threads = []

    def push_ep(self, ep):
        self.__eps.append(ep)

    @property
    def session(self):
        return self.__sess

    def run(self):
        for worker in self.__workers:
            t = threading.Thread(target=(lambda: worker.work()))
            t.start()
            self.__worker_threads.append(t)

        self.__coord.join(self.__worker_threads)

    @property
    def coord(self):
        return self.__coord

    @property
    def train_ep(self):
        return self.__train_ep
