from A3C.Net import Net
import numpy as np


class Worker:
    def __init__(self, master, name, make_env, op_actor, op_critic, beta=0.2,
                 gamma=0.9,
                 update_global_iter=5,
                 time_length=64,
                 global_ac=None):
        self.name = name
        self.env = make_env()

        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.shape[0]
        a_range = self.env.action_space.range

        self.AC = Net(name, n_state, n_action, a_range, master.session, op_actor, op_critic, time_length,beta, global_ac)
        self.__sess = master.session
        self.__update_global_iter = update_global_iter
        self.__gamma = gamma
        self.__master = master

    def work(self):
        total_step = 1
        ep = 0
        buffer_s, buffer_a, buffer_r = [], [], []

        while not self.__master.coord.should_stop() and ep < self.__master.train_ep:
            s = self.env.reset()

            assert s is not None, "Get None State"
            ep_r = 0
            rnn_state = self.__sess.run(self.AC.init_state)  # zero rnn state at beginning
            keep_state = rnn_state.copy()  # keep rnn state for updating global net
            done = False
            while not done:

                a, rnn_state_ = self.AC.choose_action(s, rnn_state)  # get the action and next rnn state
                # if(total_step>1000):
                #     print(a,rnn_state_)
                s_, r, done = self.env.step(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize

                if total_step % self.__update_global_iter == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = \
                            self.__sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[
                                0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.__gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()  # replace the keep_state as the new initial rnn state_

                s = s_
                rnn_state = rnn_state_  # renew rnn state
                total_step += 1

            ep += 1
