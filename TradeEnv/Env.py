import myalgo.broker as broker
from myalgo.feed.sqlitefeed import SQLiteFeed
import numpy as np
from TradeEnv.Strategy import TradeEnvStrategy

from Space import Box, Space
from queue import Queue
import threading


class Env:

    def __init__(self, instrument, initial_cash, action_convert=lambda x: x, log_per_trade=100, log_per_bin=1000,
                 commission=broker.commission.TradePercentage(0), start_date=None, end_date=None, env_name='Env',
                 sql_filename='sqlite', table_name='bins', feed=None):
        if feed is None:
            assert instrument is not None
            assert start_date is not None
            assert end_date is not None
            self.__bar_feed = SQLiteFeed(file_name=sql_filename, table_name=table_name)
            self.__bar_feed.load_data([instrument], start_date, end_date)
        else:
            self.__bar_feed = feed

        self.__name = env_name
        self.__action_convert = action_convert
        self.__backtest_broker = broker.backtest.BackTestBroker(initial_cash, self.__bar_feed, commission=commission)
        self.__instrument = instrument
        self.__initial_cash = initial_cash
        self.__commission = commission

        self.__action_queue = Queue()
        self.__state_queue = Queue()
        self.__str = TradeEnvStrategy(self.__backtest_broker, instrument, env=self, log_per_bin=log_per_bin,
                                      log_per_trade=log_per_trade)

        self.__env_name = env_name
        self.__backtest_thread = threading.Thread(target=self.__str.run, name=env_name + 'BackTest Work Thread')

        self.__started = False

        self.__action_space = Box(0, 1, (2,), float)
        self.__state_space = Space((9,), float)

        self.__log_per_bin = log_per_bin
        self.__log_per_trade = log_per_trade

    @property
    def observation_space(self):
        return self.__state_space

    @property
    def action_space(self):
        return self.__action_space

    @property
    def feed(self):
        return self.__bar_feed

    def start(self):
        """
            step之前必须start
            这是为了保证generator派发的顺序
        :return:
        """

        self.__backtest_thread.start()
        self.__started = True

        type_, result = self.__state_queue.get()

        state, _, __ = result

        return state

    def reset(self, timeout=None):
        if self.__started:
            self.__started = False
            self.__str.stop()
            self.__backtest_thread.join(timeout=timeout)
            self.__bar_feed.reset()
            self.__backtest_thread = threading.Thread(target=self.__str.run, name=self.__env_name + 'BackTest Work Thread')
            self.__action_queue = Queue()
            self.__state_queue = Queue()
        return self.start()

    def step(self, action):
        assert self.__started, "should step after started!"
        self.__action_queue.put(('ACTION', self.__action_convert(action) if action is not None else None))

        type_, state = self.__state_queue.get()

        assert type_ == 'STATE', 'SEND ACTION SHOULD GET STATE'

        return state

    def clone(self, name=None):
        assert not self.__started, "cannot clone when started!"
        return Env(self.__instrument, self.__initial_cash, self.__action_convert, self.__log_per_trade,
                   self.__log_per_bin, self.__commission, env_name=name if name is not None else self.__name,
                   feed=self.__bar_feed.clone())

    def action_from_state(self, state):
        assert state is not None, 'state cannot be none'
        self.__state_queue.put(state)
        return self.__action_queue.get()

    def put_state(self, state):
        assert state is not None, 'state cannot be none'
        self.__state_queue.put(state)
