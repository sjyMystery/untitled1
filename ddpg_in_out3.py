import myalgo.broker as broker
import myalgo.strategy as strategy
import myalgo.order as order
from myalgo.feed.sqlitefeed import SQLiteFeed
import datetime
import pandas as pd

import numpy as np
from collections import deque
from multiprocessing import Process, Pool

import DDPG

INSTRUMENT = "USDJPY"
EFFCIENCY_PERIOD = 60*24*7
INITIAL_CASH = 1000000
START = datetime.datetime(2014, 1, 1, 0, 0, 0)
END = datetime.datetime(2015, 1, 1, 0, 0, 0)

brain = DDPG.DDPG(2, s_dim=9, s_length=EFFCIENCY_PERIOD, a_bound=1.0, memory_cap=2048, batch_size=64)
"""
    action: ( in_amount, out_amount)
    state: ( ask_close , bid_close , quantity , equity)
"""

i = 0


class MyStrategy(strategy.BaseStrategy):

    def __init__(self, broker):
        super(MyStrategy, self).__init__(broker)

        self.history_state = np.empty([0, 9])
        self.near_data = None
        self.done_pos = set()

        self.__last_s_a = None

        self.current_Trade = None
        self.memories = []
        self.use_event_datetime_logs = True

        self.profit_trade = 0
        self.loss_trade = 0

        self.j, self.j_ = 0, 0

        self.gamma = 0.5

        self.rewards = 0
        self.last_in_price = None

        self.pool_train = Pool(16)

        self.last100 = deque([])

    def update_history(self, bars):
        bar = bars[INSTRUMENT]
        state = bar.dict
        del state["start_date"]
        del state["end_date"]
        del state["ask_open"]
        del state["bid_open"]
        state["quantity"] = self.broker.quantities[INSTRUMENT]
        state["equity"] = self.broker.equity
        state["last_in_price"] = self.last_in_price if self.last_in_price is not None else state["ask_close"]
        current_state = np.array([v for v in state.values()])

        self.history_state = np.append(self.history_state, np.array([current_state]), axis=0)

        self.j += 1
        if (self.j % 1000 == 0):
            done = (self.profit_trade + self.loss_trade)
            rate = (self.profit_trade / done) if done is not 0 else 0
            self.logger.info(
                'cash:%.3f\teq:%.3f\t bar passed:%d win_rate:%.3f w:%d l:%d' % (
                    self.broker.cash(), self.broker.equity, self.j,
                    rate, self.profit_trade, self.loss_trade))

    def update_memory(self, current_state, current_action):
        if self.__last_s_a is not None:
            pre_state, pre_action = self.__last_s_a
            reward = self.rewards
            self.rewards = 0
            brain.store_transition(pre_state, pre_action, reward, current_state)

        if brain.can_train:
            brain.learn()
        # self.gamma = self.gamma - 0.00001 if self.gamma > 0 else 0

        self.__last_s_a = (current_state, current_action)

    def onBars(self, dateTime, bars):

        bar = bars[INSTRUMENT]
        self.update_history(bars)

        if self.history_state.shape[0] > EFFCIENCY_PERIOD:
            self.ai_do(bar.out_price)

    def ai_do(self, out_price):
        current_state = self.history_state[-EFFCIENCY_PERIOD:, ]
        action = brain.choose_action(s=current_state)
        action = np.clip(np.random.normal(action, self.gamma), 0, 1)

        should_sell = np.random.choice([0, 1], 1, p=[action[0], 1 - action[0]])[0]
        should_buy = np.random.choice([0, 1], 1, p=[action[1], 1 - action[1]])[0]

        if should_buy and self.current_Trade is None:
            use_cash = self.broker.cash() * 0.9
            quantity = use_cash / out_price
            self.enterLong(INSTRUMENT, int(use_cash / quantity))

        if should_sell and self.current_Trade is not None and self.current_Trade.getEntryOrder().is_filled:
            self.current_Trade.exitMarket()
            self.current_Trade = None

        self.update_memory(current_state, action)

    def onEnterOk(self, position: strategy.position.Position):
        super().onEnterOk(position)

        entry: order.MarketOrder = position.getEntryOrder()

        self.last_in_price = entry.avg_fill_price

        self.current_Trade = position

    def onEnterCanceled(self, position: strategy.position.Position):
        super().onEnterCanceled(position)

    def onExitOk(self, position: strategy.position.Position):
        super().onExitOk(position)
        returns = position.getReturn()
        rate = returns
        self.rewards += rate  # / (self.j - self.j_ + 1) if rate > 0 else 0
        self.j_ = self.j
        if rate > 0:
            self.profit_trade += 1
        if rate < 0:
            self.loss_trade += 1

        done = (self.profit_trade + self.loss_trade)

        rate = (self.profit_trade / done) if done is not 0 else 0

        self.last100.append(position)
        if self.last100.__len__() >= 100:
            self.last100.popleft()




        if done%100 == 0:
            last100_rate = 0
            profit = 0
            amounts = 0
            for position in self.last100:
                returns_ = position.getReturn()
                quantity = position.getEntryOrder().quantity
                last100_rate += 1 if returns_ > 0 else 0
                profit += returns_ * quantity
                amounts += quantity

            profit_rate = profit / (amounts if amounts > 0 else 1)
            self.logger.info(
            'cash:%.3f\teq:%.3f\t trade done:%d win_rate:%.3f last_100_win_rate:%d last_100_profit:%.3f (%.3f/%.3f)' % (
                self.broker.cash(), self.broker.equity, done,
                rate, last100_rate, profit_rate*100, profit, amounts))

    def onExitCanceled(self, position):
        super().onExitCanceled(position)

    def onStart(self):
        super().onStart()

        self.logger.info('start trade!')

    def onFinish(self, bars):
        super().onFinish(bars)

        self.logger.info(self.broker.equity)

    def onIdle(self):
        super().onIdle()

    def onOrderUpdated(self, order):
        super().onOrderUpdated(order)

        # self.profit_trade = 0
        # self.loss_trade = 0


bar_feed = SQLiteFeed(file_name='sqlite', table_name='bins')
bar_feed.load_data([INSTRUMENT], START, END)
backtest = broker.backtest.BackTestBroker(INITIAL_CASH, bar_feed,
                                          commission=broker.commission.TradePercentage(0))

my = MyStrategy(backtest)

my.run()
