import myalgo.broker as broker
import myalgo.strategy as strategy
import myalgo.order as order
from myalgo.feed.sqlitefeed import SQLiteFeed
import datetime
import pandas as pd
import numpy as np
from queue import Queue
import DDPG

INSTRUMENT = "USDJPY"
EFFCIENCY_PERIOD = 60
INITIAL_CASH = 1000000
START = datetime.datetime(2013, 1, 1, 0, 0, 0)
END = datetime.datetime(2014, 1, 1, 0, 0)

brain = DDPG.DDPG(2, s_dim=11, s_length=EFFCIENCY_PERIOD, a_bound=1.0)

"""
    action: ( in_amount, out_amount)
    state: ( ask_close , bid_close , quantity , equity)
"""

i = 0


class MyStrategy(strategy.BaseStrategy):

    def __init__(self, broker):
        super(MyStrategy, self).__init__(broker)

        self.history_state = np.empty([0, 11])
        self.near_data = None
        self.done_pos = set()

        self.__last_s_a = None

        self.current_Trade = None
        self.memories = []
        self.use_event_datetime_logs = True

        self.profit_trade = 0
        self.loss_trade = 0

        self.i = 0
        self.gamma = 0.5

        self.rewards = 0
        self.last_in_price = None

    def update_history(self, bars):
        bar = bars[INSTRUMENT]
        state = bar.dict
        del state["start_date"]
        del state["end_date"]
        state["quantity"] = self.broker.quantities[INSTRUMENT]
        state["equity"] = self.broker.equity
        state["last_in_price"] = self.last_in_price if self.last_in_price is not None else state["ask_close"]
        current_state = np.array([v for v in state.values()])

        self.history_state = np.append(self.history_state, np.array([current_state]), axis=0)

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
            self.current_Trade: strategy.position.Position = self.enterLong(INSTRUMENT, int(use_cash / quantity))

        if should_sell and self.current_Trade is not None and self.current_Trade.getEntryOrder().is_filled:
            self.current_Trade.exitMarket(True)
            self.current_Trade = None


        self.update_memory(current_state, action)

    def onEnterOk(self, position: strategy.position.Position):
        super().onEnterOk(position)

        entry: order.MarketOrder = position.getEntryOrder()

        self.last_in_price = entry.avg_fill_price

    def onEnterCanceled(self, position: strategy.position.Position):
        super().onEnterCanceled(position)

    def onExitOk(self, position: strategy.position.Position):
        super().onExitOk(position)
        rate = position.getReturn()
        self.rewards += rate
        if rate > 0:
            self.profit_trade += 1
        if rate < 0:
            self.loss_trade += 1

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

        self.i += 1

        if self.i % 1000 == 0:
            done = (self.profit_trade + self.loss_trade)
            rate = (self.profit_trade / done) if done is not 0 else 0
            self.logger.info(
                'cash:%.3f\teq:%.3f\t bar passed:%d win_rate:%.3f' % (self.broker.cash(), self.broker.equity, self.i,
                                                                      rate))

            # self.profit_trade = 0
            # self.loss_trade = 0


bar_feed = SQLiteFeed(file_name='sqlite', table_name='bins')
bar_feed.load_data([INSTRUMENT], START, END)
backtest = broker.backtest.BackTestBroker(INITIAL_CASH, bar_feed,
                                          commission=broker.commission.TradePercentage(2.5 / 1e+6))

my = MyStrategy(backtest)

my.run()
