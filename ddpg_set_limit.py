import myalgo.broker as broker
import myalgo.strategy as strategy
from myalgo.feed.sqlitefeed import SQLiteFeed
import datetime
import pandas as pd
import numpy as np
from queue import Queue
import DDPG

INSTRUMENT = "USDJPY"
EFFCIENCY_PERIOD = 0
INITIAL_CASH = 100000

brain = DDPG.DDPG(3, 10, 1.0)

"""
    action: ( in_amount, out_amount)
    state: ( ask_close , bid_close , quantity , equity)
"""

i = 0


class MyStrategy(strategy.BaseStrategy):

    def __init__(self, broker):
        super(MyStrategy, self).__init__(broker)

        self.history = []
        self.near_data = None
        self.done_pos = set()

        self.__last_s_a = None

        self.memories = []
        self.use_event_datetime_logs = True

        self.profit_trade = 0
        self.loss_trade = 0

        self.i = 0

    def update_history(self, bars):
        bar = bars[INSTRUMENT]
        state = bar.dict
        del state["start_date"]
        del state["end_date"]
        state["quantity"] = self.broker.quantities[INSTRUMENT]
        state["equity"] = self.broker.equity

        self.history.append(state)

        if EFFCIENCY_PERIOD != 0 and len(self.history) <= EFFCIENCY_PERIOD:
            self.near_data = None
        else:
            self.near_data = pd.DataFrame([bar for bar in self.history[-EFFCIENCY_PERIOD:]])

    @property
    def near_np_data(self):
        return np.array(self.near_data[
                            ["ask_close", "ask_high", "ask_low", "ask_open", "bid_close", "bid_high", "bid_low",
                             "bid_open"]])

    def update_memory(self, current_state, current_action):
        if self.__last_s_a is not None:
            pre_state, pre_action = self.__last_s_a
            reward = self.broker.equity / INITIAL_CASH
            brain.store_transition(pre_state, pre_action, reward, current_state)

        if brain.can_train:
            brain.learn()

        self.__last_s_a = (current_state, current_action)

    def onBars(self, dateTime, bars):

        bar = bars[INSTRUMENT]
        self.update_history(bars)

        current_state = np.array([v for k, v in self.history[-1].items()])
        action = brain.choose_action(s=current_state)

        buy_precent = action[0]
        up_percent = action[1]
        down_percent = action[2]
        use_cash = self.broker.cash() * buy_precent * 0.9

        buy_quant = int(use_cash / bars[INSTRUMENT].in_price)
        if buy_quant > 10:
            position: strategy.position.Position = self.enterLong('USDJPY', buy_quant)
            position.up_percent = up_percent
            position.down_percent = down_percent
        else:
            self.logger.info('cannot enter')
        for p in self.active_positions:
            if p.getEntryOrder().is_filled and not p.exitActive():
                if p in self.done_pos and (bar.out_price >= p.up_price or bar.out_price <= p.down_price):
                    p.exitMarket()

        self.update_memory(current_state, action)

    def onEnterOk(self, position):
        super().onEnterOk(position)
        price = position.getEntryOrder().avg_fill_price

        position.up_price = price * position.up_percent
        position.down_price = price * position.down_percent
        self.done_pos.add(position)

    def onEnterCanceled(self, position):
        super().onEnterCanceled(position)

    def onExitOk(self, position):
        super().onExitOk(position)

        self.done_pos.remove(position)

        exit_price = position.getExitOrder().avg_fill_price
        enter_price = position.getEntryOrder().avg_fill_price

        rate = (exit_price - enter_price) / enter_price

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
            self.logger.info(
                'cash:%.3f\teq:%.3f\t bar passed:%d win_rate:%.3f' % (self.broker.cash(), self.broker.equity, self.i,
                                                                      self.profit_trade / (
                                                                                  self.profit_trade + self.loss_trade)))

            self.profit_trade=0
            self.loss_trade=0
bar_feed = SQLiteFeed(file_name='sqlite',table_name='bins')
bar_feed.load_data(['USDJPY'], datetime.datetime(2013, 1, 1, 0, 0, 0), datetime.datetime(2013, 2, 1, 0, 0, 0))
backtest = broker.backtest.BackTestBroker(INITIAL_CASH, bar_feed, commission=broker.commission.FixedPerTrade(0.0))

my = MyStrategy(backtest)

my.run()


