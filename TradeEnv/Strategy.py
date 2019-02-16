import myalgo.broker as broker
import myalgo.strategy as strategy
import myalgo.order as order
import numpy as np


class TradeEnvStrategy(strategy.BaseStrategy):
    def __init__(self, broker_: broker.BackTestBroker, instrument: str, env, log_per_trade, log_per_bin):
        super(TradeEnvStrategy, self).__init__(broker_)

        self.__log_per_trade = log_per_trade
        self.__log_per_bin = log_per_bin

        self.history_state = np.empty([0, 9])

        self.__instrument = instrument

        self.current_Trade = None
        self.use_event_datetime_logs = True

        self.rewards = 0
        self.last_in_price = None

        self.exited_positions = []

        self.passed_bars = 0

        self.__env = env

    def reset(self, _):

        super(TradeEnvStrategy, self).reset(_)

        self.current_Trade = None
        self.use_event_datetime_logs = True

        self.rewards = 0
        self.last_in_price = None

        self.exited_positions = []

        self.passed_bars = 0

    def get_observe(self, bars, final=False):

        bar = bars[self.__instrument]
        state = bar.dict
        del state["start_date"]
        del state["end_date"]
        del state["ask_open"]
        del state["bid_open"]
        state["quantity"] = self.broker.quantities[self.__instrument]
        state["equity"] = self.broker.equity
        state["last_in_price"] = self.last_in_price if self.last_in_price is not None else state["ask_close"]
        current_state = np.array([v for v in state.values()])
        return_value = (current_state, self.rewards, final)
        self.rewards = 0
        return return_value

    def onBars(self, dateTime, bars):

        bar = bars[self.__instrument]
        observe = self.get_observe(bars)

        assert observe is not None, "Observe Can't be None"

        type_, action = self.__env.action_from_state(('STATE', observe))

        assert type_ == 'ACTION', 'SEND STATTE SHOULD GET ACTION'

        self.ai_do(out_price=bar.out_price, action=action)

        if self.passed_bars % self.__log_per_bin == 0:
            self.logger.info('cash:%d equity:%d quant:%d' % (
                self.broker.cash(), self.broker.equity, self.broker.quantities[self.__instrument]))

        self.passed_bars += 1

    def ai_do(self, out_price, action):
        should_sell = action[0]
        should_buy = action[1]

        if should_buy and self.current_Trade is None:
            use_cash = self.broker.cash() * 0.9
            quantity = use_cash / out_price
            self.enterLong(self.__instrument, int(use_cash / quantity))

        if should_sell and (self.current_Trade is not None) and self.current_Trade.getEntryOrder().is_filled:
            self.current_Trade.exitMarket()
            self.current_Trade = None

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

        self.exited_positions.append(position)

        if len(self.exited_positions) % self.__log_per_trade == 0 and len(self.exited_positions) > 0:
            amounts = 0
            profit = 0
            win = 0
            for position in self.exited_positions[-self.__log_per_trade:]:
                position: strategy.position.Position = position
                entry: order.MarketOrder = position.getEntryOrder()
                exit: order.MarketOrder = position.getExitOrder()
                amounts += entry.quantity
                delta = exit.avg_fill_price - entry.avg_fill_price
                profit += delta * entry.quantity
                if delta > 0:
                    win += 1

            self.logger.info(
                'cash:%d equity:%d quant:%d, \t in last 100 trade: profit:%.3f amounts:%.3f avg:%.3f win:%d' % (
                    self.broker.cash(),
                    self.broker.equity,
                    self.broker.quantities[self.__instrument],
                    profit, amounts, profit / amounts * 100, win))

    def onExitCanceled(self, position):
        super().onExitCanceled(position)

    def onStart(self):
        super().onStart()

    def onFinish(self, bars):
        super().onFinish(bars)

        self.logger.info('finish trade!')
        amounts = 0
        profit = 0
        win = 0
        for position in self.exited_positions:
            position: strategy.position.Position = position
            entry: order.MarketOrder = position.getEntryOrder()
            exit: order.MarketOrder = position.getExitOrder()
            amounts += entry.quantity
            delta = exit.avg_fill_price - entry.avg_fill_price
            profit += delta * entry.quantity
            if delta > 0:
                win += 1

        self.logger.info(
            '[TOTAL]cash:%d equity:%d quant:%d, \t in all trades: profit:%.3f amounts:%.3f avg:%.3f win rate:%.3f' % (
                self.broker.cash(),
                self.broker.equity,
                self.broker.quantities[self.__instrument],
                profit, amounts, profit / amounts * 100 if amounts>0 else 0,
                100 * win / len(self.exited_positions) if len(self.exited_positions) > 0 else 0))

        self.__env.put_state(('STATE', self.get_observe(bars, True)))

    def onIdle(self):
        super().onIdle()

    def onOrderUpdated(self, order):
        super().onOrderUpdated(order)

        # self.profit_trade = 0
        # self.loss_trade = 0
