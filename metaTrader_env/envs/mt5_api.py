# Meta Trader API
# Collect information from meta trader software which is connected to your broker account
import math

# Get the Meta Trader module

import MetaTrader5 as mt5
import datetime
import time

import pandas as pd


class MetaTrader:
    def __init__(self):

        self._magic_num = 0
        self._balance = 0
        self._equity = 0
        self._profit = 0
        self._margin = 0
        self._free_margin = 0
        self.algo_mode_active = False
        self._initialized = False
        self._connected = False
        self._current_order = 0
        self._point = 0
        self._orders = []
        self._symbol_list = []
        self._freeze_level = 0
        self._stops_level = 0
        self._curr_symbol = None
        self._digit = 0 # symbol digit/ carry only one symbol at a time
        self._tick_size = None
        self._tick_value = None
        self._lot_step = None
        self._min_volume = None
        self._max_volume = None
        self._volume_limit = None

    def initialize(self, account_num: object, passwd: object, server: object, magic_num: int = 4447) -> bool:
        # get values of account and balance from metatrader
        # initial the connection to metatrader

        self._initialized = mt5.initialize()
        self._connected = mt5.login(login=account_num,passwd=passwd, server=server)  # returns true if successful else false
        if self._connected:
            account_info = mt5.account_info()
            print(">: Successfully connected to MetaTrader")
            if account_info is not None:
                data = mt5.account_info()._asdict()

                # search for useful info
                self._magic_num = magic_num
                self._balance = data["balance"]
                self._equity = data["equity"]
                self._profit = data["profit"]
                self._margin = data["margin"]
                self._free_margin = data["margin_free"]
                self.algo_mode_active = data["trade_expert"]

                #self.set_symbol_info(symbol)

                del data
                del account_info
                return True
            else:
                print("Login failed")
                return PermissionError
        else:
            print("Initialization failed")
            return False

    def get_symbols(self):
        # if empty initialize symbols first
        return self._symbol_list

    def get_balance(self):
        account_info = mt5.account_info()
        if account_info is not None:
            data = mt5.account_info()._asdict()
            return data["balance"]
        return None

    def get_equity(self):
        account_info = mt5.account_info()
        if account_info is not None:
            data = mt5.account_info()._asdict()
            return data["equity"]
        return None

    def set_symbol_info(self, symbol):
        data = mt5.symbol_info(symbol)._asdict()
        #print("Symbol data ",data)
        self._point = data["point"]
        self._digit = data["digits"]
        self._freeze_level = data["trade_freeze_level"]
        self._stops_level = data["trade_stops_level"]

        self._tick_size = data["trade_tick_size"]
        self._tick_value = data["trade_tick_value"]
        self._lot_step = data["volume_step"]
        self._min_volume = data["volume_min"]
        self._max_volume = data["volume_max"]
        self._volume_limit = data["volume_limit"]

        del data


    @staticmethod
    def get_bid(symbol: str):
        """Get current bid price for symbol"""
        return mt5.symbol_info(symbol)._asdict()["bid"]

    @staticmethod
    def get_ask(symbol: str):
        """Get current ask price for symbol"""
        return mt5.symbol_info(symbol)._asdict()["ask"]

    def get_digits(self):
        """Get current digits price for symbol"""
        return self._digit

    #@staticmethod

    def modify_order(self, ticket: int, symbol: str, new_sl: float, new_tp: float, retries:int=5):
        # set symbols characteristics if first time
        # create request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": str(symbol),
            "sl": float(new_sl),
            "tp": float(new_tp),
            "position": int(ticket),
        }

        result = mt5.order_send(request)
        if result.retcode == 10009:
            return True
        else:
            for _ in range(retries):
                result = mt5.order_send(request)
                if result.retcode == 10009:
                    return True
                else:
                    time.sleep(1)
            return False

    @staticmethod
    def cancel_order(order_num):
        # create_request
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order_num,
            "comment": "Order Remove"
        }
        result = mt5.order_send(request)
        return result

    @staticmethod
    def disconnect():
        return mt5.shutdown()

    # Function to convert a timeframe string in MetaTrader 5 friendly format
    @staticmethod
    def set_query_timeframe(timeframe):
        if timeframe == "M1":
            return mt5.TIMEFRAME_M1
        elif timeframe == "M2":
            return mt5.TIMEFRAME_M2
        elif timeframe == "M3":
            return mt5.TIMEFRAME_M3
        elif timeframe == "M4":
            return mt5.TIMEFRAME_M4
        elif timeframe == "M5":
            return mt5.TIMEFRAME_M5
        elif timeframe == "M6":
            return mt5.TIMEFRAME_M6
        elif timeframe == "M10":
            return mt5.TIMEFRAME_M10
        elif timeframe == "M12":
            return mt5.TIMEFRAME_M12
        elif timeframe == "M15":
            return mt5.TIMEFRAME_M15
        elif timeframe == "M20":
            return mt5.TIMEFRAME_M20
        elif timeframe == "M30":
            return mt5.TIMEFRAME_M30
        elif timeframe == "H1":
            return mt5.TIMEFRAME_H1
        elif timeframe == "H2":
            return mt5.TIMEFRAME_H2
        elif timeframe == "H3":
            return mt5.TIMEFRAME_H3
        elif timeframe == "H4":
            return mt5.TIMEFRAME_H4
        elif timeframe == "H6":
            return mt5.TIMEFRAME_H6
        elif timeframe == "H8":
            return mt5.TIMEFRAME_H8
        elif timeframe == "H12":
            return mt5.TIMEFRAME_H12
        elif timeframe == "D1":
            return mt5.TIMEFRAME_D1
        elif timeframe == "W1":
            return mt5.TIMEFRAME_W1
        elif timeframe == "MN1":
            return mt5.TIMEFRAME_MN1
        return None

    def cal_profit(self, direction:int, symbol:str, lot:float, entry_price:float, exit_price:float):
        profit = 0
        if direction == 1:
            profit = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, str(symbol), float(lot), float(entry_price), float(exit_price)) if mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, str(symbol), float(lot), float(entry_price), float(exit_price)) is not None else print(f"Profit calculation failed {mt5.last_error()}")
        elif direction == -1:
            profit = mt5.order_calc_profit(mt5.ORDER_TYPE_SELL, str(symbol), float(lot), float(entry_price), float(exit_price)) if mt5.order_calc_profit(mt5.ORDER_TYPE_SELL,str(symbol), float(lot), float(entry_price), float(exit_price)) is not None else print(f"Profit calculation failed {mt5.last_error()}")
        if profit is None:
            print(mt5.last_error())
        return profit


    @staticmethod
    def queryHistory(symbol:str, timeframe:str="M5", number_of_candles:int=100):
        # convert timeframe to mt5 ENUM
        timeframe = MetaTrader.set_query_timeframe(timeframe)
        # Retrieve data from mt5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, number_of_candles)
        return rates

    # get opened positions
    @staticmethod
    def get_open_orders():
        orders = mt5.orders_get()
        orders_list = []
        for order in orders:
            orders_list.append(order[0])
        return orders_list

    @staticmethod
    def get_open_positions():
        # Get position objects
        positions = mt5.positions_get()
        return positions

    @staticmethod
    def is_new_bar(timeframe:int = 5):
        previous_time = 0 # may work different because of different types
        curr_time = datetime.datetime.now()
        if previous_time != curr_time:
            previous_time = curr_time
            return True
        else:
            return False

    @staticmethod
    def get_bars(symbol:str = "XAUUSD+", timeframe:str = "M5", limit:int=10):
        time_ = MetaTrader.set_query_timeframe(timeframe)
        bars = mt5.copy_rates_from_pos(symbol, time_, 1, limit) # change start position from lastest (0) to Closed bar index (1)
        return bars

    # function to get symbols form mt5
    def initialize_symbols(self, symbol_list):
        """
        Get the full list of available symbols in meta trader and check if the list of symbols passed are available
        :param symbol_list: List of symbols to be used
        :type symbol_list: list
        :return:
        """
        # get all symbols from broker
        s_list = mt5.symbols_get()
        symbols_names = []
        # retrieve name from symbols
        for symbol in s_list:
            symbols_names.append(symbol.name)
        self._symbol_list = symbols_names
        # Check each symbol in symbol list to ensure it exist
        for s in symbol_list:
            if s in symbols_names:
                if mt5.symbol_select(s, True):
                    print(f"symbol {s} enabled")
                else:
                    return ValueError
            else:
                return SyntaxError
        # return true if all are presents
        return True

    @staticmethod
    def get_position_details():
        pos = mt5.positions_get()
        ps = pd.DataFrame([pos[n] for n in range(len(pos))], columns=["ticket", "time", "time_msc", "time_update", "time_update_msc", "type", "magic", "identifier", "reason", "volume", "price_open", "sl", "tp", "price_current", "swap", "profit", "symbol", "comment","external_id"])

        return ps[["ticket", "time","type","volume","price_open","symbol", "magic", "sl", "tp"]]


    def get_num_positions(self):
        num = MetaTrader.get_position_details()
        if num is not None:
            return len(num)
        else:
            return 0

    def send_order(self, req:dict, n=5):
        res = None
        for _ in range(n):
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                break
            else:
                print(mt5.last_error())
        return res



    def calculate_lots(self, sl_points, symbol:str, risk_percentage:float=5):
        # set symbol characteristics e.i., Digits, Point
        self.set_symbol_info(symbol)
        risk = (self.get_balance() * risk_percentage) / 100.

        money_per_lotStep = (sl_points / self._tick_size) * (self._tick_value * self._lot_step)
        lots = math.floor(risk / money_per_lotStep) * self._lot_step #math.floor(risk_percentage / money_per_lotStep) * self._lot_step

        if self._volume_limit:
            lots = min(lots, self._volume_limit) #lots if lots < self._volume_limit else self._volume_limit
        if self._max_volume != 0:

            lots = min(lots, self._max_volume)
        if self._min_volume != 0:

            lots = max(lots, self._min_volume) #lots if lots > self._min_volume else

        lots = round(lots, 2)
        return lots



    def close_latest_position(self, symbol:str):
        pos = MetaTrader.get_position_details().iloc[-1]
        #print(pos)
        if symbol == pos["symbol"]:
            close_order = dict(action  =mt5.TRADE_ACTION_DEAL,
                                type   = mt5.ORDER_TYPE_SELL if pos["type"] == 0 else mt5.ORDER_TYPE_BUY,
                                volume = float(pos["volume"]),
                                price  = float(MetaTrader.get_bid(symbol)) if pos["type"]==1 else float(MetaTrader.get_ask(symbol)),
                                symbol = str(pos["symbol"]),
                                magic  = int(pos["magic"]),
                                position = int(pos["ticket"]),
                               deviation = 50,
                               comment = "Agent closing position",
                               type_time = mt5.ORDER_TIME_GTC,
                               type_filling = mt5.ORDER_FILLING_IOC,
                               )
            result = self.send_order(close_order)
            print(result)
        else:
            print(f">: position {symbol} not available")

    def close_position_at(self, symbol:str, idx:int):
        try:
            pos = MetaTrader.get_position_details().iloc[idx]
        except IndexError:
            print(f">: Index out of bond, no position to close")
            return
        #print(pos)
        if symbol == pos["symbol"]:
            close_order = dict(action  =mt5.TRADE_ACTION_DEAL,
                                type   = mt5.ORDER_TYPE_SELL if pos["type"] == 0 else mt5.ORDER_TYPE_BUY,
                                volume = float(pos["volume"]),
                                price  = float(MetaTrader.get_bid(symbol)) if pos["type"]==1 else float(MetaTrader.get_ask(symbol)),
                                symbol = str(pos["symbol"]),
                                magic  = int(pos["magic"]),
                                position = int(pos["ticket"]),
                               deviation = 50,
                               comment = "Agent closing position",
                               type_time = mt5.ORDER_TIME_GTC,
                               type_filling = mt5.ORDER_FILLING_IOC,
                               )
            result = self.send_order(close_order)
            print(result)
        else:
            print(f"position {symbol} not available")


    def close_all_positions(self, symbol:str):
        pos = MetaTrader.get_position_details()

        for i in range(len(pos)):
            if symbol == pos.iloc[i]["symbol"]:
                close_order = dict(action=mt5.TRADE_ACTION_DEAL,
                                   type=mt5.ORDER_TYPE_SELL if pos.iloc[i]["type"] == 0 else mt5.ORDER_TYPE_BUY,
                                   volume=float(pos.iloc[i]["volume"]),
                                   price=float(MetaTrader.get_bid(symbol)) if pos.iloc[i]["type"] == 1 else float(
                                       MetaTrader.get_ask(symbol)),
                                   symbol=str(pos.iloc[i]["symbol"]),
                                   magic=int(pos.iloc[i]["magic"]),
                                   position=int(pos.iloc[i]["ticket"]),
                                   deviation=50,
                                   comment="Agent closing position",
                                   type_time=mt5.ORDER_TIME_GTC,
                                   type_filling=mt5.ORDER_FILLING_IOC,
                                   )
                result = self.send_order(close_order)
                print(result)

    def test_profit(self):
        ask = self.get_ask("XAUUSD+")
        exit_price = ask - 500 * 0.1
        symbol = "XAUUSD+"
        lot = 0.01
        result = mt5.order_calc_profit(mt5.ORDER_TYPE_SELL, symbol, lot, ask, exit_price)
        if result is None:
            print(mt5.last_error())
        else:
            print(result)

    def trailing_stop(self, symbol:str, trigger_points:int, step_points:int):
        """
        Trail each position to get the maximum profit.
        :return:
        """
        sl = 0.
        tp = 0.
        ask = self.get_ask(symbol)
        bid = self.get_bid(symbol)

        if not self._point:
            self.set_symbol_info(symbol)

        positions = pd.DataFrame(MetaTrader.get_position_details())

        if positions:
            for i in range(len(positions)):
                pos = positions.iloc[i]
                ticket = pos["ticket"]
                if  pos["symbol"] == symbol: #pos["magic"] == self._magic_num and
                    if pos["type"] == mt5.ORDER_TYPE_BUY:
                        # simply
                        if (bid - pos["open_price"]) > trigger_points * self._point:
                            tp = pos["tp"]
                            sl = round(bid - (step_points * self._point), self._digit)
                            if sl > pos["sl"] and sl != 0:
                                try:
                                    if self.modify_order(ticket, symbol, sl, tp):
                                        self.modify_order(ticket, symbol, sl, tp)
                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")
                    else:
                        if (ask + trigger_points * self._point) < pos["open_price"]:
                            tp = pos["tp"]
                            sl = round(ask + step_points * self._point, self._digit)
                            if sl < pos["sl"] and sl != 0:
                                try:
                                    self.modify_order(ticket, symbol, sl, tp)
                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")


    # second version with the use of the fraction of the take profit
    def trailing_stop_fraction(self, symbol:str, trigger_fraction:float, trigger_fraction_stop:float, step_points:int):
        """
        Trail each position to get the maximum profit. Once we've entered certain fraction
        :return:
        """

        if self.get_digits() is None:
            self.set_symbol_info(symbol)

        sl = 0.
        tp = 0.
        ask = self.get_ask(symbol)
        bid = self.get_bid(symbol)

        if not self._point:
            self.set_symbol_info(symbol)

        positions = pd.DataFrame(MetaTrader.get_position_details())
        #print(positions)
        if not positions.empty:
            #print(positions)
            for i in range(0,len(positions)):
                #print(f">: Inner. Position: {positions.iloc[i]}")
                pos = positions.iloc[i]
                ticket = pos["ticket"]
                ret = False
                if  pos["magic"] == self._magic_num and pos["symbol"] == symbol: #
                    if pos["type"] == mt5.ORDER_TYPE_BUY:
                        tp = pos["tp"]
                        trigger_points = (trigger_fraction * tp) / 100
                        if (bid - pos["price_open"]) >= trigger_points * self._point:
                            sl = round(bid - (step_points * self._point), self._digit)  if (step_points * self._point) > self._stops_level else round(bid - (self._stops_level * self._point), self._digit)
                            if sl > pos["sl"] and sl != 0:
                                try:
                                    ret = self.modify_order(ticket, symbol, sl, tp)
                                    #if self.modify_order(ticket, symbol, sl, tp):
                                    #else:
                                    #    sl = round(bid - ((self._stops_level+50) * self._point), self._digit)
                                    #    ret = self.modify_order(ticket, symbol, sl, tp)

                                    if ret:
                                        print(">: Stop loss modification successfully changed.")
                                    else:
                                        self.close_position_at(symbol=symbol, idx=i)
                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")
                        else:
                            trigger_points = (trigger_fraction_stop * pos["sl"]) / 100
                            if (pos["price_open"] - bid) <= trigger_points * self._point:
                                self.close_position_at(symbol=symbol, idx=i)
                                print(f">: Forced closure {ticket}.")

                    else:
                        tp = pos["tp"]
                        trigger_points = (trigger_fraction * tp) / 100
                        if (ask + trigger_points * self._point) <= pos["price_open"]:
                            sl = round(ask + (step_points * self._point), self._digit) if (step_points * self._point) >= self._stops_level else round(ask + (self._stops_level * self._point), self._digit)
                            if sl < pos["sl"] and sl != 0:
                                try:
                                    ret = self.modify_order(ticket, symbol, sl, tp)
                                    #if self.modify_order(ticket, symbol, sl, tp):
                                    #else:
                                    #    sl = round(ask + ((self._stops_level+50) * self._point), self._digit)
                                    #    ret=self.modify_order(ticket, symbol, sl, tp)

                                    if ret:
                                        print(">: Stop loss modification successfully changed.")
                                    else:
                                        self.close_position_at(symbol=symbol, idx=i)

                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")

                        else:
                            trigger_points = (trigger_fraction_stop * pos["sl"]) / 100
                            if (ask + trigger_points * self._point) >= pos["price_open"]:
                                self.close_position_at(symbol=symbol, idx=i)
                                print(f">: Forced closure of position {ticket}.")

    # second version with the use of the fraction of the take profit
    def trailing_stop_fraction_2(self, symbol: str, trigger_fraction: float, step_points: int):
        """
        Trail each position to get the maximum profit. Once we've entered certain fraction
        :return:
        """

        if self.get_digits() is None:
            self.set_symbol_info(symbol)

        ask = self.get_ask(symbol)
        bid = self.get_bid(symbol)

        if not self._point:
            self.set_symbol_info(symbol)

        positions = pd.DataFrame(MetaTrader.get_position_details())
        if not positions.empty:
            for i in range(0, len(positions)):
                pos = positions.iloc[i]
                ticket = pos["ticket"]
                ret = False
                if pos["magic"] == self._magic_num and pos["symbol"] == symbol:
                    if pos["type"] == mt5.ORDER_TYPE_BUY:
                        # Distance to Take Profit in points
                        tp_dist_points = (pos["tp"] - pos["price_open"]) / self._point if pos["tp"] > 0 else 0
                        trigger_tp_points = (trigger_fraction * tp_dist_points) / 100.0

                        # Positive pip movement
                        target_reached = trigger_tp_points > 0 and (
                                    bid - pos["price_open"]) >= trigger_tp_points * self._point

                        if target_reached:
                            # Trail SL favorably!
                            dist_points = step_points if step_points >= self._stops_level else self._stops_level
                            sl_new = round(bid - (dist_points * self._point), self._digit)

                            if sl_new > pos["sl"] and sl_new != 0:
                                try:
                                    ret = self.modify_order(ticket, symbol, sl_new, pos["tp"])
                                    if ret:
                                        print(">: Stop loss modification successfully changed.")
                                    else:
                                        self.close_position_at(symbol=symbol, idx=i)
                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")
                        #else:
                            # Kill switch logic if trade is losing
                            #sl_dist_points = (pos["price_open"] - pos["sl"]) / self._point if pos["sl"] > 0 else 0
                            #trigger_stop_points = (trigger_fraction_stop * sl_dist_points) / 100.0

                            # Negative pip movement
                            #loss_reached = trigger_stop_points > 0 and (
                            #            pos["price_open"] - bid) >= trigger_stop_points * self._point

                            #if loss_reached:
                            #    self.close_position_at(symbol=symbol, idx=i)
                            #    print(f">: Forced closure BUY {ticket} by distance kill switch.")

                    else:  # SELL ORDER
                        # Distance to Take Profit in points
                        tp_dist_points = (pos["price_open"] - pos["tp"]) / self._point if pos["tp"] > 0 else 0
                        trigger_tp_points = (trigger_fraction * tp_dist_points) / 100.0

                        # Positive pip movement
                        target_reached = trigger_tp_points > 0 and (
                                    pos["price_open"] - ask) >= trigger_tp_points * self._point

                        if target_reached:
                            dist_points = step_points if step_points >= self._stops_level else self._stops_level
                            sl_new = round(ask + (dist_points * self._point), self._digit)

                            if sl_new < pos["sl"] and sl_new != 0:
                                try:
                                    ret = self.modify_order(ticket, symbol, sl_new, pos["tp"])
                                    if ret:
                                        print(">: Stop loss modification successfully changed.")
                                    else:
                                        self.close_position_at(symbol=symbol, idx=i)
                                except Exception as e:
                                    print(f"> Unable to modify order {ticket}: {e}")
                        #else:
                            # Kill switch logic if trade is losing
                            #sl_dist_points = (pos["sl"] - pos["price_open"]) / self._point if pos["sl"] > 0 else 0
                            #trigger_stop_points = (trigger_fraction_stop * sl_dist_points) / 100.0

                            # Negative pip movement
                            #loss_reached = trigger_stop_points > 0 and (
                            #            ask - pos["price_open"]) >= trigger_stop_points * self._point

                            #if loss_reached:
                            #    self.close_position_at(symbol=symbol, idx=i)
                            #    print(f">: Forced closure SELL {ticket} by distance kill switch.")

    # kill switch :- We are loosing more than wining, kill all
    def kill_switch(self, symbol:str, trigger_fraction:float, called=False):
        """
        Kill the position if we loose more than X% of the original stop loss (ideally there's no use but having multiple positions puts us at a greater risk).
        To be called once, once the position goes south kill it, it would be difficult to adjust the stop loss when price is moving
        """

        if called:
            if self.get_digits() is None:
                self.set_symbol_info(symbol)

            ask = self.get_ask(symbol)
            bid = self.get_bid(symbol)

            positions = pd.DataFrame(MetaTrader.get_position_details())
            if not positions.empty:
                for i in range(0, len(positions)):
                    pos = positions.iloc[i]
                    ticket = pos["ticket"]
                    ret = False
                    if pos["magic"] == self._magic_num and pos["symbol"] == symbol:
                        if pos["type"] == mt5.ORDER_TYPE_BUY:
                            # Distance to Take Profit in points
                            # Kill switch logic if trade is losing
                            sl_dist_points = (pos["price_open"] - pos["sl"]) / self._point if pos["sl"] > 0 else 0
                            trigger_stop_points = (trigger_fraction * sl_dist_points) / 100.0

                            # Negative pip movement
                            loss_reached = trigger_stop_points > 0 and (
                                    pos["price_open"] - bid) >= trigger_stop_points * self._point

                            if loss_reached:
                                self.close_position_at(symbol=symbol, idx=i)
                                print(f">: Forced closure BUY {ticket} by distance kill switch.")

                        else:  # SELL ORDER
                            # Distance to Take Profit in points
                            # Kill switch logic if trade is losing
                            sl_dist_points = (pos["sl"] - pos["price_open"]) / self._point if pos["sl"] > 0 else 0
                            trigger_stop_points = (trigger_fraction * sl_dist_points) / 100.0

                            # Negative pip movement
                            loss_reached = trigger_stop_points > 0 and (
                                    ask - pos["price_open"]) >= trigger_stop_points * self._point

                            if loss_reached:
                                self.close_position_at(symbol=symbol, idx=i)
                                print(f">: Forced closure SELL {ticket} by distance kill switch.")
                # Doesn't check if a position was triggered (point of check here), just return true if called
                return True
        return False




    #--- TODO: <investigate the way of filling orders>
    #--- News it works great, no touching
    def place_order(self, order_type: str, lot_size: float, price: float,
                          symbol: str,     sl_point: int = 100, order_dist_point:int=50,
                          tp_point: int = 100, deviation: float = 20):
        """
        Place an order using the meta trader api, differentiate between BUY and SELL.
        Create a request dictionary then send it through the api call.

         :param order_type: Order type, either BUY or SELL
         :type order_type: str
         :param lot_size: Order lot size
         :type lot_size: float
         :param price: Current price to execute order
         :type price: float
         :param symbol: Current symbol to execute order
         :type symbol: str
         :param sl_point: Stop loss in point, to be combined to entry/Price to have an exit price
         :type sl_point: float
         :param order_dist_point: Distance from the price or can also be used as distance from order
         :type order_dist_point: float
         :param tp_point: Take profit in point, to be combined to entry/Price to have an exit price
         :type tp_point: float
         :param deviation: Acceptable deviation of price
         :type deviation: float
         :return: Result code from meta trader, e.i., 1009 for successful
        """
        # set symbol characteristics e.i., Digits, Point
        self.set_symbol_info(symbol)

        if order_type.lower() == "sell":
            if sl_point >= self._stops_level+5:
                sl = price + (sl_point * self._point)
                tp = price - (tp_point * self._point)
            else:
                sl = price + (self._stops_level+5 * self._point)
                tp = price - (self._stops_level+5 * self._point)
            order_type = mt5.ORDER_TYPE_SELL
            # Create request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": round(float(price + (order_dist_point * self._point)), self._digit),
                "sl": round(float(sl), self._digit),
                "tp": round(float(tp), self._digit),
                "deviation": deviation,
                "magic": int(self._magic_num),
                "comment": "RL Trader Open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        else:
            if sl_point >= self._stops_level+5:
                sl = price - (sl_point * self._point)
                tp = price + (tp_point * self._point)
            else:
                sl = price - (self._stops_level+5 * self._point)  #if self._stops_level <=  else price
                tp = price + (self._stops_level+5 * self._point)
            order_type = mt5.ORDER_TYPE_BUY
            # Create request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": round(float((price + order_dist_point * self._point)), self._digit),
                "sl": round(float(sl), self._digit),
                "tp": round(float(tp), self._digit),
                "deviation": deviation,
                "magic": int(self._magic_num),
                "comment": "RL Trader Open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        # send request to MT5
        result = mt5.order_send(request)
        # modify based on return outcomes
        #print(mt5.last_error())
        if result.retcode == 10009:
            print(f"Order for {symbol} successful")
        else:
            print(f"Error placing order. ErrorCode {result[0]}, Error Details: {result}")
        return result


    # class to get new bar
    class NewBarDetector:
        def __init__(self, symbol_, timeframe):
            self.symbol = symbol_
            self.timeframe = timeframe
            self.previous_time = None

        def is_new_bar(self):
            # Fetch the current bar
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 10)
            if rates is None:
                return False

            current_time = rates[-1]['time']
            #print(">: Current time is ", current_time)

            # Handle first call
            if self.previous_time is None:
                self.previous_time = current_time
                return False

            # Detect change
            if current_time != self.previous_time:
                self.previous_time = current_time
                return True

            return False



