# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from freqtrade.persistence import Trade
from freqtrade.optimize.hyperopt import IHyperOptLoss
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, List
from talib import MA_Type
logger = logging.getLogger(__name__)


class CryptalStrategyOptimTrendRoiSharpe(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.139,
        "40": 0.051,
        "53": 0.025,
        "157": 0
    }

    # Stoploss:
    stoploss = -0.324

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.034
    trailing_only_offset_is_reached = True

    ### Hyperoptable parameters
    # SPACE BUY
    buy_adx = IntParameter(18, 32, default=30, optimize=True, space="buy")
    buy_macd = IntParameter(-10, 20, default=15, optimize=False, space="buy")

    # ENABLER BUY
    #buy_trigger_macd_increase_enabled = CategoricalParameter([True, False], default=False, optimize=True, space="buy")
    #buy_trigger_macd_enabled = CategoricalParameter([True, False], default=False, optimize=True, space="buy")
    #buy_trigger_cross = CategoricalParameter(["macd", "no_cross_cond"], default="no_cross_cond", optimize=True, space="buy")

    # SPACE SELL
    sell_adx = IntParameter(18, 32, default=27, optimize=False, space="sell")
    sell_stochrsi = IntParameter(60, 87, default=61, optimize=False, space="sell")
    #sell_macd = IntParameter(0, 30, default=7, optimize=True, space="sell")

    # ENABLER SELL
    # REPLACE BOLLINGER by KELTNER
    #sell_trigger_macd_decrease_enabled = CategoricalParameter([True, False], default=True, optimize=True, space="sell")

    buy_ema_short_period = IntParameter(3, 9, default=8, optimize=False, space="buy")
    #sell_ema_short_period = IntParameter(3, 9, default=5, optimize=True, space="sell")

    buy_ema_long_period = IntParameter(12, 30, default=29, optimize=False, space="buy")
    #sell_ema_long_period = IntParameter(9, 30, default=20, optimize=True, space="sell")

    # Strategy for 5m timeframe.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    #ignore_roi_if_buy_signal = True # Overridden

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 500

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False, # Gérer le stoploss directement dans freqtrade pour éviter la latence imposée par Binance (d'autant plus si je n'utilise plus les bougies par la suite)
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.995
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # MACD
        macd  = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['macd_previous_s'] = dataframe['macd'].shift(1)

        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        dataframe['fastd_rsi'] = stoch_rsi['fastd']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        upperband, middleband, lowerband = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=MA_Type.T3)
        dataframe['bb_up'] = upperband
        dataframe['bb_lw'] = lowerband
        dataframe['bb_mid'] = middleband


        dataframe["ema_5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)

        k = 30
        dataframe["ema_5_history"] = [
            dataframe['ema_5'].iloc[max(0, i - k + 1):i + 1].tolist() 
            for i in range(len(dataframe))
        ]
        dataframe["ema_20_history"] = [
            dataframe['ema_20'].iloc[max(0, i - k + 1):i + 1].tolist() 
            for i in range(len(dataframe))
        ]

        dataframe["ema_10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # Create a new column with arrays of current + 29 previous EMA-50 values
        dataframe["ema_50_history"] = [
            dataframe['ema_50'].iloc[max(0, i - k + 1):i + 1].tolist() 
            for i in range(len(dataframe))
        ]

        dataframe["ema_10_history"] = [
            dataframe['ema_10'].iloc[max(0, i - k + 1):i + 1].tolist() 
            for i in range(len(dataframe))
        ]

        dataframe["chaikin_ad_osc"] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)

        fastk, fastd = ta.STOCHF(
            dataframe["high"], dataframe["low"], dataframe["close"], fastk_period=14, fastd_period=3, fastd_matype=0
        )
        dataframe["fastk"] = fastk
        dataframe["fastd"] = fastd

        #if (metadata['pair'] == 'BTC/USDT'):
        #    print(dataframe)
        #    print(dataframe.columns)
        #    dataframe['json'] = dataframe.apply(lambda x: str(x.to_json()), axis=1)
        #    print(dataframe.tail(1)['json'].values[0])
            #print(stoch_rsi)
            #print(dataframe['fastk_rsi'])
            #print(dataframe['ema_var'])
            #print(dataframe.columns)
            #print(metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        #conditions = []

        # Some "BUY" conditions
        #if self.buy_trigger_macd_enabled.value  and (not self.buy_trigger_cross.value == "macd") and (not self.buy_trigger_macd_increase_enabled.value) :
        #    conditions.append(dataframe['macd'] > self.buy_macd.value /100)
        #if self.buy_trigger_macd_increase_enabled.value and (not self.buy_trigger_macd_enabled.value) and (not self.buy_trigger_cross.value == "macd"): #MACD increase et redressement du signal
        #if self.buy_trigger_macd_enabled.value:
        #conditions.append(dataframe['macd'] > dataframe['macd_previous_s'])
        #conditions.append(dataframe['macd_previous_s'] >= dataframe['macd_previous_l'])
        #if self.buy_trigger_cross.value == "macd":
        #    conditions.append(qtpylib.crossed_above(dataframe['macd'], self.buy_macd.value/100))

        #### GUARDS CONDITIONS
        #conditions.append((dataframe['volume'] > 0)) #& (dataframe['adx'] > self.buy_adx.value))

        #if conditions:
        #    dataframe.loc[(
        #        reduce(lambda x, y: x & y, conditions)),
        #        'buy'] = 1

        dataframe.loc[(
                (
                    (qtpylib.crossed_above(dataframe['macd'], self.buy_macd.value/100))
                    | (
                        (dataframe[f'ema_10'] >= dataframe[f'ema_5'])
                        & ((dataframe['macd'] > dataframe['macd_previous_s']) | ((dataframe['ema_5'] < dataframe['ema_5'])))
                    )
                )
                & dataframe['volume'] > 0 & (dataframe['adx'] > self.buy_adx.value)
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """

        #conditions = []

        ### Some "SELL" conditions

        #if self.sell_trigger_macd_decrease_enabled.value:
        #if dataframe['ema5'] < dataframe['ema20']:
        #    conditions.append(dataframe['macd'] < dataframe['macd_previous_s'])
        #else:
        #   # If increase period, allow more resilience if check two frames than one
        #    conditions.append(dataframe['macd'] < dataframe['macd_previous_s'])
        #    conditions.append(dataframe['macd_previous_s'] <= dataframe['macd_previous_l'])
        #else:
        #    conditions.append(qtpylib.crossed_below(dataframe['macd'], self.sell_macd.value/100))
        
        ### GUARDS CONDITIONS
        #conditions.append(dataframe['volume'] > 0)# & (dataframe['adx'] > self.sell_adx.value))

        #if conditions:
        #    dataframe.loc[(
        #        reduce(lambda x, y: x & y, conditions)),
        #        'sell'] = 1
        dataframe.loc[((
                qtpylib.crossed_below(dataframe['fastk_rsi'], self.sell_stochrsi.value)
                ) & dataframe['volume'] > 0 & (dataframe['adx'] > self.sell_adx.value)
            ),
            ['exit_long', 'exit_tag']] = (1, 'some_exit_tag')

        return dataframe


    #use_custom_stoploss = True
#
#
    #def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
    #                    current_rate: float, current_profit: float, **kwargs) -> float:
#
    #    # Pour chaque pair, optimisation du trailing stoploss/stoploss
    #    if (current_profit > 0.0) & (current_profit < self.stpl_limit_sup.value /100) :
    #        return self.stpl_val_inter.value /100
    #    elif current_profit >= self.stpl_limit_sup.value /100:
    #        return self.stpl_val_sup.value /1000
    #    elif current_profit <= 0.0:
    #        return self.stpl_val_inf.value /100
    #    return 3 /100
