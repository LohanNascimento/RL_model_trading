# strategy_rules.py
from abc import ABC, abstractmethod
import numpy as np


class BaseStrategy(ABC):
    @abstractmethod
    def check_entry(self, env, current_step) -> bool:
        pass
    
    @abstractmethod
    def check_exit(self, env, current_step) -> bool:
        pass

class ReversalStrategy(BaseStrategy):
    def __init__(self, trend_lookback=50):
        self.trend_lookback = trend_lookback
    
    def _get_market_structure(self, df):
        # Análise de tendência baseada em médias móveis
        sma = df['sma_14']
        ema = df['ema_14']
        uptrend = ema.iloc[-1] > sma.iloc[-1]
        downtrend = ema.iloc[-1] < sma.iloc[-1]
        return {'uptrend': uptrend, 'downtrend': downtrend}
    
    def check_entry(self, env, current_step):
        df_window = env.df.iloc[current_step - env.window_size:current_step]
        structure = self._get_market_structure(df_window)
        
        # Condições para reversão
        return (
            env.df.iloc[current_step]['CHoCH'] == 1 and
            env.df.iloc[current_step]['FVG'] != 0 and
            (structure['uptrend'] and env.df.iloc[current_step]['close'] < df_window['low'].min()) or
            (structure['downtrend'] and env.df.iloc[current_step]['close'] > df_window['high'].max())
        )
    
    def check_exit(self, env, current_step):
        # Lógica de saída baseada em liquidity
        return env.next_liquidity_target is not None

class TrendFollowingStrategy(BaseStrategy):
    """
    Estratégia de tendência baseada apenas em indicadores técnicos clássicos.
    Exemplo: entrada quando EMA cruza SMA para cima e RSI < 30;
    saída quando EMA cruza SMA para baixo ou RSI > 70.
    """
    def __init__(self, rsi_oversold=30, rsi_overbought=70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def check_entry(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Condição: EMA cruza SMA para cima e RSI < oversold
        ema = df['ema_14'].iloc[-1]
        sma = df['sma_14'].iloc[-1]
        ema_prev = df['ema_14'].iloc[-2]
        sma_prev = df['sma_14'].iloc[-2]
        rsi = df['rsi_14'].iloc[-1]
        cruzamento_alta = ema_prev < sma_prev and ema > sma
        return cruzamento_alta and rsi < self.rsi_oversold

    def check_exit(self, env, current_step):
        df = env.df.iloc[current_step - env.window_size:current_step]
        # Condição: EMA cruza SMA para baixo ou RSI > overbought
        ema = df['ema_14'].iloc[-1]
        sma = df['sma_14'].iloc[-1]
        ema_prev = df['ema_14'].iloc[-2]
        sma_prev = df['sma_14'].iloc[-2]
        rsi = df['rsi_14'].iloc[-1]
        cruzamento_baixa = ema_prev > sma_prev and ema < sma
        return cruzamento_baixa or rsi > self.rsi_overbought
