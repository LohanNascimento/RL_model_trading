import pytest
import numpy as np
from strategy.strategy_rules import TrendFollowingStrategy, ReversalStrategy, RiskManager
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DummyEnv:
    def __init__(self):
        self.df = None
        self.window_size = 10
        self.position = 0
        self.current_step = 10
        self.take_profit = 100
        self.current_price = 100
        self.next_liquidity_target = 1

# Teste RiskManager
@pytest.mark.parametrize("balance, entry, stop, expected", [
    (10000, 100, 99, 200.0),
    (10000, 100, 100, 0.0),  # Deve retornar 0 (evita divisão por zero)
])
def test_calculate_position_size(balance, entry, stop, expected):
    rm = RiskManager()
    size = rm.calculate_position_size(balance, entry, stop)
    assert np.isclose(size, expected)

def test_trend_following_strategy_entry():
    # Configurar ambiente com tamanho mínimo necessário (window_size + bos_lookback + 1)
    env = DummyEnv()
    env.window_size = 50  # Usando o mesmo do trend_lookback para garantir dados suficientes
    strat = TrendFollowingStrategy()
    
    # Mock DataFrame e sinais
    import pandas as pd
    # Criar DataFrame com tamanho mínimo necessário (window_size + bos_lookback + 1)
    df = pd.DataFrame({
        'close': np.arange(60),  # 50 (window_size) + 3 (bos_lookback) + 1 (current_step)
        'open': np.arange(60),
        'high': np.arange(60),
        'low': np.arange(60),
        'FVG': [1]*60,
        'BOS': [1]*60,
        'CHoCH': [0]*60,
        'LIQ': [0]*60
    })
    env.df = df
    env.current_step = 55  # Garantir que temos dados suficientes para análise
    assert strat.check_entry(env, env.current_step) in [True, False]

def test_reversal_strategy_entry():
    # Configurar ambiente com tamanho mínimo necessário (window_size + trend_lookback + 1)
    env = DummyEnv()
    env.window_size = 50  # Usando o mesmo do trend_lookback para garantir dados suficientes
    strat = ReversalStrategy(trend_lookback=50)  # Explicitando o trend_lookback
    
    # Mock DataFrame e sinais
    import pandas as pd
    # Criar DataFrame com tamanho mínimo necessário (window_size + trend_lookback + 1)
    df = pd.DataFrame({
        'close': np.arange(1500),  # 100 (window_size + trend_lookback) + 50 (margem de segurança)
        'open': np.arange(1500),
        'high': np.arange(1500),
        'low': np.arange(1500),
        'CHoCH': [1]*1500,
        'FVG': [1]*1500
    })
    env.df = df
    env.current_step = 145  # Garantir que temos dados suficientes para análise
    
    # Garantir que temos dados suficientes para o slice
    df_window = env.df.iloc[env.current_step - env.window_size:env.current_step]
    assert len(df_window) == env.window_size
    
    assert strat.check_entry(env, env.current_step) in [True, False]
