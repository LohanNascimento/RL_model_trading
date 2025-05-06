import pandas as pd
import numpy as np
from envs.smc_trading_env import SMCTradingEnv

# Carrega dados reais BTC/USDT 1h
csv_path = 'BTCUSDT_1h.csv'
df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

# Calcula RSI (14 períodos)
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calc_rsi(df['close'])
df = df.dropna().reset_index(drop=True)

# Inicializa o ambiente com dados reais
env = SMCTradingEnv(df, window_size=10)
obs, info = env.reset()
print('Observação inicial:', obs.shape)

done = False
total_reward = 0
steps = 0

while not done:
    action = env.action_space.sample()  # Ação aleatória
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if steps % 200 == 0:
        env.render()

print(f'\nEpisódio finalizado em {steps} passos. Recompensa total: {total_reward:.2f}')
print(f'Lucro final: {info["episode_profit"]:.2f}, Drawdown máximo: {info["max_drawdown"]:.2f}')
