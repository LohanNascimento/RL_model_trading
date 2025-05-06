import pandas as pd
from smartmoneyconcepts.smc import smc

# Substitua o caminho pelo seu arquivo real de dados
df = pd.read_csv('data/BTC_USDT_1h.csv')  # ou outro arquivo que você usa no projeto

print(smc.fvg(df.tail(15)))
print(smc.bos_choch(df.tail(15), smc.swing_highs_lows(df.tail(15))))
print(smc.liquidity(df.tail(15), smc.swing_highs_lows(df.tail(15))))