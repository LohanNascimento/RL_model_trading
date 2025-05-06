import pandas as pd
import numpy as np
import logging
from utils.config_loader import load_config
from utils.technical_indicators import add_technical_indicators
from sklearn.preprocessing import StandardScaler

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Carregar configuração
env_cfg = load_config('config/env_config.yaml')

# Função para limpar valores negativos
def fix_negative_values(df, stage_name):
    negative_cols = []
    for col in df.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_cols.append((col, neg_count))
            # Tratar valores negativos para evitar problemas
            if col in ['close', 'open', 'high', 'low', 'volume']:
                # Para preços e volume, substituir negativos por um valor pequeno positivo
                df.loc[df[col] < 0, col] = 0.0001
            elif col in ['macd', 'macd_signal']:
                # MACD pode ser negativo naturalmente, não precisa tratar
                pass
            else:
                # Para outros indicadores técnicos, substituir por valor absoluto
                df.loc[df[col] < 0, col] = df[col].abs()
    
    if negative_cols:
        logging.warning(f"[{stage_name}] Encontrados valores negativos (já corrigidos):")
        for col, count in negative_cols:
            logging.warning(f"  - Coluna {col}: {count} valores negativos")
    return df

# Carregar dados dos símbolos
dfs = []
for sym in env_cfg['environment']['symbols']:
    try:
        logging.info(f"Carregando dados para {sym}...")
        df = pd.read_csv(f'data/{sym}_1h.csv', parse_dates=['timestamp'])
        df['symbol'] = sym
        
        # Corrigir valores negativos nos dados brutos
        df = fix_negative_values(df, f"Dados brutos - {sym}")
        
        dfs.append(df)
        logging.info(f"Dados para {sym} carregados com sucesso: {len(df)} linhas.")
    except Exception as e:
        logging.error(f"Erro ao carregar dados para {sym}: {str(e)}")

# Concatenar todos os dataframes
df_all = pd.concat(dfs, ignore_index=True)
logging.info(f"Dataset combinado: {len(df_all)} linhas.")

# Dividir em train/val/test
train_size = 0.7
val_size = 0.2
test_size = 0.1
total_len = len(df_all)
train_end = int(total_len * train_size)
val_end = train_end + int(total_len * val_size)
train_df = df_all.iloc[:train_end].reset_index(drop=True)
val_df = df_all.iloc[train_end:val_end].reset_index(drop=True)
test_df = df_all.iloc[val_end:].reset_index(drop=True)

# Adicionar indicadores técnicos
logging.info("Adicionando indicadores técnicos...")
train_df = add_technical_indicators(train_df)
val_df = add_technical_indicators(val_df)
test_df = add_technical_indicators(test_df)

# Corrigir valores negativos após adicionar indicadores
train_df = fix_negative_values(train_df, "Train após indicadores")
val_df = fix_negative_values(val_df, "Val após indicadores")
test_df = fix_negative_values(test_df, "Test após indicadores")

# Remover NaNs
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Verificar antes da normalização
for col in ['close', 'open', 'high', 'low']:
    neg_count = (train_df[col] < 0).sum()
    if neg_count > 0:
        logging.error(f"ANTES da normalização - Coluna {col}: {neg_count} valores negativos")

# Normalizar os dados
feature_cols = env_cfg['observation']['features']
logging.info(f"Normalizando dados com StandardScaler para colunas: {feature_cols}")

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

# Transform nos dados
train_df_norm = train_df.copy()
val_df_norm = val_df.copy()
test_df_norm = test_df.copy()

train_df_norm[feature_cols] = scaler.transform(train_df[feature_cols])
val_df_norm[feature_cols] = scaler.transform(val_df[feature_cols])
test_df_norm[feature_cols] = scaler.transform(test_df[feature_cols])

# Verificar valores negativos após normalização
logging.info("Verificando valores negativos após normalização...")
for col in train_df_norm.columns:
    if col in ['timestamp', 'symbol']:
        continue
    neg_count = (train_df_norm[col] < 0).sum()
    if neg_count > 0:
        neg_percentage = (neg_count / len(train_df_norm)) * 100
        if col in ['close', 'open', 'high', 'low', 'volume']:
            logging.error(f"IMPORTANTE - Coluna {col} após normalização: {neg_count} valores negativos ({neg_percentage:.2f}%)")
        else:
            logging.warning(f"Coluna {col} após normalização: {neg_count} valores negativos ({neg_percentage:.2f}%)")

# Verificar primeiras linhas após normalização para colunas de preço
logging.info("Primeiras 5 linhas de 'close' após normalização:")
logging.info(train_df_norm['close'].head(5).tolist())

# Solução: não normalizar as colunas de preço, apenas os indicadores
logging.info("\n--- Tentando solução alternativa: não normalizar close, open, high, low ---")

# Lista de features sem as colunas de preço e volume
price_cols = ['close', 'open', 'high', 'low', 'volume']
indicator_cols = [col for col in feature_cols if col not in price_cols]

if indicator_cols:
    logging.info(f"Normalizando apenas indicadores: {indicator_cols}")
    
    # Reiniciar os DataFrames
    train_df_fixed = train_df.copy()
    val_df_fixed = val_df.copy()
    test_df_fixed = test_df.copy()
    
    # Normalizar apenas os indicadores
    scaler_indicators = StandardScaler()
    scaler_indicators.fit(train_df[indicator_cols])
    
    train_df_fixed[indicator_cols] = scaler_indicators.transform(train_df[indicator_cols])
    val_df_fixed[indicator_cols] = scaler_indicators.transform(val_df[indicator_cols])
    test_df_fixed[indicator_cols] = scaler_indicators.transform(test_df[indicator_cols])
    
    # Verificar valores negativos após normalização alternativa
    for col in price_cols:
        neg_count = (train_df_fixed[col] < 0).sum()
        if neg_count > 0:
            logging.error(f"Ainda temos problema - Coluna {col}: {neg_count} valores negativos")
        else:
            logging.info(f"Coluna {col} OK - Sem valores negativos")
    
    # Salvar resultado para debug
    train_df_fixed.to_csv('train_df_fixed.csv', index=False)
    logging.info(f"Arquivo train_df_fixed.csv salvo para análise com {len(train_df_fixed)} linhas")
else:
    logging.warning("Todas as features são de preço ou volume, não há indicadores para normalizar separadamente.") 