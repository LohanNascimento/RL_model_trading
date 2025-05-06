import pandas as pd
import logging
import numpy as np
from utils.config_loader import load_config
from utils.technical_indicators import add_technical_indicators

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Carregar configuração
env_cfg = load_config('config/env_config.yaml')

def inspect_negative_values(df, symbol):
    """Inspeciona valores negativos no DataFrame"""
    logging.info(f"Inspecionando valores negativos para {symbol}:")
    
    # Verificar colunas básicas
    for col in ['open', 'high', 'low', 'close']:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logging.warning(f"Coluna {col}: {neg_count} valores negativos")
    
    # Após adicionar indicadores, verificar novamente
    df_with_indicators = add_technical_indicators(df)
    
    # Verificar todas as colunas após indicadores
    for col in df_with_indicators.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_count = (df_with_indicators[col] < 0).sum()
        if neg_count > 0:
            logging.warning(f"Após indicadores - Coluna {col}: {neg_count} valores negativos")
    
    # Retornar valores negativos específicos em macd e outros indicadores
    if 'macd' in df_with_indicators.columns:
        neg_macd = df_with_indicators[df_with_indicators['macd'] < 0]
        if len(neg_macd) > 0:
            logging.info(f"Exemplo de valores MACD negativos: {neg_macd['macd'].head(5).tolist()}")
    
    # Verificar se algum indicador tem predominantemente valores negativos
    for col in df_with_indicators.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_percentage = (df_with_indicators[col] < 0).mean() * 100
        if neg_percentage > 40:  # Se mais de 40% dos valores são negativos
            logging.warning(f"Coluna {col}: {neg_percentage:.2f}% de valores negativos")
    
    return df_with_indicators

# Carregar e inspecionar cada símbolo
for sym in env_cfg['environment']['symbols']:
    logging.info(f"\n--- Analisando dados para {sym} ---")
    try:
        df = pd.read_csv(f'data/{sym}_1h.csv', parse_dates=['timestamp'])
        df['symbol'] = sym
        
        # Inspecionar valores negativos nos dados originais e após indicadores
        df_with_indicators = inspect_negative_values(df, sym)
        
        # Estatísticas gerais
        logging.info(f"Estatísticas para {sym}:")
        logging.info(f"  - Linhas originais: {len(df)}")
        logging.info(f"  - Preço mínimo (close): {df['close'].min()}")
        logging.info(f"  - MACD Range: {df_with_indicators['macd'].min()} a {df_with_indicators['macd'].max()}")
        logging.info(f"  - RSI Range: {df_with_indicators['rsi_14'].min()} a {df_with_indicators['rsi_14'].max()}")
        
    except Exception as e:
        logging.error(f"Erro ao analisar dados para {sym}: {str(e)}") 