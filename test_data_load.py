import pandas as pd
import logging
import numpy as np
from utils.config_loader import load_config
from utils.technical_indicators import add_technical_indicators

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Carregar configuração
env_cfg = load_config('config/env_config.yaml')

# Função para validar dataframe (versão simplificada da função no TradingEnv)
def validate_dataframe(df):
    # Verifica se temos as colunas necessárias
    required_cols = ['close', 'open', 'high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame deve conter a coluna {col}")
            
    # Verifica valores inválidos (NaN, infinitos, negativos)
    for col in required_cols:
        if df[col].isna().any():
            logging.warning(f"Encontradas {df[col].isna().sum()} linhas com NaN na coluna {col}. Removendo...")
            df = df.dropna(subset=[col])
            
        if (df[col] < 0).any():
            logging.warning(f"Encontradas {(df[col] < 0).sum()} linhas com valores negativos na coluna {col}. Removendo...")
            df = df[df[col] >= 0]
    
    return df

# Carregar dados dos múltiplos ativos
dfs = []
for sym in env_cfg['environment']['symbols']:
    logging.info(f"Carregando dados para {sym}...")
    try:
        df = pd.read_csv(f'data/{sym}_1h.csv', parse_dates=['timestamp'])
        df['symbol'] = sym
        
        # Validar dataframe
        original_len = len(df)
        df = validate_dataframe(df)
        if len(df) < original_len:
            logging.warning(f"Removidas {original_len - len(df)} linhas de {sym} durante a validação.")
        
        # Adicionar indicadores técnicos
        df = add_technical_indicators(df)
        
        dfs.append(df)
        logging.info(f"Dados para {sym} carregados com sucesso: {len(df)} linhas após validação e adição de indicadores.")
    except Exception as e:
        logging.error(f"Erro ao carregar dados para {sym}: {str(e)}")

# Concatenar todos os dataframes
if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    logging.info(f"Dataset combinado: {len(df_all)} linhas no total.")
    
    # Mostrar um resumo dos dados
    for sym in env_cfg['environment']['symbols']:
        sym_data = df_all[df_all['symbol'] == sym]
        logging.info(f"Resumo para {sym}:")
        logging.info(f"  - Total de linhas: {len(sym_data)}")
        logging.info(f"  - Intervalo de datas: {sym_data['timestamp'].min()} a {sym_data['timestamp'].max()}")
        logging.info(f"  - Preço mínimo: {sym_data['close'].min():.6f}, Preço máximo: {sym_data['close'].max():.6f}")
else:
    logging.error("Nenhum dataframe foi carregado com sucesso.") 