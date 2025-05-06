import pandas as pd
from envs.trading_env import TradingEnv
from utils.technical_indicators import add_technical_indicators
from strategy.strategy_rules import TrendFollowingStrategy
from utils.risk_manager import RiskManager, RiskParameters
from utils.config_loader import load_config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
# import matplotlib.pyplot as plt  # Removido, não é mais necessário para gráficos
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import os
import torch
from datetime import datetime

# Configuração de logging padronizada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%H:%M:%S'
)

# Função para limpar valores negativos em um DataFrame
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
                # Para outros indicadores técnicos, substituir por valor absoluto ou mínimo
                df.loc[df[col] < 0, col] = df[col].abs()
    
    if negative_cols:
        logging.warning(f"[{stage_name}] Encontrados valores negativos (já corrigidos):")
        for col, count in negative_cols:
            logging.warning(f"  - Coluna {col}: {count} valores negativos")
    return df

# Função para verificar valores negativos em um DataFrame
def check_negative_values(df, stage_name):
    negative_cols = []
    for col in df.columns:
        if col in ['timestamp', 'symbol']:
            continue
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_cols.append((col, neg_count))
    
    if negative_cols:
        logging.warning(f"[{stage_name}] Encontrados valores negativos:")
        for col, count in negative_cols:
            logging.warning(f"  - Coluna {col}: {count} valores negativos")
    return df

# Função para normalizar dados com StandardScaler, apenas para indicadores técnicos
def normalize_data(train_df, val_df, test_df, feature_cols):
    # Colunas que não devem ser normalizadas (preços e volume)
    price_cols = ['close', 'open', 'high', 'low', 'volume']
    
    # Filtrar apenas indicadores técnicos para normalização
    indicator_cols = [col for col in feature_cols if col not in price_cols]
    logging.info(f"Normalizando apenas indicadores técnicos: {indicator_cols}")
    
    if indicator_cols:
        # Criar cópia dos DataFrames originais
        train_norm = train_df.copy()
        val_norm = val_df.copy()
        test_norm = test_df.copy()
        
        # Aplicar StandardScaler apenas nos indicadores
        scaler = StandardScaler()
        scaler.fit(train_df[indicator_cols])
        
        train_norm[indicator_cols] = scaler.transform(train_df[indicator_cols])
        val_norm[indicator_cols] = scaler.transform(val_df[indicator_cols])
        test_norm[indicator_cols] = scaler.transform(test_df[indicator_cols])
        
        # Verificar se não há valores negativos nas colunas de preço após a normalização
        for col in price_cols:
            if col in train_norm.columns:
                neg_count = (train_norm[col] < 0).sum()
                if neg_count > 0:
                    logging.error(f"Ainda existem {neg_count} valores negativos na coluna {col} após normalização!")
                    
        return train_norm, val_norm, test_norm, scaler
    else:
        logging.warning("Nenhum indicador técnico para normalizar, mantendo dados originais.")
        return train_df, val_df, test_df, None

# Carregar configurações dinâmicas
training_cfg = load_config('config/training_config.yaml')
env_cfg = load_config('config/env_config.yaml')
risk_cfg = load_config('config/risk_config.yaml')

# Configuração de performance
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    logging.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("GPU não disponível, usando CPU")

# Carregar múltiplos ativos se necessário
dfs = []
for sym in env_cfg['environment']['symbols']:
    try:
        df = pd.read_csv(f'data/{sym}_1h.csv', parse_dates=['timestamp'])
        df['symbol'] = sym
        
        # Verificar valores negativos nos dados brutos
        logging.info(f"Verificando dados brutos para {sym}...")
        df = fix_negative_values(df, f"Dados brutos - {sym}")
        
        dfs.append(df)
        logging.info(f"Dados para {sym} carregados com sucesso: {len(df)} linhas.")
    except Exception as e:
        logging.error(f"Erro ao carregar dados para {sym}: {str(e)}")
        
if not dfs:
    logging.error("Nenhum dado foi carregado. Encerrando o script.")
    import sys
    sys.exit(1)
    
df_all = pd.concat(dfs, ignore_index=True)

# Split de dados (train/val/test)
split = training_cfg['split']
total_len = len(df_all)
train_end = int(total_len * split['train_size'])
val_end = train_end + int(total_len * split['val_size'])
train_df = df_all.iloc[:train_end].reset_index(drop=True)
val_df = df_all.iloc[train_end:val_end].reset_index(drop=True)
test_df = df_all.iloc[val_end:].reset_index(drop=True)

# Adiciona indicadores técnicos ao DataFrame
logging.info("Adicionando indicadores técnicos...")
train_df = add_technical_indicators(train_df)
val_df = add_technical_indicators(val_df)
test_df = add_technical_indicators(test_df)

# Verificar e corrigir valores negativos após adicionar indicadores
logging.info("Verificando e corrigindo valores negativos após adicionar indicadores...")
train_df = fix_negative_values(train_df, "Train após indicadores")
val_df = fix_negative_values(val_df, "Val após indicadores")
test_df = fix_negative_values(test_df, "Test após indicadores")

# Remover NaNs
logging.info("Removendo NaNs...")
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Verificar quantidades após remover NaNs
logging.info(f"Após remover NaNs - Train: {len(train_df)} linhas, Val: {len(val_df)} linhas, Test: {len(test_df)} linhas")

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Aplica RSI a todos os splits
def add_rsi_to_df(df):
    df['rsi'] = calc_rsi(df['close'])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Verificar se algum dos DataFrames ficou vazio
if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    logging.error("Um ou mais DataFrames ficaram vazios após processar NaNs!")
    logging.error(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    import sys
    sys.exit(1)

# Normalização de dados
feature_cols = env_cfg['observation']['features']
logging.info(f"Verificando se temos todas as features necessárias...")
for col in feature_cols:
    if col not in train_df.columns:
        logging.error(f"Coluna {col} não encontrada no DataFrame!")
        for col_name in train_df.columns:
            logging.info(f"Coluna disponível: {col_name}")
        import sys
        sys.exit(1)

if env_cfg['observation'].get('normalization', True):
    logging.info("Normalizando features...")
    train_df, val_df, test_df, scaler = normalize_data(train_df, val_df, test_df, feature_cols)

# Criar diretório para logs com timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("ppo_smc_tensorboard", f"run_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

# Inicializa o gerenciador de risco com configuração personalizada
risk_manager = RiskManager(RiskParameters(
    risk_per_trade=risk_cfg['risk_manager']['risk_per_trade'],
    rr_ratio=risk_cfg['risk_manager']['rr_ratio'],
    max_position_size=0.1,  # Limita a 10% do capital por posição
    max_trades_per_day=1000,  # Valor alto para evitar limitações durante o treinamento
    max_drawdown_percent=0.15,  # Interrompe se drawdown > 15%
    enforce_trade_limit=False  # Desativa o limite de trades para o treinamento
))

# Estratégia com configuração personalizada
strategy = TrendFollowingStrategy.from_config(risk_cfg['trend_following'])

# Instanciar ambiente, estratégia e risk manager
try:
    logging.info("Criando ambiente de treinamento...")
    env = TradingEnv(
        df=train_df,
        window_size=env_cfg['environment']['window_size'],
        initial_balance=env_cfg['environment']['initial_balance'],
        fee=env_cfg['environment']['fee'],
        symbols=env_cfg['environment']['symbols'],
        strategy=strategy,
        risk_manager=risk_manager
    )
    logging.info("Ambiente de treinamento criado com sucesso!")

    # Ambiente de validação com os mesmos parâmetros
    logging.info("Criando ambiente de validação...")
    eval_env = Monitor(TradingEnv(
        df=val_df,
        window_size=env_cfg['environment']['window_size'],
        initial_balance=env_cfg['environment']['initial_balance'],
        fee=env_cfg['environment']['fee'],
        symbols=env_cfg['environment']['symbols'],
        strategy=strategy,
        risk_manager=risk_manager
    ))
    logging.info("Ambiente de validação criado com sucesso!")
except Exception as e:
    logging.error(f"Erro ao criar ambientes: {str(e)}")
    import sys
    sys.exit(1)

# Instancia modelo RL com parâmetros do YAML e otimizações de performance
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=training_cfg['training']['learning_rate'],
    n_steps=training_cfg['training']['n_steps'],
    gamma=training_cfg['training']['gamma'],
    gae_lambda=training_cfg['training']['gae_lambda'],
    ent_coef=training_cfg['training']['ent_coef'],
    vf_coef=training_cfg['training']['vf_coef'],
    max_grad_norm=training_cfg['training']['max_grad_norm'],
    tensorboard_log=log_dir,
    device=device,
    verbose=1
)

# Callbacks
# 1. Checkpoint para salvar o modelo periodicamente
checkpoint_callback = CheckpointCallback(
    save_freq=training_cfg['checkpoint']['save_freq'],
    save_path=training_cfg['checkpoint']['save_path'],
    name_prefix='rl_model'
)

# 2. Early stopping para interromper o treinamento quando não houver melhoria
early_stopping_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=training_cfg['early_stopping']['patience'],
    min_evals=5,
    verbose=1
)

# 3. Avaliação periódica para monitorar performance
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=training_cfg['checkpoint']['save_path'],
    log_path=training_cfg['checkpoint']['save_path'],
    eval_freq=training_cfg['checkpoint']['save_freq'],
    n_eval_episodes=5,
    deterministic=True,
    callback_after_eval=early_stopping_callback,
    render=False
)

# Combina todos os callbacks
callback_list = CallbackList([checkpoint_callback, eval_callback])

# Treinamento
try:
    model.learn(
        total_timesteps=training_cfg['training']['total_timesteps'],
        callback=callback_list,
        tb_log_name=f"ppo_smc_{timestamp}"
    )
    
    # Salva o modelo final
    model.save(os.path.join(training_cfg['checkpoint']['save_path'], "final_model"))
    logging.info(f"Modelo final salvo em {os.path.join(training_cfg['checkpoint']['save_path'], 'final_model')}")
    
except Exception as e:
    logging.error(f"Erro durante o treinamento: {str(e)}")
    raise

# Função de validação da estratégia treinada

def evaluate_model(env, model, price_series, plot=True, name='Validação'):
    logging.info("\n===================== [EVAL] Início da avaliação =====================")
    obs, _ = env.reset()  # Compatível Gymnasium
    terminated = False
    truncated = False
    trade_log = []
    prev_position = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        ts = price_series.index[min(env.unwrapped.current_step, len(price_series)-1)]
        # Marca entradas
        if env.unwrapped.position != prev_position:
            logging.debug(f"[TRADE] [{name}] {'ABERTURA' if env.unwrapped.position != 0 else 'FECHAMENTO'} | Tipo: {env.unwrapped.position} | Preço: {info['entry_price'] if env.unwrapped.position != 0 else price_series.iloc[env.unwrapped.current_step-1]:.2f} | Saldo: {info['balance']:.2f}")
            trade_log.append({
                'timestamp': ts,
                'price': info['entry_price'] if env.unwrapped.position != 0 else price_series.iloc[env.unwrapped.current_step-1],
                'type': 'entry' if env.unwrapped.position != 0 else 'exit',
                'position': env.unwrapped.position,
                'prev_position': prev_position,
                'balance': info['balance'],
                'drawdown': info['max_drawdown']
            })
        # Marca saídas (toda vez que fecha posição)
        if env.unwrapped.position == 0 and prev_position != 0:
            logging.debug(f"[TRADE] [{name}] FECHAMENTO | Preço: {price_series.iloc[env.unwrapped.current_step-1]:.2f} | Saldo: {info['balance']:.2f}")
            trade_log.append({
                'timestamp': ts,
                'price': price_series.iloc[env.unwrapped.current_step-1],
                'type': 'exit',
                'position': env.unwrapped.position,
                'prev_position': prev_position,
                'balance': info['balance'],
                'drawdown': info['max_drawdown']
            })
        prev_position = env.unwrapped.position
    # Métricas simples
    if trade_log:
        final_balance = trade_log[-1]['balance']
        drawdowns = [t['drawdown'] for t in trade_log if 'drawdown' in t]
        n_trades = len([t for t in trade_log if t['type']=='exit'])
        logging.info(f"[EVAL] Saldo final: {final_balance:.2f} | Máx. Drawdown: {min(drawdowns):.2f} | Nº de trades: {n_trades}")
        # if plot:
        #     plot_trades_and_performance(trade_log, price_series)
        # Plot removido, relatório será externo.
    else:
        logging.info(f"[EVAL] Nenhuma operação válida encontrada.")
    logging.info("===================== [EVAL] Fim da avaliação =====================\n")

# Função de plot removida, não é mais utilizada para geração de gráficos ou relatório.
# Nova função de validação com log detalhado para plot

def validate_and_plot(env, model, price_series):
    import pandas as pd
    logging.info("\n===================== [EVAL] Início da validação detalhada =====================")
    obs, _ = env.reset()  # Compatível Gymnasium
    terminated = False
    truncated = False
    trade_log = []
    prev_position = 0
    last_entry_price = None
    last_entry_step = None
    last_entry_obs = None
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs_next, _, terminated, truncated, info = env.step(action)
        ts = price_series.index[min(env.unwrapped.current_step, len(price_series)-1)]
        # Marca entradas
        if env.unwrapped.position != prev_position:
            if env.unwrapped.position != 0:  # Entrada
                last_entry_price = info['entry_price']
                last_entry_step = env.unwrapped.current_step
                last_entry_obs = obs.copy() if hasattr(obs, 'copy') else obs
                trade = {
                    'datetime': ts,
                    'type': 'entry',
                    'preco_entrada': info['entry_price'],
                    'preco_saida': '',
                    'pnl': '',
                    'saldo': info['balance'],
                    'motivo': info.get('event',''),
                    'drawdown': info.get('max_drawdown',''),
                }
                obs_flat = obs.flatten()
                for i in range(len(obs_flat)):
                    trade[f'feature_{i}'] = obs_flat[i]
                trade_log.append(trade)
            else:  # Saída manual
                if last_entry_price is not None:
                    pnl = info['balance'] - trade_log[-1]['saldo']
                    trade = {
                        'datetime': ts,
                        'type': 'exit',
                        'preco_entrada': last_entry_price,
                        'preco_saida': price_series.iloc[env.unwrapped.current_step-1],
                        'pnl': pnl,
                        'saldo': info['balance'],
                        'motivo': info.get('event',''),
                        'drawdown': info.get('max_drawdown',''),
                    }
                    obs_flat = obs.flatten()
                    assert len(obs_flat) == 120, f"Esperado {120} features, mas veio {len(obs_flat)}"
                    for i in range(len(obs_flat)):
                        trade[f'feature_{i}'] = obs_flat[i]
                    trade_log.append(trade)
        # Marca saídas automáticas (stop/target)
        if env.unwrapped.position == 0 and prev_position != 0:
            if last_entry_price is not None:
                pnl = info['balance'] - trade_log[-1]['saldo']
                trade = {
                    'datetime': ts,
                    'type': 'exit',
                    'preco_entrada': last_entry_price,
                    'preco_saida': price_series.iloc[env.unwrapped.current_step-1],
                    'pnl': pnl,
                    'saldo': info['balance'],
                    'motivo': info.get('event',''),
                    'drawdown': info.get('max_drawdown',''),
                }
                obs_last = obs[-1]
                for i in range(obs_last.shape[0]):
                    trade[f'feature_{i}'] = obs_last[i]
                trade_log.append(trade)
        prev_position = env.unwrapped.position
        obs = obs_next
    n_trades = len([t for t in trade_log if t['type']=='exit'])
    logging.info(f"[EVAL] Nº de trades na validação: {n_trades}")
    # plot_trades_and_performance(trade_log, price_series)
    # Gráficos removidos, apenas salva trade_log.csv
    
    # Salva o log de trades para análise posterior
    df_trades = pd.DataFrame(trade_log)
    df_trades.to_csv('trade_log.csv', index=False)
    logging.info(f"Log de trades salvo em trade_log.csv com {len(df_trades)} registros")
    
    return df_trades

# Avalia o modelo no conjunto de teste
logging.info("Avaliando modelo final no conjunto de teste...")
test_env = TradingEnv(
    df=test_df,
    window_size=env_cfg['environment']['window_size'],
    initial_balance=env_cfg['environment']['initial_balance'],
    fee=env_cfg['environment']['fee'],
    symbols=env_cfg['environment']['symbols'],
    strategy=strategy,
    risk_manager=risk_manager
)

# Carrega o melhor modelo
if os.path.exists(os.path.join(training_cfg['checkpoint']['save_path'], 'best_model.zip')):
    logging.info("Carregando o melhor modelo...")
    best_model = PPO.load(
        os.path.join(training_cfg['checkpoint']['save_path'], 'best_model'),
        env=test_env,
        device=device
    )
    # Executa validação detalhada no conjunto de teste
    trade_log_df = validate_and_plot(test_env, best_model, test_df)
else:
    logging.warning("Modelo 'best_model.zip' não encontrado. Usando modelo final.")
    # Executa validação detalhada no conjunto de teste
    trade_log_df = validate_and_plot(test_env, model, test_df)

# Análise de performance
if 'pnl' in trade_log_df.columns:
    # Converte a coluna 'pnl' para tipo numérico (float) antes da comparação
    trade_log_df['pnl'] = pd.to_numeric(trade_log_df['pnl'], errors='coerce')
    
    win_rate = (trade_log_df['pnl'] > 0).mean() * 100
    avg_win = trade_log_df[trade_log_df['pnl'] > 0]['pnl'].mean()
    avg_loss = trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].mean()
    profit_factor = abs(trade_log_df[trade_log_df['pnl'] > 0]['pnl'].sum() / 
                        trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum()) if trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
    
    logging.info(f"Estatísticas finais:")
    logging.info(f"- Win Rate: {win_rate:.2f}%")
    logging.info(f"- Profit Factor: {profit_factor:.2f}")
    logging.info(f"- Média de ganhos: {avg_win:.2f}")
    logging.info(f"- Média de perdas: {avg_loss:.2f}")
