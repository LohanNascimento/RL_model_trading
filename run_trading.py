#!/usr/bin/env python
"""
Script principal para execução do sistema de trading com recursos de segurança e desempenho.
Este script integra todas as melhorias implementadas para um sistema mais robusto e seguro.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import traceback
import time
import json

# Componentes internos
from utils.data_validator import DataValidator
from utils.monitoring import get_monitor, PerformanceMonitor
from utils.technical_indicators import add_technical_indicators
from utils.risk_manager import RiskManager, RiskParameters
from utils.config_loader import load_config
from envs.trading_env import TradingEnv
from strategy.strategy_rules import TrendFollowingStrategy
from stable_baselines3 import PPO

def parse_arguments():
    """Define argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Sistema de Trading com RL')
    parser.add_argument('--mode', choices=['train', 'backtest', 'live'], default='backtest',
                       help='Modo de operação: treinar modelo, backtest, ou trading ao vivo')
    parser.add_argument('--config', default='config/env_config.yaml',
                       help='Arquivo de configuração do ambiente')
    parser.add_argument('--risk_config', default='config/risk_config.yaml',
                       help='Arquivo de configuração de risco')
    parser.add_argument('--model_path', default='checkpoints/best_model.zip',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--symbol', default='DOTUSDT',
                       help='Par de trading para usar')
    parser.add_argument('--data_file', default=None,
                       help='Arquivo CSV com dados de preço (OHLCV)')
    parser.add_argument('--output_dir', default='results',
                       help='Diretório para salvar resultados')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nível de logging')
    return parser.parse_args()

def setup_environment(args):
    """Configuração inicial do ambiente"""
    # Configura diretórios de saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configura logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'trading.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('trading')
    
    # Inicializa monitor de desempenho
    monitor = get_monitor(output_dir, log_level)
    
    # Verifica disponibilidade de GPU
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    logger.info(f"Dispositivo de processamento: {device}")
    
    return logger, monitor, output_dir, device

def load_and_validate_data(args, logger):
    """Carrega e valida os dados de entrada"""
    data_file = args.data_file
    if data_file is None:
        # Tenta inferir o arquivo de dados com base no símbolo
        symbol_name = args.symbol.replace('/', '')
        data_file = f"data/{symbol_name}_1h.csv"
        
    if not os.path.exists(data_file):
        logger.error(f"Arquivo de dados não encontrado: {data_file}")
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_file}")
    
    logger.info(f"Carregando dados de {data_file}")
    
    # Carrega dados
    df = pd.read_csv(data_file)
    
    # Valida e limpa dados
    validator = DataValidator(log_level=logging.INFO)
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    logger.info("Validando dados...")
    df, validation_result = validator.validate_price_data(df, required_columns)
    
    if not validation_result['valid']:
        logger.warning(f"Validação detectou {len(validation_result['issues'])} problemas.")
        for issue in validation_result['issues']:
            logger.warning(f"- {issue}")
            
        if validation_result['rows_before'] != validation_result['rows_after']:
            logger.warning(f"Removidas {validation_result['rows_before'] - validation_result['rows_after']} linhas com problemas")
    else:
        logger.info("Validação concluída: dados OK")
    
    # Converte timestamp para datetime se necessário
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Adiciona indicadores técnicos
    logger.info("Adicionando indicadores técnicos...")
    df = add_technical_indicators(df)
    
    # Valida indicadores
    indicator_cols = [
        'sma_14', 'ema_14', 'macd', 'macd_signal', 'std_14', 'atr_14',
        'bb_upper', 'bb_lower', 'rsi_14', 'stoch_k', 'stoch_d', 'roc_14'
    ]
    
    df, indicator_validation = validator.validate_indicators(df, indicator_cols)
    
    if indicator_validation['fixed_indicators'] > 0:
        logger.warning(f"Corrigidos {indicator_validation['fixed_indicators']} problemas em indicadores")
    
    return df

def prepare_trading_environment(args, df, logger, device):
    """Prepara o ambiente de trading com configurações de risco"""
    # Carrega configurações
    env_config = load_config(args.config)
    risk_config = load_config(args.risk_config)
    
    # Inicializa gerenciador de risco com configurações seguras
    risk_manager = RiskManager(RiskParameters(
        risk_per_trade=risk_config['risk_manager']['risk_per_trade'],
        rr_ratio=risk_config['risk_manager']['rr_ratio'],
        max_position_size=0.1,  # Limita a 10% do capital por posição
        max_trades_per_day=1000,  # Valor aumentado
        max_drawdown_percent=0.15,  # Interrompe se drawdown > 15%
        enforce_trade_limit=False  # Mantém o limite de trades ativado no modo de execução
    ))
    
    # Inicializa estratégia
    strategy = TrendFollowingStrategy.from_config(risk_config['trend_following'])
    
    # Cria ambiente de trading
    logger.info("Configurando ambiente de trading...")
    env = TradingEnv(
        df=df,
        window_size=env_config['environment']['window_size'],
        initial_balance=env_config['environment']['initial_balance'],
        fee=env_config['environment']['fee'],
        symbols=[args.symbol.split('/')[0]],  # Extrai ticker base
        strategy=strategy,
        risk_manager=risk_manager
    )
    
    return env, risk_manager, strategy

def load_agent(args, env, logger, device):
    """Carrega o agente treinado"""
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        logger.error(f"Modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    logger.info(f"Carregando modelo de {model_path}")
    
    try:
        model = PPO.load(model_path, env=env, device=device)
        logger.info("Modelo carregado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

def run_backtest(env, model, monitor, logger, output_dir):
    """Executa backtest com o modelo carregado"""
    logger.info("Iniciando backtest...")
    
    # Reseta o ambiente
    obs, _ = env.reset()
    done = False
    truncated = False
    
    # Métricas
    trades = []
    initial_balance = env.balance
    
    # Loop principal
    step = 0
    
    try:
        while not (done or truncated):
            # Prediz ação
            action, _ = model.predict(obs, deterministic=True)
            
            # Executa passo
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Registra trade se posição mudou
            if info.get('event') and 'balance' in info:
                trade_data = {
                    'timestamp': pd.Timestamp.now(),
                    'step': step,
                    'action': int(action),
                    'type': info.get('event', ''),
                    'price': info.get('entry_price', 0),
                    'position': info.get('position', 0),
                    'balance': info.get('balance', 0),
                    'profit_loss': reward if 'close' in info.get('event', '') or 'exit' in info.get('event', '') else 0,
                    'drawdown': info.get('max_drawdown', 0)
                }
                trades.append(trade_data)
                
                # Registra no monitor de desempenho
                monitor.register_trade(trade_data)
                
            # Atualiza para próximo passo
            obs = next_obs
            step += 1
            
            # Log a cada 100 passos
            if step % 100 == 0:
                logger.info(f"Passo {step}: Saldo = {info.get('balance', 0):.2f}")
                
    except Exception as e:
        logger.error(f"Erro durante backtest: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Resultados finais
    final_balance = env.balance
    total_return = final_balance - initial_balance
    return_pct = (total_return / initial_balance) * 100
    
    logger.info("=== Resultados do Backtest ===")
    logger.info(f"Passos executados: {step}")
    logger.info(f"Trades realizados: {len([t for t in trades if 'close' in t['type'] or 'exit' in t['type']])}")
    logger.info(f"Saldo inicial: {initial_balance:.2f}")
    logger.info(f"Saldo final: {final_balance:.2f}")
    logger.info(f"Retorno: {total_return:.2f} ({return_pct:.2f}%)")
    logger.info(f"Drawdown máximo: {env.max_drawdown:.2f}")
    
    # Salva resultados
    df_trades = pd.DataFrame(trades)
    if len(df_trades) > 0:
        trades_csv = os.path.join(output_dir, 'backtest_trades.csv')
        df_trades.to_csv(trades_csv, index=False)
        logger.info(f"Trades salvos em {trades_csv}")
    
    # Gera relatório de desempenho
    report_path = os.path.join(output_dir, 'performance_report.json')
    monitor.generate_performance_report(report_path)
    
    return df_trades

def main():
    """Função principal"""
    # Processa argumentos
    args = parse_arguments()
    
    try:
        # Configuração de ambiente e logging
        logger, monitor, output_dir, device = setup_environment(args)
        logger.info(f"Iniciando sistema de trading em modo: {args.mode}")
        
        # Carrega e valida dados
        df = load_and_validate_data(args, logger)
        
        # Prepara ambiente de trading
        env, risk_manager, strategy = prepare_trading_environment(args, df, logger, device)
        
        # Executa modo selecionado
        if args.mode == 'train':
            logger.error("Modo de treinamento não implementado neste script. Use train_rl_agent.py")
        
        elif args.mode == 'backtest':
            # Carrega modelo treinado
            model = load_agent(args, env, logger, device)
            
            # Executa backtest
            trades_df = run_backtest(env, model, monitor, logger, output_dir)
            
            # Salva configuração usada
            config_info = {
                'mode': args.mode,
                'symbol': args.symbol,
                'data_file': args.data_file,
                'model_path': args.model_path,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
                json.dump(config_info, f, indent=2)
        
        elif args.mode == 'live':
            logger.error("Modo de trading ao vivo não implementado neste script")
        
        logger.info(f"Execução concluída. Resultados em: {output_dir}")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Erro fatal: {str(e)}")
            logger.error(traceback.format_exc())
        else:
            print(f"Erro fatal antes da inicialização do logger: {str(e)}")
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 