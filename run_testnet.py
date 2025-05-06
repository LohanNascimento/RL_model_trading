#!/usr/bin/env python
"""
Script para executar o módulo de execução do modelo na Binance Testnet.
Permite testar o modelo em múltiplos símbolos e monitorar posições.
"""

import argparse
import logging
import os
import time
from datetime import datetime

from dryrun.execucao_modelo import ExecutorModelo
from dryrun.binance_futures_testnet import BinanceFuturesTestnet


def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Execução de modelo na Binance Testnet')
    parser.add_argument('--config', type=str, default='config/env_config.yaml',
                       help='Caminho para o arquivo de configuração do ambiente')
    parser.add_argument('--risk_config', type=str, default='config/risk_config.yaml',
                       help='Caminho para o arquivo de configuração de risco')
    parser.add_argument('--model_path', type=str,
                       help='Caminho para o modelo treinado (sobrescreve config)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Lista de símbolos para monitorar (sobrescreve config)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Intervalo entre verificações (segundos)')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nível de log')
    parser.add_argument('--api_key', type=str,
                       help='Chave de API Binance Testnet')
    parser.add_argument('--api_secret', type=str,
                       help='Chave secreta API Binance Testnet')
    parser.add_argument('--test_trade', action='store_true',
                       help='Envia um trade de teste para verificar conexão')
    parser.add_argument('--check_balance', action='store_true',
                       help='Apenas verificar saldo e posições e sair')
    parser.add_argument('--monitor_only', action='store_true',
                       help='Apenas monitorar sem executar trades')
    parser.add_argument('--run_once', action='store_true',
                       help='Executa apenas um ciclo e sai')
    return parser.parse_args()


def setup_logging(log_level):
    """Configura o sistema de logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Nível de log inválido: {log_level}')
    
    os.makedirs('dryrun/logs', exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(f'dryrun/logs/execucao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('run_testnet')


def test_connection(bot):
    """Testa a conexão com a Binance Testnet"""
    logger = logging.getLogger('run_testnet')
    
    try:
        # Verifica saldo
        saldo = bot.get_balance()
        logger.info(f"Saldo USDT: {saldo}")
        
        # Verifica posições ativas
        posicoes = bot.get_all_positions()
        if posicoes:
            logger.info(f"Posições ativas: {posicoes}")
        else:
            logger.info("Nenhuma posição ativa")
        
        # Verifica timestamp
        server_time = bot.exchange.fapiPublicGetTime()
        logger.info(f"Timestamp do servidor: {server_time['serverTime']}")
        
        # Verifica ordens abertas
        ordens = bot.get_open_orders()
        if ordens:
            logger.info(f"Ordens abertas: {len(ordens)}")
        else:
            logger.info("Nenhuma ordem aberta")
        
        logger.info("Conexão com Binance Testnet OK!")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao testar conexão: {e}")
        return False


def enviar_trade_teste(bot, symbol='BTC/USDT'):
    """Envia um trade de teste com valor mínimo"""
    logger = logging.getLogger('run_testnet')
    
    try:
        # Consulta preço atual
        preco = bot.get_market_price(symbol)
        if not preco:
            logger.error(f"Não foi possível obter preço para {symbol}")
            return False
        
        # Calcula quantidade mínima (100 USDT é o valor mínimo exigido pela Binance)
        quantidade = round(100 / preco, 6)  # 6 casas decimais
        quantidade = max(quantidade, 0.001)  # Garante mínimo de 0.001
        
        logger.info(f"Enviando ordem de teste - Symbol: {symbol} - Side: buy - Quantidade: {quantidade} (valor ~{quantidade*preco:.2f} USDT)")
        
        # Envia ordem de compra
        resultado = bot.send_order(symbol, 'buy', quantidade)
        
        if resultado:
            ordem_id = resultado.get('id', 'N/A')
            logger.info(f"Ordem de teste enviada com sucesso! ID: {ordem_id}")
            
            # Fecha a posição imediatamente
            time.sleep(3)  # Aguarda 3 segundos
            posicao = bot.get_position(symbol)
            
            if posicao > 0:
                logger.info(f"Fechando posição de teste - Symbol: {symbol} - Side: sell - Quantidade: {posicao}")
                resultado_fechamento = bot.send_order(symbol, 'sell', abs(posicao))
                if resultado_fechamento:
                    logger.info("Posição de teste fechada com sucesso!")
                else:
                    logger.warning("Não foi possível fechar posição de teste automaticamente")
            
            return True
        else:
            logger.error("Falha ao enviar ordem de teste")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao enviar trade de teste: {e}")
        return False


def check_balance_and_positions(bot, symbols):
    """Verifica e exibe saldo e posições atuais"""
    logger = logging.getLogger('run_testnet')
    
    try:
        # Resumo da conta
        resumo = bot.get_account_summary()
        
        logger.info("=== RESUMO DA CONTA ===")
        logger.info(f"Saldo USDT: {resumo['balance']:.2f}")
        logger.info(f"Exposição: {resumo['exposure']:.2f} ({resumo['exposure_pct']:.2f}%)")
        logger.info(f"Posições ativas: {resumo['active_positions_count']}")
        
        # Detalhes de cada símbolo
        logger.info("\n=== DETALHES POR SÍMBOLO ===")
        for symbol in symbols:
            preco = bot.get_market_price(symbol)
            posicao = bot.get_position(symbol)
            
            valor_posicao = abs(posicao * preco) if posicao and preco else 0
            tipo_posicao = "LONG" if posicao > 0 else "SHORT" if posicao < 0 else "NEUTRO"
            
            logger.info(f"{symbol}: Preço={preco:.6f} | Posição={posicao} ({tipo_posicao}) | Valor={valor_posicao:.2f} USDT")
        
        return True
    
    except Exception as e:
        logger.error(f"Erro ao verificar saldo e posições: {e}")
        return False


def main():
    """Função principal"""
    # Processa argumentos
    args = parse_arguments()
    
    # Configura logging
    logger = setup_logging(args.log_level)
    logger.info("Iniciando sistema de execução na Binance Testnet")
    
    # Se foram especificados símbolos pela linha de comando, formata corretamente
    formatted_symbols = []
    if args.symbols:
        for s in args.symbols:
            # Converte símbolo para formato com barra (ex: BTCUSDT -> BTC/USDT)
            if '/' not in s:
                # Identifica onde termina o símbolo base (normalmente antes de USDT, BUSD, etc)
                for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']:
                    if s.endswith(quote):
                        base = s[:-len(quote)]
                        formatted_symbols.append(f"{base}/{quote}")
                        break
                else:
                    # Se não conseguiu formatar, usa o original
                    formatted_symbols.append(s)
            else:
                formatted_symbols.append(s)
    
    try:
        # Inicializa cliente Binance Testnet
        bot = BinanceFuturesTestnet(
            api_key=args.api_key,
            api_secret=args.api_secret
        )
        
        # Testa conexão
        if not test_connection(bot):
            logger.error("Falha ao conectar com Binance Testnet. Encerrando.")
            return 1
        
        # Modo de verificação de saldo
        if args.check_balance:
            symbols_to_check = formatted_symbols or ['BTC/USDT', 'ETH/USDT', 'DOT/USDT']
            check_balance_and_positions(bot, symbols_to_check)
            return 0
        
        # Modo de teste de trade
        if args.test_trade:
            test_symbol = formatted_symbols[0] if formatted_symbols else 'BTC/USDT'
            logger.info(f"Realizando trade de teste em {test_symbol}")
            if enviar_trade_teste(bot, test_symbol):
                logger.info("Teste de trade concluído com sucesso!")
            else:
                logger.error("Teste de trade falhou.")
                return 1
            return 0
        
        # Inicia executor do modelo
        executor = ExecutorModelo(
            config_path=args.config,
            risk_config_path=args.risk_config
        )
        
        # Sobrescreve configurações se especificadas
        if args.model_path:
            executor.model_path = args.model_path
            executor.carregar_modelo()
        
        if formatted_symbols:
            executor.symbols = formatted_symbols
            logger.info(f"Sobrescrevendo símbolos para: {executor.symbols}")
        
        if args.interval:
            executor.intervalo_loop = args.interval
        
        if args.monitor_only:
            logger.info("Modo de monitoramento apenas (sem execução de trades)")
            # Sobrescreve método de execução para não executar ordens
            executor.executar_acao = lambda symbol, action, info: (
                logger.info(f"[MONITOR] Ação {action} para {symbol} seria executada. Preço: {info.get('preco_atual', 0):.6f}"),
                False
            )[1]
        
        # Executar loop principal ou apenas uma vez
        if args.run_once:
            logger.info("Executando apenas um ciclo")
            # Processa cada símbolo
            for symbol in executor.symbols:
                logger.info(f"Analisando {symbol}...")
                action, info = executor.decidir_acao(symbol)
                if action is not None and not args.monitor_only:
                    executor.executar_acao(symbol, action, info)
            
            # Registra métricas
            executor.registrar_metricas()
        else:
            # Executa loop contínuo
            executor.executar_loop()
    
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 