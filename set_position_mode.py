#!/usr/bin/env python
"""
Script para configurar o modo de posição da conta Binance Testnet.
Permite alternar entre modo Hedge (posições long e short simultâneas) e
modo One-way (apenas uma direção por vez).
"""

import argparse
import logging
import os
from dryrun.binance_futures_testnet import BinanceFuturesTestnet

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Configurar modo de posição da Binance Testnet')
    parser.add_argument('--mode', choices=['hedge', 'one-way'], required=True,
                      help='Modo de posição desejado: hedge (dual) ou one-way (único)')
    parser.add_argument('--api_key', type=str,
                      help='Chave de API Binance Testnet')
    parser.add_argument('--api_secret', type=str,
                      help='Chave secreta API Binance Testnet')
    parser.add_argument('--test', action='store_true',
                      help='Apenas verificar o modo atual sem alterá-lo')
    return parser.parse_args()

def main():
    """Função principal"""
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logger = logging.getLogger('set_position_mode')
    
    # Processa argumentos
    args = parse_arguments()
    
    # Se API keys foram fornecidas via argumentos, usá-las
    api_key = args.api_key
    api_secret = args.api_secret
    
    # Se não, verificar variáveis de ambiente
    if not api_key:
        api_key = os.environ.get('BINANCE_TESTNET_KEY')
    if not api_secret:
        api_secret = os.environ.get('BINANCE_TESTNET_SECRET')
    
    try:
        # Se modo de teste, apenas verificar modo atual
        if args.test:
            logger.info("Modo de teste: apenas verificando configuração atual")
            bot = BinanceFuturesTestnet(api_key=api_key, api_secret=api_secret)
            return 0
        
        # Inicializa cliente Binance com o modo de posição solicitado
        logger.info(f"Configurando modo de posição para: {args.mode.upper()}")
        bot = BinanceFuturesTestnet(
            api_key=api_key,
            api_secret=api_secret,
            position_mode=args.mode
        )
        
        # Verifica modo atual para confirmar
        position_mode = bot.exchange.fapiPrivateGetPositionSideDual()
        is_hedge_mode = position_mode.get('dualSidePosition', False)
        current_mode = "HEDGE" if is_hedge_mode else "ONE-WAY"
        
        if (args.mode == 'hedge' and is_hedge_mode) or (args.mode == 'one-way' and not is_hedge_mode):
            logger.info(f"✅ Configuração bem sucedida! Modo atual: {current_mode}")
            
            # Exibe instruções sobre como operar em cada modo
            if is_hedge_mode:
                logger.info("📚 No modo HEDGE, você precisa especificar 'positionSide' nas ordens:")
                logger.info("  - Para long: positionSide=LONG")
                logger.info("  - Para short: positionSide=SHORT")
                logger.info("  - É possível ter posições long e short simultaneamente no mesmo ativo")
            else:
                logger.info("📚 No modo ONE-WAY:")
                logger.info("  - Não é necessário especificar 'positionSide'")
                logger.info("  - Comprar aumenta/abre posição long")
                logger.info("  - Vender aumenta/abre posição short")
                logger.info("  - Para reverter posição, é preciso fechar a atual e abrir outra")
                
            return 0
        else:
            logger.error(f"❌ Falha ao configurar modo! Modo atual: {current_mode}")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Erro: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 