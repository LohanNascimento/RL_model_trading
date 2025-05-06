import ccxt
import time
import logging
import traceback

class BinanceFuturesTestnet:
    """
    Módulo de integração para operar na Binance Futures Testnet usando CCXT.
    Permite enviar ordens, consultar saldo e monitorar posições em ambiente de simulação.
    Suporta múltiplos símbolos e inclui melhor tratamento de erros.
    """
    def __init__(self, api_key=None, api_secret=None, default_symbol='ETH/USDT', position_mode=None):
        # Credenciais de API
        self.api_key = api_key or 'af48d3f1963b76c51f52066079b70af894b059d230ef2a850001f4f7e9431327'
        self.api_secret = api_secret or '5169ec900dc7bb90ef5c28ab5db6ccd1eb1584080efca2ec4b41cd305b690a8b'
        self.default_symbol = default_symbol
        
        # Configuração da exchange
        self.exchange = ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            },
            'recvWindow': 60000,
            'timeout': 30000  # Aumenta timeout para evitar erros de conexão
        })
        
        # Ativa modo sandbox (testnet)
        self.exchange.set_sandbox_mode(True)
        
        try:
            # Carrega mercados e verifica credenciais
            self.exchange.load_markets()
            self.exchange.check_required_credentials()
            
            # Sincroniza timestamp
            self._sync_server_time()
            
            # Configura modo de posição (hedge ou one-way)
            # position_mode pode ser 'hedge', 'one-way' ou None (manter atual)
            self._configure_hedge_mode(position_mode)
            
            logging.info('Binance Futures Testnet (CCXT) inicializado em modo sandbox.')
        except Exception as e:
            logging.error(f'Erro ao inicializar Binance Futures Testnet: {e}')
            logging.error(traceback.format_exc())
            raise

    def _sync_server_time(self):
        """Sincroniza o timestamp local com o servidor"""
        try:
            server_time = self.exchange.fapiPublicGetTime()
            server_time_ms = int(server_time['serverTime'])
            local_time_ms = int(time.time() * 1000)
            time_diff = server_time_ms - local_time_ms
            
            self.exchange.options['timestamp'] = server_time_ms
            logging.info(f'Timestamp sincronizado. Diferença: {time_diff}ms')
            return True
        except Exception as e:
            logging.warning(f'Erro ao sincronizar timestamp: {e}')
            return False

    def _configure_hedge_mode(self, force_mode=None):
        """
        Configura o modo de posição para negociação de futuros.
        
        Args:
            force_mode: Se definido como 'hedge' ou 'one-way', força esse modo específico.
                        Se None, mantém o modo atual da conta.
        
        Returns:
            bool: True se a configuração foi bem-sucedida, False caso contrário
        """
        try:
            # Verifica o modo atual
            position_mode = self.exchange.fapiPrivateGetPositionSideDual()
            current_mode = 'hedge' if position_mode.get('dualSidePosition', False) else 'one-way'
            
            # Decide se deve mudar o modo
            if force_mode is None:
                # Se não forçar um modo, apenas registra o modo atual
                logging.info(f'Modo de posição atual: {current_mode.upper()}')
                return True
                
            # Se já estiver no modo desejado, não faz nada
            if (force_mode == 'hedge' and current_mode == 'hedge') or \
               (force_mode == 'one-way' and current_mode == 'one-way'):
                logging.info(f'Já está em modo {force_mode.upper()}')
                return True
                
            # Muda para o modo desejado
            if force_mode == 'hedge':
                # Muda para modo Hedge (dualSidePosition = true)
                self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': True})
                logging.info('Modo de posição alterado para HEDGE')
            elif force_mode == 'one-way':
                # Muda para modo One-way (dualSidePosition = false)
                self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': False})
                logging.info('Modo de posição alterado para ONE-WAY')
            else:
                logging.warning(f'Modo inválido: {force_mode}. Use "hedge" ou "one-way".')
                return False
                
            return True
        except Exception as e:
            logging.warning(f'Erro ao configurar modo de posição: {e}')
            return False

    def get_balance(self, asset='USDT'):
        """Consulta saldo disponível do ativo."""
        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get(asset, 0)
        except Exception as e:
            logging.error(f'Erro ao consultar saldo: {e}')
            return 0

    def _format_symbol(self, symbol):
        """Converte o símbolo para o formato correto da Exchange"""
        if symbol is None:
            return self.default_symbol
        
        # Se já tem barra, retorna como está
        if '/' in symbol:
            return symbol
        
        # Tenta converter para formato com barra (ex: BTCUSDT -> BTC/USDT)
        for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # Se não conseguiu identificar, retorna o original
        return symbol

    def get_position(self, symbol=None):
        """
        Consulta posição aberta atual para o símbolo.
        Retorna o tamanho da posição (positivo para long, negativo para short, 0 se não houver posição).
        """
        symbol = self._format_symbol(symbol)
        
        try:
            positions = self.exchange.fetch_positions([symbol])
            position_size = 0
            
            for pos in positions:
                if pos['symbol'] == symbol.replace('/', ''):
                    if pos['side'] == 'long':
                        position_size += float(pos['contracts'] or 0)
                    elif pos['side'] == 'short':
                        position_size -= float(pos['contracts'] or 0)
            
            return position_size
        except Exception as e:
            logging.error(f'Erro ao consultar posição para {symbol}: {e}')
            return 0

    def get_all_positions(self):
        """Retorna todas as posições abertas"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = {}
            
            for pos in positions:
                if float(pos['contracts'] or 0) > 0:
                    symbol = pos['symbol']
                    side = pos['side']
                    size = float(pos['contracts'] or 0)
                    
                    if symbol not in active_positions:
                        active_positions[symbol] = 0
                    
                    if side == 'long':
                        active_positions[symbol] += size
                    elif side == 'short':
                        active_positions[symbol] -= size
            
            return {k: v for k, v in active_positions.items() if v != 0}
        except Exception as e:
            logging.error(f'Erro ao consultar todas as posições: {e}')
            return {}

    def send_order(self, symbol, side, quantity, order_type='market', price=None, reduce_only=False):
        """
        Envia ordem para a Binance Futures Testnet.
        
        Args:
            symbol: Par de negociação (ex: 'BTC/USDT')
            side: 'buy' ou 'sell'
            quantity: Quantidade a ser negociada
            order_type: 'market' ou 'limit'
            price: Preço limite (apenas para order_type='limit')
            reduce_only: Se True, a ordem apenas reduz posição existente
            
        Returns:
            Objeto de ordem ou None em caso de erro
        """
        # Formata símbolo corretamente
        symbol = self._format_symbol(symbol)
        
        # Se quantidade for zero ou negativa, não faz nada
        if quantity <= 0:
            logging.warning(f'Quantidade inválida para ordem: {quantity}')
            return None
            
        try:
            # Define parâmetros adicionais
            params = {}
            
            # Verifica qual o modo de posição atual da conta
            try:
                position_mode = self.exchange.fapiPrivateGetPositionSideDual()
                is_hedge_mode = position_mode.get('dualSidePosition', False)
                
                # Se estiver em modo hedge, especificar positionSide
                if is_hedge_mode:
                    position_side = 'LONG' if side == 'buy' else 'SHORT'
                    params['positionSide'] = position_side
                    logging.info(f"Usando modo hedge com positionSide={position_side}")
            except Exception as e:
                logging.warning(f"Não foi possível verificar modo de posição: {e}")
            
            # Adiciona reduceOnly se for uma ordem de redução
            if reduce_only:
                params['reduceOnly'] = True
            
            # Envia ordem conforme o tipo
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, quantity, params=params)
            elif order_type == 'limit' and price is not None:
                order = self.exchange.create_limit_order(symbol, side, quantity, price, params=params)
            else:
                raise ValueError('Tipo de ordem inválido ou preço não especificado para limit order.')
            
            logging.info(f'Ordem enviada para {symbol}: {side} {quantity} @ {price if price else "MARKET"}')
            return order
        except Exception as e:
            logging.error(f'Erro ao enviar ordem para {symbol}: {e}')
            logging.error(traceback.format_exc())
            return None

    def get_open_orders(self, symbol=None):
        """Consulta ordens abertas para o símbolo. Se symbol=None, retorna todas as ordens."""
        if symbol is not None:
            symbol = self._format_symbol(symbol)
        
        try:
            return self.exchange.fetch_open_orders(symbol=symbol)
        except Exception as e:
            logging.error(f'Erro ao consultar ordens abertas: {e}')
            return []

    def cancel_all_orders(self, symbol=None):
        """Cancela todas as ordens abertas para o símbolo. Se symbol=None, usa o símbolo padrão."""
        symbol = self._format_symbol(symbol)
        
        try:
            open_orders = self.get_open_orders(symbol=symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol=symbol)
            logging.info(f'Todas as ordens abertas para {symbol} foram canceladas.')
            return True
        except Exception as e:
            logging.error(f'Erro ao cancelar ordens para {symbol}: {e}')
            return False

    def fetch_ohlcv(self, symbol=None, timeframe='1h', limit=100):
        """
        Busca candles OHLCV para o símbolo.
        Usa API spot para garantir funcionamento mesmo na testnet.
        
        Args:
            symbol: Par de negociação (ex: 'BTC/USDT')
            timeframe: Intervalo de tempo ('1m', '5m', '15m', '1h', '4h', '1d', etc)
            limit: Número de candles a retornar
            
        Returns:
            Lista de candles OHLCV ou lista vazia em caso de erro
        """
        symbol = self._format_symbol(symbol)
        
        try:
            # Usa a API spot para obter dados OHLCV (mais confiável)
            spot = ccxt.binance()
            return spot.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            logging.error(f'Erro ao buscar OHLCV para {symbol}: {e}')
            return []

    def get_market_price(self, symbol=None):
        """Obtém o preço atual de mercado para o símbolo"""
        symbol = self._format_symbol(symbol)
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f'Erro ao obter preço de mercado para {symbol}: {e}')
            # Tenta alternativa
            try:
                ohlcv = self.fetch_ohlcv(symbol, '1m', 1)
                if ohlcv and len(ohlcv) > 0:
                    return ohlcv[0][4]  # Último preço de fechamento
            except:
                pass
            return None

    def get_account_summary(self):
        """Retorna um resumo da conta com saldo, posições e estatísticas"""
        try:
            balance = self.get_balance()
            positions = self.get_all_positions()
            
            # Calcula exposição total
            exposure = 0
            for symbol, pos_size in positions.items():
                price = self.get_market_price(symbol)
                if price and pos_size:
                    exposure += abs(pos_size * price)
            
            # Calcula exposição percentual
            exposure_pct = (exposure / balance) * 100 if balance > 0 else 0
            
            return {
                'balance': balance,
                'positions': positions,
                'active_positions_count': len(positions),
                'exposure': exposure,
                'exposure_pct': exposure_pct,
                'timestamp': int(time.time() * 1000)
            }
        except Exception as e:
            logging.error(f'Erro ao obter resumo da conta: {e}')
            return {
                'balance': 0,
                'positions': {},
                'active_positions_count': 0,
                'exposure': 0,
                'exposure_pct': 0,
                'timestamp': int(time.time() * 1000),
                'error': str(e)
            }

if __name__ == '__main__':
    # Exemplo de uso
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    bot = BinanceFuturesTestnet()
    print('Saldo USDT:', bot.get_balance())
    print('Posições ativas:', bot.get_all_positions())
    print('Resumo da conta:', bot.get_account_summary())
    
    # Exemplo de consulta OHLCV
    ohlcv = bot.fetch_ohlcv('BTC/USDT', '1h', 5)
    if ohlcv:
        print('Últimos 5 candles BTC/USDT:')
        for candle in ohlcv:
            timestamp, open_price, high, low, close, volume = candle
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))
            print(f"{timestamp_str}: O={open_price} H={high} L={low} C={close} V={volume}")
