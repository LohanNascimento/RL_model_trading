import time
import numpy as np
import pandas as pd
import logging
import os
import json
import yaml
from datetime import datetime
from collections import defaultdict
from stable_baselines3 import PPO
from dryrun.binance_futures_testnet import BinanceFuturesTestnet
from utils.technical_indicators import add_technical_indicators
from utils.risk_manager import RiskManager, RiskParameters
from utils.config_loader import load_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(f'dryrun/logs/execucao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('execucao_modelo')

class ExecutorModelo:
    """
    Classe para executar modelo treinado na Binance Testnet com múltiplos ativos.
    """
    def __init__(self, config_path='config/env_config.yaml', risk_config_path='config/risk_config.yaml'):
        # Cria diretório de logs se não existir
        os.makedirs('dryrun/logs', exist_ok=True)
        
        # Carregar configurações
        self.env_config = load_config(config_path)
        self.risk_config = load_config(risk_config_path)
        
        # Configuração API Binance Testnet
        self.api_key = os.environ.get('BINANCE_TESTNET_KEY', 'af48d3f1963b76c51f52066079b70af894b059d230ef2a850001f4f7e9431327')
        self.api_secret = os.environ.get('BINANCE_TESTNET_SECRET', '5169ec900dc7bb90ef5c28ab5db6ccd1eb1584080efca2ec4b41cd305b690a8b')
        
        # Caminho do modelo
        self.model_path = self.env_config.get('model_path', 'checkpoints/best_model.zip')
        
        # Lista de símbolos para monitorar
        self.symbols = self.env_config['environment']['symbols']
        logger.info(f"Monitorando os seguintes ativos: {self.symbols}")
        
        # Parâmetros de execução
        self.window_size = self.env_config['environment']['window_size']
        self.intervalo_loop = 60  # segundos entre verificações
        self.timeframe = '1h'  # Timeframe usado no treinamento
        
        # Inicializar cliente Binance Testnet
        self.bot = BinanceFuturesTestnet(api_key=self.api_key, api_secret=self.api_secret)
        
        # Inicializar gerenciador de risco
        self.risk_manager = RiskManager(RiskParameters(
            risk_per_trade=self.risk_config['risk_manager']['risk_per_trade'],
            rr_ratio=self.risk_config['risk_manager']['rr_ratio'],
            max_position_size=0.1,  # Limita a 10% do capital por posição
            max_trades_per_day=1,   # Limita a 1 trade por símbolo por dia
            max_drawdown_percent=0.15,  # Stop se drawdown > 15%
            enforce_trade_limit=True     # Ativa limites
        ))
        
        # Carregar modelo treinado
        self.carregar_modelo()
        
        # Dicionário para armazenar posições atuais
        self.posicoes_atuais = {}
        
        # Registro de trades para evitar operações repetidas no mesmo símbolo
        self.trades_hoje = defaultdict(int)
        self.ultima_verificacao = datetime.now()
    
    def carregar_modelo(self):
        """Carrega o modelo treinado"""
        try:
            logger.info(f"Carregando modelo de {self.model_path}")
            self.model = PPO.load(self.model_path)
            logger.info("Modelo carregado com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise

    def obter_observacao(self, symbol):
        """
        Busca os últimos candles, calcula indicadores técnicos
        e prepara observação no formato esperado pelo modelo.
        """
        try:
            # Busca candles OHLCV
            ohlcv = self.bot.fetch_ohlcv(symbol=symbol, timeframe=self.timeframe, 
                                         limit=self.window_size+30)
            
            # Converte para DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            # Calcula indicadores técnicos
            df_feat = add_technical_indicators(df)
            
            # Tratar valores negativos
            for col in df_feat.columns:
                if col in ['timestamp', 'symbol']:
                    continue
                if col in ['macd', 'macd_signal']:
                    # MACD pode ser negativo, pular
                    continue
                if (df_feat[col] < 0).any():
                    if col in ['close', 'open', 'high', 'low', 'volume']:
                        df_feat.loc[df_feat[col] < 0, col] = 0.0001
                    else:
                        df_feat.loc[df_feat[col] < 0, col] = df_feat[col].abs()
            
            # Remove NaNs
            df_feat = df_feat.dropna().reset_index(drop=True)
            
            # Seleciona apenas as últimas linhas necessárias
            df_obs = df_feat.iloc[-self.window_size:]
            
            # Verifica se temos dados suficientes
            if len(df_obs) < self.window_size:
                logger.warning(f"Dados insuficientes para {symbol}: apenas {len(df_obs)} candles disponíveis")
                return None
                
            # Features na mesma ordem usada no treinamento
            features = self.env_config['observation']['features']
            
            # Log das features disponíveis e esperadas
            logger.debug(f"Features disponíveis: {df_obs.columns.tolist()}")
            logger.debug(f"Features esperadas: {features}")
            
            # Verificar se precisamos ajustar as features para corresponder ao modelo
            # O modelo espera 12 features, mas estamos recebendo 14
            if len(features) != 12 and 'close' in features and 'volume' in features:
                # Ver quantas features o modelo espera
                expected_shape = self.model.policy.observation_space.shape
                logger.info(f"Formato esperado pelo modelo: {expected_shape}")
                
                # Se modelo espera 12 features, usar apenas indicadores técnicos básicos
                if expected_shape[1] == 12:
                    reduced_features = [
                        'sma_14', 'ema_14', 'macd', 'macd_signal', 
                        'std_14', 'atr_14', 'bb_upper', 'bb_lower', 
                        'rsi_14', 'stoch_k', 'stoch_d', 'roc_14'
                    ]
                    # Verificar se todas as features reduzidas existem
                    for feat in reduced_features:
                        if feat not in df_obs.columns:
                            logger.error(f"Feature necessária {feat} não encontrada no DataFrame")
                            return None
                    features = reduced_features
                    logger.info(f"Usando conjunto reduzido de features: {len(features)}")
            
            # Verifica se todas as features existem
            for feature in features:
                if feature not in df_obs.columns:
                    logger.error(f"Feature {feature} não encontrada para {symbol}")
                    return None
            
            # Extrai valores das features
            obs = []
            for i in range(len(df_obs)):
                row = df_obs.iloc[i]
                obs.append([row[col] for col in features])
            
            # Converte para numpy array
            obs = np.array(obs, dtype=np.float32)
            
            # Verifica se há NaNs
            if np.isnan(obs).any():
                logger.warning(f"Valores NaN encontrados na observação para {symbol}")
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Observação preparada para {symbol}: shape {obs.shape}")
            return obs
            
        except Exception as e:
            logger.error(f"Erro ao obter observação para {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def decidir_acao(self, symbol):
        """
        Obtém observação e decide ação para um símbolo específico.
        Retorna a ação e informações adicionais.
        """
        # Verifica se já fez trade com este símbolo hoje
        if self.trades_hoje[symbol] >= 1:
            logger.info(f"Já foi realizado o máximo de trades permitidos para {symbol} hoje")
            return None, {}
            
        # Obtém observação atual
        obs = self.obter_observacao(symbol)
        if obs is None:
            return None, {}
            
        # Consulta posição atual
        posicao = self.bot.get_position(symbol)
        
        # Prediz ação com o modelo
        try:
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Verifica se a ação é válida (0, 1 ou 2)
            action = int(action)
            if action not in [0, 1, 2]:
                logger.warning(f"Modelo retornou ação inválida {action} para {symbol}, ajustando para HOLD (0)")
                action = 0
            
            ultimo_candle = self.bot.fetch_ohlcv(symbol=symbol, timeframe=self.timeframe, limit=1)[0]
            preco_atual = ultimo_candle[4]  # Close price
            
            # Log da ação decidida
            acao_texto = "HOLD" if action == 0 else "COMPRAR" if action == 1 else "VENDER"
            logger.info(f"Decisão para {symbol}: {acao_texto} (action={action}) | Preço: {preco_atual:.6f} | Posição atual: {posicao}")
            
            info = {
                'observacao': obs,
                'preco_atual': preco_atual,
                'posicao_atual': posicao,
                'timestamp': datetime.now()
            }
            
            return action, info
            
        except Exception as e:
            logger.error(f"Erro ao predizer ação para {symbol}: {str(e)}")
            return None, {}

    def executar_acao(self, symbol, action, info):
        """
        Executa a ação decidida pelo modelo, respeitando o gerenciador de risco.
        """
        if action is None:
            return False
            
        preco_atual = info['preco_atual']
        posicao_atual = info['posicao_atual']
        tamanho_posicao = abs(posicao_atual) if posicao_atual else 0
        
        # Consulta saldo disponível
        saldo = self.bot.get_balance()
        
        # Verifica limites de risco
        verificacao_risco = self.risk_manager.check_trade_limits(saldo, preco_atual)
        if not verificacao_risco['allowed']:
            logger.warning(f"Operação em {symbol} bloqueada: {verificacao_risco['reason']}")
            return False
            
        # Calcula tamanho da posição com base no risco
        quantidade = self.risk_manager.calculate_position_size(saldo, preco_atual)
        
        # Garante que o valor mínimo da ordem seja pelo menos 100 USDT
        valor_minimo = 100.0  # Valor mínimo em USDT
        min_quantidade = valor_minimo / preco_atual
        quantidade = max(quantidade, min_quantidade)
        
        # Limita a exposição máxima
        max_exposicao = saldo * 0.1  # Máximo de 10% do saldo por posição
        max_quantidade = max_exposicao / preco_atual
        quantidade = min(quantidade, max_quantidade)
        
        # Arredonda a quantidade conforme precisão do ativo
        precisao = 3  # Assumindo 3 casas decimais, pode ser ajustado por ativo
        quantidade = round(quantidade, precisao)
        
        # Log detalhado dos parâmetros da ordem
        logger.info(f"Parâmetros da ordem: Symbol={symbol} | Ação={action} | Quantidade={quantidade:.6f} | "
                   f"Valor={quantidade*preco_atual:.2f} USDT | Saldo={saldo:.2f} USDT | Risco={self.risk_manager.params.risk_per_trade*100:.1f}%")
        
        try:
            # Ação: 0=Hold, 1=Buy, 2=Sell
            if action == 1 and posicao_atual <= 0:  # Comprar
                logger.info(f"Enviando ordem de COMPRA para {symbol} - Qtd: {quantidade:.6f} - Preço: {preco_atual:.6f}")
                resultado = self.bot.send_order(symbol, 'buy', quantidade)
                self.trades_hoje[symbol] += 1
                logger.info(f"Ordem de compra enviada: {resultado}")
                return True
                
            elif action == 2 and posicao_atual >= 0:  # Vender
                logger.info(f"Enviando ordem de VENDA para {symbol} - Qtd: {quantidade:.6f} - Preço: {preco_atual:.6f}")
                resultado = self.bot.send_order(symbol, 'sell', quantidade)
                self.trades_hoje[symbol] += 1
                logger.info(f"Ordem de venda enviada: {resultado}")
                return True
                
            elif action == 2 and posicao_atual > 0:  # Fechar posição comprada
                logger.info(f"Fechando posição COMPRADA existente em {symbol} - Qtd: {tamanho_posicao:.6f}")
                resultado = self.bot.send_order(symbol, 'sell', tamanho_posicao)
                logger.info(f"Ordem de fechamento enviada: {resultado}")
                return True
                
            elif action == 1 and posicao_atual < 0:  # Fechar posição vendida
                logger.info(f"Fechando posição VENDIDA existente em {symbol} - Qtd: {tamanho_posicao:.6f}")
                resultado = self.bot.send_order(symbol, 'buy', tamanho_posicao)
                logger.info(f"Ordem de fechamento enviada: {resultado}")
                return True
                
            else:
                logger.info(f"Ação HOLD para {symbol} (ação={action}, posição={posicao_atual})")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao executar ordem para {symbol}: {str(e)}")
            return False

    def verificar_reset_diario(self):
        """
        Verifica se deve resetar o contador de trades diários.
        """
        agora = datetime.now()
        # Se mudou o dia, reseta contadores
        if agora.date() > self.ultima_verificacao.date():
            logger.info("Novo dia, resetando contadores de trades diários")
            self.trades_hoje = defaultdict(int)
        self.ultima_verificacao = agora

    def registrar_metricas(self):
        """
        Registra métricas de desempenho e estado atual da carteira.
        """
        try:
            # Saldo total
            saldo = self.bot.get_balance()
            
            # Posições atuais
            posicoes = {}
            for symbol in self.symbols:
                pos = self.bot.get_position(symbol)
                if pos != 0:
                    posicoes[symbol] = pos
            
            # Salva relatório
            relatorio = {
                'timestamp': datetime.now().isoformat(),
                'saldo': saldo,
                'posicoes': posicoes,
                'trades_hoje': dict(self.trades_hoje)
            }
            
            # Salva em arquivo JSON
            with open(f'dryrun/logs/status_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(relatorio, f, indent=2)
                
            logger.info(f"Métricas registradas: Saldo={saldo:.2f} | Posições abertas: {len(posicoes)}")
            
        except Exception as e:
            logger.error(f"Erro ao registrar métricas: {str(e)}")

    def executar_loop(self):
        """Loop principal de execução"""
        logger.info("Iniciando loop de execução do modelo")
        
        while True:
            try:
                # Verifica se deve resetar contadores diários
                self.verificar_reset_diario()
                
                acao_tomada = False
                
                # Processa cada símbolo
                for symbol in self.symbols:
                    logger.info(f"Analisando {symbol}...")
                    
                    # Decide ação para este símbolo
                    action, info = self.decidir_acao(symbol)
                    
                    # Executa ação se necessário
                    if action is not None:
                        acao_executada = self.executar_acao(symbol, action, info)
                        acao_tomada = acao_tomada or acao_executada
                
                # Registra métricas a cada ciclo, independente de ter tomado ação
                self.registrar_metricas()
                
                # Log de status
                logger.info(f"Ciclo de verificação completo. Próxima verificação em {self.intervalo_loop} segundos")
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {str(e)}")
            
            # Aguarda até próximo ciclo
            time.sleep(self.intervalo_loop)

def main():
    """Função principal para iniciar a execução do modelo"""
    try:
        executor = ExecutorModelo()
        executor.executar_loop()
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
    except Exception as e:
        logger.critical(f"Erro fatal durante execução: {str(e)}")
        raise

if __name__ == '__main__':
    main()
