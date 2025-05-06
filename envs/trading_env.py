import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import logging

from utils.risk_manager import RiskManager, RiskParameters

class TradingEnv(gym.Env):
    """
    Ambiente de RL customizado para trading com Smart Money Concepts (SMC).
    O vetor de estado inclui preço, volume, sinais SMC (FVG, BOS, CHOCH, liquidez), e indicadores técnicos.
    Ações discretas: 0 = Manter, 1 = Comprar, 2 = Vender.
    Recompensa baseada em P&L líquido e penalização de risco.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size=10, initial_balance=10000, fee=0.001, strategy=None, risk_manager=None, symbols=None, log_level=logging.INFO):
        super().__init__()
        # Configura logging
        self.strategy = strategy  # Garante que sempre existe o atributo strategy
        logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s:%(message)s')
        self.symbols = symbols
        self.df = df.reset_index(drop=True)
        self.window_size = window_size  # Definido antes da validação
        
        # Validação dos dados de entrada
        self._validate_dataframe()
        
        self.initial_balance = initial_balance
        self.fee = fee
        self.stop_loss = None
        self.take_profit = None
        self.action_space = spaces.Discrete(3)
        # Lista de indicadores técnicos
        self.indicator_cols = [
            'sma_14', 'ema_14', 'macd', 'macd_signal', 'std_14', 'atr_14',
            'bb_upper', 'bb_lower', 'rsi_14', 'stoch_k', 'stoch_d', 'roc_14'
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(self.indicator_cols)), dtype=np.float32
        )
        
        # Inicializa o RiskManager com parâmetros padrão ou customizados
        if isinstance(risk_manager, RiskManager):
            self.risk_manager = risk_manager
        elif isinstance(risk_manager, dict):
            self.risk_manager = RiskManager(RiskParameters(**risk_manager))
        else:
            # Parâmetros padrão
            self.risk_manager = RiskManager()
            
        self.risk_manager.log_risk_parameters()
        self.reset()

    def _validate_dataframe(self):
        """Valida o DataFrame para garantir integridade dos dados"""
        # Verifica se temos as colunas necessárias
        required_cols = ['close', 'open', 'high', 'low']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame deve conter a coluna {col}")
                
        # Verifica valores inválidos (NaN, infinitos, negativos)
        for col in required_cols:
            if self.df[col].isna().any():
                logging.warning(f"Encontradas {self.df[col].isna().sum()} linhas com NaN na coluna {col}. Removendo...")
                self.df = self.df.dropna(subset=[col])
                
            if (self.df[col] < 0).any():
                logging.warning(f"Encontradas {(self.df[col] < 0).sum()} linhas com valores negativos na coluna {col}. Removendo...")
                self.df = self.df[self.df[col] > 0]  # Alterado: remove apenas valores negativos, mantém zeros
                
        # Ordena pelo timestamp, se existir
        if 'timestamp' in self.df.columns:
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
        if len(self.df) <= self.window_size:
            raise ValueError(f"DataFrame muito pequeno: {len(self.df)} linhas. Necessário pelo menos {self.window_size + 1} linhas.")

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        obs = []
        for i in range(len(window)):
            row = window.iloc[i]
            obs.append([
                row[col] if not pd.isna(row[col]) else 0.0 for col in self.indicator_cols
            ])
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.isnan(obs).any(), f"Encontrado NaN na observação: {obs}"
        return obs

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0  # +1 comprado, -1 vendido, 0 neutro
        self.entry_price = 0
        self.current_step = self.window_size
        self.done = False
        self.total_profit = 0
        self.max_drawdown = 0
        self.episode_profit = 0
        self.last_balance = self.initial_balance
        self.trade_count = 0
        
        # Reseta o histórico de preços no risk manager
        self.risk_manager.price_history = []
        
        obs = self._get_observation()
        assert not np.isnan(obs).any(), f"Encontrado NaN na observação no reset: {obs}"
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        row = self.df.iloc[self.current_step]
        price = row["close"]
        reward = 0
        info = {}
        terminated = False
        truncated = False
        event = None

        # Validação dos dados de entrada
        if np.isnan(price) or np.isinf(price) or price <= 0:
            logging.warning(f"Preço inválido detectado em step {self.current_step}: {price}")
            self.done = True
            return self._get_observation(), -100, True, False, {"error": "Preço inválido"}

        # Verifica limites de risco antes de qualquer ação
        risk_check = self.risk_manager.check_trade_limits(self.balance, price)
        if not risk_check['allowed'] and action != 0:  # Só permite manter posição
            logging.warning(f"Ação bloqueada: {risk_check['reason']}")
            event = f"blocked_{risk_check['reason'].replace(' ', '_')}"
            action = 0  # Força "manter"
            # Pequena penalidade por tentar violar limites de risco
            reward = -0.01

        # Parâmetros de risco
        position_type = 'long' if action == 1 else 'short'
        position_size = self.risk_manager.calculate_position_size(self.balance, price)
        
        # Limita o tamanho da posição para evitar valores extremos
        max_safe_position = 1000.0 / price  # Limita o valor máximo da posição
        position_size = min(position_size, max_safe_position)
        
        # Execução da ação
        if action == 1 and self.position == 0:  # Comprar
            self.position = 1
            self.entry_price = price
            # Calcula stop loss e take profit para long
            self.stop_loss, self.take_profit = self._calculate_risk_levels(price, 'long')
            logging.info(f"[ABERTURA] LONG em {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f} | Saldo: {self.balance:.2f}")
            event = 'open_long'
            self.trade_count += 1
            self.risk_manager.register_trade({
                'type': 'open_long',
                'price': price,
                'position_size': position_size
            })
        elif action == 2 and self.position == 0:  # Vender
            self.position = -1
            self.entry_price = price
            # Calcula stop loss e take profit para short
            self.stop_loss, self.take_profit = self._calculate_risk_levels(price, 'short')
            logging.info(f"[ABERTURA] SHORT em {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f} | Saldo: {self.balance:.2f}")
            event = 'open_short'
            self.trade_count += 1
            self.risk_manager.register_trade({
                'type': 'open_short',
                'price': price,
                'position_size': position_size
            })
        elif action == 2 and self.position == 1:  # Fecha compra manualmente
            profit = (price - self.entry_price) * position_size * (1 - self.fee)
            # Limita o valor do lucro para evitar overflow
            profit = np.clip(profit, -1e6, 1e6)
            self.balance += profit
            self.position = 0
            self.entry_price = 0
            reward = profit
            logging.info(f"[FECHAMENTO MANUAL] LONG em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance:.2f}")
            event = 'close_long_manual'
            self.risk_manager.register_trade({
                'type': 'close_long_manual',
                'price': price,
                'profit': profit
            })
        elif action == 1 and self.position == -1:  # Fecha venda manualmente
            profit = (self.entry_price - price) * position_size * (1 - self.fee)
            # Limita o valor do lucro para evitar overflow
            profit = np.clip(profit, -1e6, 1e6)
            self.balance += profit
            self.position = 0
            self.entry_price = 0
            reward = profit
            logging.info(f"[FECHAMENTO MANUAL] SHORT em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance:.2f}")
            event = 'close_short_manual'
            self.risk_manager.register_trade({
                'type': 'close_short_manual',
                'price': price,
                'profit': profit
            })

        # Checagem automática de stop loss/take profit
        if self.position != 0 and hasattr(self, 'stop_loss') and hasattr(self, 'take_profit'):
            if self._hit_stop_or_target(price):
                if self.position == 1:  # LONG
                    if price <= self.stop_loss:
                        # Stop Loss LONG
                        profit = (self.stop_loss - self.entry_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        reward = profit
                        logging.info(f"[STOP LOSS] LONG atingido em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance + profit:.2f}")
                        event = 'stop_loss_long'
                        self.risk_manager.register_trade({
                            'type': 'stop_loss_long',
                            'price': self.stop_loss,
                            'profit': profit
                        })
                    elif price >= self.take_profit:
                        # Take Profit LONG
                        profit = (self.take_profit - self.entry_price) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        reward = profit
                        logging.info(f"[TAKE PROFIT] LONG atingido em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance + profit:.2f}")
                        event = 'take_profit_long'
                        self.risk_manager.register_trade({
                            'type': 'take_profit_long',
                            'price': self.take_profit,
                            'profit': profit
                        })
                elif self.position == -1:  # SHORT
                    if price >= self.stop_loss:
                        # Stop Loss SHORT
                        profit = (self.entry_price - self.stop_loss) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        reward = profit
                        logging.info(f"[STOP LOSS] SHORT atingido em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance + profit:.2f}")
                        event = 'stop_loss_short'
                        self.risk_manager.register_trade({
                            'type': 'stop_loss_short',
                            'price': self.stop_loss,
                            'profit': profit
                        })
                    elif price <= self.take_profit:
                        # Take Profit SHORT
                        profit = (self.entry_price - self.take_profit) * position_size * (1 - self.fee)
                        # Limita o valor do lucro para evitar overflow
                        profit = np.clip(profit, -1e6, 1e6)
                        reward = profit
                        logging.info(f"[TAKE PROFIT] SHORT atingido em {price:.2f} | PnL: {profit:.2f} | Saldo: {self.balance + profit:.2f}")
                        event = 'take_profit_short'
                        self.risk_manager.register_trade({
                            'type': 'take_profit_short',
                            'price': self.take_profit,
                            'profit': profit
                        })
                self.balance += reward
                # Limita o valor do saldo para evitar overflow
                self.balance = np.clip(self.balance, 0, 1e9)
                self.position = 0
                self.entry_price = 0

        # Atualiza métricas de performance
        self.episode_profit = self.balance - self.initial_balance
        drawdown = min(0, self.episode_profit)
        self.max_drawdown = min(self.max_drawdown, drawdown)

        # Métricas de risco
        sharpe_ratio = self.episode_profit / (np.std([self.balance, self.initial_balance]) + 1e-8)
        calmar_ratio = self.episode_profit / (abs(self.max_drawdown) + 1e-8) if self.max_drawdown < 0 else 0
        
        # Penalidade por drawdown severo (adiciona pressão para controle de risco)
        if self.max_drawdown < -self.initial_balance * 0.1:  # Drawdown > 10%
            reward = reward * 0.8  # Reduz recompensa

        # Atualiza estado
        self.current_step += 1
        if self.current_step >= len(self.df) or self.balance <= 0:
            self.done = True
            terminated = True

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "episode_profit": self.episode_profit,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "event": event,
            "position_size": position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_per_trade": self.risk_manager.params.risk_per_trade,
            "rr_ratio": self.risk_manager.params.rr_ratio,
            "trade_count": self.trade_count
        }
        return obs, reward, terminated, truncated, info

    def _calculate_risk_levels(self, price, entry_type=None):
        """
        Calcula stop loss e take profit para uma entrada baseada no preço atual e tipo de entrada.
        
        Args:
            price: Preço de entrada
            entry_type: 'long' ou 'short'
            
        Returns:
            Tuple (stop_loss, take_profit)
        """
        # Determina o tipo de posição se não for fornecido
        if entry_type is None and hasattr(self, 'position'):
            entry_type = 'long' if self.position == 1 else 'short'
        elif entry_type is None:
            entry_type = 'long'  # Valor padrão
            
        stop_loss = 0.0
        take_profit = 0.0
            
        # Garante que self.strategy existe e é do tipo esperado
        if hasattr(self, 'strategy') and self.strategy is not None and getattr(self.strategy, '__class__', None) and self.strategy.__class__.__name__ == 'TrendFollowingStrategy':
            if entry_type == 'long':  # Long
                # Para long, stop loss abaixo do preço
                stop_loss = self._find_previous_ll(price)  # Lower Low para long
                # Limita o stop loss para valores razoáveis
                stop_loss = max(stop_loss, price * 0.7)  # No máximo 30% abaixo do preço
                
                # Calcula take profit com base no stop loss
                tp_distance = (price - stop_loss) * self.risk_manager.params.rr_ratio
                take_profit = price + tp_distance
                # Limita o take profit para valores razoáveis
                take_profit = min(take_profit, price * 2.0)  # No máximo o dobro do preço
            else:  # Short
                # Para short, stop loss acima do preço
                stop_loss = self._find_previous_hh(price)  # Higher High para short
                # Limita o stop loss para valores razoáveis
                stop_loss = min(stop_loss, price * 1.3)  # No máximo 30% acima do preço
                
                # Calcula take profit com base no stop loss
                tp_distance = (stop_loss - price) * self.risk_manager.params.rr_ratio
                take_profit = price - tp_distance
                # Limita o take profit para valores razoáveis
                take_profit = max(take_profit, price * 0.5)  # No mínimo metade do preço
        else:
            # Lógica padrão para outras estratégias
            if entry_type == 'long':  # Long
                stop_loss = price * 0.98  # Stop loss padrão 2% abaixo
                take_profit = price * (1 + self.risk_manager.params.risk_per_trade * self.risk_manager.params.rr_ratio)
                # Limita o take profit para valores razoáveis
                take_profit = min(take_profit, price * 1.1)  # No máximo 10% acima do preço
            else:  # Short
                stop_loss = price * 1.02  # Stop loss padrão 2% acima
                take_profit = price * (1 - self.risk_manager.params.risk_per_trade * self.risk_manager.params.rr_ratio)
                # Limita o take profit para valores razoáveis
                take_profit = max(take_profit, price * 0.9)  # No mínimo 10% abaixo do preço
                
        return stop_loss, take_profit

    def _hit_stop_or_target(self, price):
        # Verifica se atingiu stop ou alvo
        return (
            (self.position == 1 and (price <= self.stop_loss or price >= self.take_profit)) or
            (self.position == -1 and (price >= self.stop_loss or price <= self.take_profit))
        )

    def render(self, mode="human"):
        logging.info(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position} | Profit: {self.episode_profit:.2f}")
        
    def _find_previous_hh(self, price):
        """
        Encontra o Higher High (HH) anterior para usar como stop loss
        """
        # Olha para trás 10 barras para encontrar um máximo local
        lookback = min(20, self.current_step)
        window = self.df.iloc[self.current_step - lookback:self.current_step]
        
        # Encontra o maior high no período
        highest_high = window['high'].max()
        
        # Se o preço atual estiver muito próximo do maior high, 
        # usamos um valor padrão de stop loss (2% abaixo)
        if abs(highest_high - price) / price < 0.01:  # menos de 1% de distância
            return price * 0.98
            
        return highest_high
        
    def _find_previous_ll(self, price):
        """
        Encontra o Lower Low (LL) anterior para usar como stop loss
        """
        # Olha para trás 10 barras para encontrar um mínimo local
        lookback = min(20, self.current_step)
        window = self.df.iloc[self.current_step - lookback:self.current_step]
        
        # Encontra o menor low no período
        lowest_low = window['low'].min()
        
        # Se o preço atual estiver muito próximo do menor low, 
        # usamos um valor padrão de stop loss (2% acima)
        if abs(lowest_low - price) / price < 0.01:  # menos de 1% de distância
            return price * 1.02
            
        return lowest_low
