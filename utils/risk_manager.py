import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta

@dataclass
class RiskParameters:
    risk_per_trade: float = 0.02  # 2% do saldo por trade
    rr_ratio: float = 2.0        # Risk/Reward ratio
    max_position_size: float = 0.1  # Máximo 10% do saldo
    stop_loss_buffer: float = 0.001  # Buffer para stop loss (0.1%)
    take_profit_buffer: float = 0.001  # Buffer para take profit (0.1%)
    max_trades_per_day: int = 1000  # Limite máximo de operações por dia (aumentado)
    max_drawdown_percent: float = 0.15  # Limite máximo de drawdown (15%)
    enforce_trade_limit: bool = True  # Controla se o limite de trades deve ser aplicado

class RiskManager:
    def __init__(self, params: Optional[RiskParameters] = None):
        self.params = params if params else RiskParameters()
        self.validate_parameters()
        self.logger = logging.getLogger(__name__)
        self.trade_history: List[Dict] = []
        self.initial_balance: float = 0
        self.last_balance_check: float = 0
        self.last_volatility_check: datetime = datetime.now()
        self.price_history: List[float] = []

    def validate_parameters(self):
        """Valida se os parâmetros estão dentro de limites aceitáveis."""
        if not 0 < self.params.risk_per_trade <= 0.1:
            raise ValueError("risk_per_trade deve estar entre 0 e 0.1 (10%)")
        if self.params.rr_ratio < 1:
            raise ValueError("rr_ratio deve ser maior ou igual a 1")
        if not 0 < self.params.max_position_size <= 0.2:
            raise ValueError("max_position_size deve estar entre 0 e 0.2 (20%)")
        if self.params.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day deve ser pelo menos 1")
        if not 0 < self.params.max_drawdown_percent < 1:
            raise ValueError("max_drawdown_percent deve estar entre 0 e 1 (100%)")

    def calculate_position_size(self, balance: float, price: float) -> float:
        """
        Calcula o tamanho da posição com base no risco e saldo.
        
        Args:
            balance: Saldo atual
            price: Preço de entrada
            
        Returns:
            Tamanho da posição em quantidade
        """
        # Registra o saldo inicial na primeira chamada
        if self.initial_balance == 0:
            self.initial_balance = balance
            self.last_balance_check = balance
            
        risk_amount = balance * self.params.risk_per_trade
        position_size = risk_amount / price
        
        # Aplica o limite máximo de posição
        max_position_value = balance * self.params.max_position_size
        position_size = min(position_size, max_position_value / price)
        
        return position_size

    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """
        Calcula o nível de stop loss.
        
        Args:
            entry_price: Preço de entrada
            position_type: 'long' ou 'short'
            
        Returns:
            Nível de stop loss
        """
        if position_type == 'long':
            stop_loss = entry_price * (1 - self.params.risk_per_trade - self.params.stop_loss_buffer)
        else:  # short
            stop_loss = entry_price * (1 + self.params.risk_per_trade + self.params.stop_loss_buffer)
        return stop_loss

    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """
        Calcula o nível de take profit.
        
        Args:
            entry_price: Preço de entrada
            position_type: 'long' ou 'short'
            
        Returns:
            Nível de take profit
        """
        if position_type == 'long':
            take_profit = entry_price * (1 + self.params.risk_per_trade * self.params.rr_ratio - self.params.take_profit_buffer)
        else:  # short
            take_profit = entry_price * (1 - self.params.risk_per_trade * self.params.rr_ratio + self.params.take_profit_buffer)
        return take_profit

    def log_risk_parameters(self):
        """Log dos parâmetros de risco para auditoria."""
        self.logger.info(f"Parâmetros de Risco:"
                         f"\n- Risco por trade: {self.params.risk_per_trade * 100:.2f}%"
                         f"\n- R/R Ratio: {self.params.rr_ratio}"
                         f"\n- Tamanho máximo de posição: {self.params.max_position_size * 100:.2f}%"
                         f"\n- Buffer SL: {self.params.stop_loss_buffer * 100:.2f}%"
                         f"\n- Buffer TP: {self.params.take_profit_buffer * 100:.2f}%"
                         f"\n- Máx trades por dia: {self.params.max_trades_per_day if self.params.enforce_trade_limit else 'Desativado'}"
                         f"\n- Limite de drawdown: {self.params.max_drawdown_percent * 100:.2f}%")

    def register_trade(self, trade_info: Dict):
        """Registra um trade para controle de limites"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'info': trade_info
        })
    
    def check_trade_limits(self, current_balance: float, current_price: float) -> Dict:
        """
        Verifica se os limites de risco foram atingidos
        
        Returns:
            Dict com {'allowed': bool, 'reason': str}
        """
        result = {'allowed': True, 'reason': ''}
        
        # Atualiza histórico de preços para referência futura
        self.price_history.append(current_price)
        if len(self.price_history) > 20:  # Mantém apenas os últimos 20 preços
            self.price_history.pop(0)
            
        # 1. Verifica limite de trades por dia (se ativado)
        if self.params.enforce_trade_limit:
            today_trades = [t for t in self.trade_history 
                            if t['timestamp'] > datetime.now() - timedelta(days=1)]
            if len(today_trades) >= self.params.max_trades_per_day:
                result['allowed'] = False
                result['reason'] = 'Limite diário de trades atingido'
                return result
            
        # 2. Verifica drawdown
        if self.initial_balance > 0:
            current_drawdown = (self.initial_balance - current_balance) / self.initial_balance
            if current_drawdown > self.params.max_drawdown_percent:
                result['allowed'] = False
                result['reason'] = f'Drawdown máximo atingido: {current_drawdown:.2%}'
                return result
                
        return result
