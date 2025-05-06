import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from functools import wraps
import time
import traceback
from typing import Dict, List, Any, Callable, Optional

class PerformanceMonitor:
    """
    Monitor de desempenho para o sistema de trading.
    Registra métricas de desempenho, executa alertas e gera relatórios.
    """
    
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configura logging
        self.logger = logging.getLogger('trading_monitor')
        self.logger.setLevel(log_level)
        
        # Adiciona handlers se não existirem
        if not self.logger.handlers:
            # Log para arquivo
            log_file = os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
            
            # Log para console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(console_handler)
        
        # Estatísticas e métricas
        self.trade_history = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0
        }
        
        # Métricas de desempenho do sistema
        self.system_metrics = {
            'execution_times': [],  # Tempos de execução 
            'memory_usage': [],     # Uso de memória
            'errors': [],           # Registro de erros
            'warnings': []          # Registro de avisos
        }
        
        # Limites para alertas
        self.alert_thresholds = {
            'max_drawdown': 0.15,  # Alerta se drawdown > 15%
            'win_rate_min': 0.40,  # Alerta se win rate < 40%
            'consecutive_losses': 5,  # Alerta após 5 perdas consecutivas
            'execution_time_max': 5.0  # Alerta se tempo de execução > 5s
        }
        
        self.logger.info(f"Monitor de desempenho inicializado. Logs em: {log_dir}")
    
    def register_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Registra um trade e atualiza métricas
        
        Args:
            trade_data: Dicionário com dados do trade
                Deve conter: timestamp, type, price, position, profit_loss
        """
        # Adiciona timestamp se não existir
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        # Registra o trade
        self.trade_history.append(trade_data)
        self.logger.info(f"Trade registrado: {trade_data['type']} | PnL: {trade_data.get('profit_loss', 0):.2f}")
        
        # Atualiza métricas
        self._update_metrics()
        
        # Verifica alertas
        self._check_alerts(trade_data)
    
    def _update_metrics(self) -> None:
        """Atualiza todas as métricas de desempenho"""
        if not self.trade_history:
            return
            
        # Converte para DataFrame para análise
        df = pd.DataFrame(self.trade_history)
        
        # Filtra apenas trades fechados com PnL
        closed_trades = df[df['type'].str.contains('exit|close')]
        
        if len(closed_trades) == 0:
            return
            
        # Métricas básicas
        self.performance_metrics['total_trades'] = len(closed_trades)
        self.performance_metrics['win_trades'] = (closed_trades['profit_loss'] > 0).sum()
        self.performance_metrics['loss_trades'] = (closed_trades['profit_loss'] <= 0).sum()
        
        # Win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = self.performance_metrics['win_trades'] / self.performance_metrics['total_trades']
        
        # Médias de ganhos e perdas
        if self.performance_metrics['win_trades'] > 0:
            self.performance_metrics['avg_win'] = closed_trades[closed_trades['profit_loss'] > 0]['profit_loss'].mean()
        
        if self.performance_metrics['loss_trades'] > 0:
            self.performance_metrics['avg_loss'] = closed_trades[closed_trades['profit_loss'] <= 0]['profit_loss'].mean()
        
        # Profit factor
        total_gains = closed_trades[closed_trades['profit_loss'] > 0]['profit_loss'].sum()
        total_losses = abs(closed_trades[closed_trades['profit_loss'] <= 0]['profit_loss'].sum())
        
        if total_losses > 0:
            self.performance_metrics['profit_factor'] = total_gains / total_losses
        else:
            self.performance_metrics['profit_factor'] = float('inf') if total_gains > 0 else 0.0
        
        # Calcular drawdown máximo
        if 'balance' in closed_trades.columns:
            balance_series = closed_trades['balance']
            peak = balance_series.expanding().max()
            drawdown = (balance_series - peak) / peak
            self.performance_metrics['max_drawdown'] = abs(drawdown.min())
        
        # Sharpe Ratio (simplificado)
        if 'profit_loss' in closed_trades.columns and len(closed_trades) > 1:
            returns = closed_trades['profit_loss']
            if returns.std() > 0:
                self.performance_metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)  # Anualizado
    
    def _check_alerts(self, trade_data: Dict[str, Any]) -> None:
        """Verifica condições para disparar alertas"""
        # Alerta de drawdown
        if self.performance_metrics['max_drawdown'] > self.alert_thresholds['max_drawdown']:
            self.logger.warning(f"ALERTA: Drawdown máximo excedido: {self.performance_metrics['max_drawdown']:.2%}")
        
        # Alerta de win rate
        if (self.performance_metrics['total_trades'] >= 10 and 
                self.performance_metrics['win_rate'] < self.alert_thresholds['win_rate_min']):
            self.logger.warning(f"ALERTA: Win rate abaixo do mínimo: {self.performance_metrics['win_rate']:.2%}")
        
        # Alerta de perdas consecutivas
        recent_trades = [t for t in self.trade_history if t['type'].startswith('close') or t['type'].startswith('exit')]
        recent_trades = recent_trades[-self.alert_thresholds['consecutive_losses']:]
        
        if (len(recent_trades) >= self.alert_thresholds['consecutive_losses'] and 
                all(t.get('profit_loss', 0) <= 0 for t in recent_trades)):
            self.logger.warning(
                f"ALERTA: {len(recent_trades)} perdas consecutivas detectadas")

    def track_execution_time(self, func: Callable) -> Callable:
        """
        Decorator para monitorar tempo de execução de funções
        
        Args:
            func: Função a ser monitorada
            
        Returns:
            Callable: Função decorada
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Registra tempo de execução
                self.system_metrics['execution_times'].append({
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'execution_time': execution_time
                })
                
                # Alerta se tempo for excessivo
                if execution_time > self.alert_thresholds['execution_time_max']:
                    self.logger.warning(
                        f"ALERTA: Tempo de execução excessivo em {func.__name__}: {execution_time:.2f}s")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Registra erro
                error_info = {
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'execution_time': execution_time
                }
                self.system_metrics['errors'].append(error_info)
                
                self.logger.error(
                    f"Erro em {func.__name__}: {str(e)} (tempo: {execution_time:.2f}s)")
                
                # Re-raise para tratamento adequado
                raise
        
        return wrapper
    
    def log_system_warning(self, message: str, details: Optional[Dict] = None) -> None:
        """Registra um aviso do sistema"""
        warning_info = {
            'timestamp': datetime.now(),
            'message': message,
            'details': details or {}
        }
        self.system_metrics['warnings'].append(warning_info)
        self.logger.warning(f"Aviso do sistema: {message}")
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera relatório de desempenho
        
        Args:
            output_path: Caminho para salvar o relatório (opcional)
            
        Returns:
            Dict: Métricas e estatísticas
        """
        # Atualiza métricas
        self._update_metrics()
        
        # Prepara relatório
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.performance_metrics,
            'system_health': {
                'error_count': len(self.system_metrics['errors']),
                'warning_count': len(self.system_metrics['warnings']),
                'avg_execution_time': np.mean([e['execution_time'] for e in self.system_metrics['execution_times']]) 
                                     if self.system_metrics['execution_times'] else 0.0
            },
            'trades': {
                'total': self.performance_metrics['total_trades'],
                'recent_trades': self.trade_history[-10:] if self.trade_history else []
            }
        }
        
        # Salvar relatório se caminho fornecido
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Relatório de desempenho salvo em {output_path}")
            
            # Gera gráficos básicos se tiver trades suficientes
            if len(self.trade_history) >= 5:
                self._generate_performance_charts(os.path.dirname(output_path))
        
        return report
    
    def _generate_performance_charts(self, output_dir: str) -> None:
        """Gera gráficos de desempenho básicos"""
        df = pd.DataFrame(self.trade_history)
        
        if len(df) < 5 or 'balance' not in df.columns:
            return
            
        # Gráfico de evolução do saldo
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(df)), df['balance'])
        plt.title('Evolução do Saldo')
        plt.xlabel('Trades')
        plt.ylabel('Saldo')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'saldo_evolucao.png'))
        plt.close()
        
        # Distribuição de PnL
        if 'profit_loss' in df.columns:
            plt.figure(figsize=(8, 4))
            plt.hist(df['profit_loss'], bins=20)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Distribuição de PnL')
            plt.xlabel('Profit/Loss')
            plt.ylabel('Frequência')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'distribuicao_pnl.png'))
            plt.close()
            
            # Retornos acumulados
            plt.figure(figsize=(10, 5))
            cumulative_returns = (df['profit_loss'].cumsum())
            plt.plot(range(len(df)), cumulative_returns)
            plt.title('Retornos Acumulados')
            plt.xlabel('Trades')
            plt.ylabel('PnL Acumulado')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'retornos_acumulados.png'))
            plt.close()
            
    def set_alert_threshold(self, alert_name: str, value: float) -> None:
        """Atualiza um limite para alertas"""
        if alert_name in self.alert_thresholds:
            self.alert_thresholds[alert_name] = value
            self.logger.info(f"Limite de alerta atualizado: {alert_name} = {value}")
        else:
            self.logger.warning(f"Alerta desconhecido: {alert_name}")

# Função de utilidade para criar um monitor global
_GLOBAL_MONITOR = None

def get_monitor(log_dir='logs', log_level=logging.INFO) -> PerformanceMonitor:
    """
    Retorna a instância global do monitor ou cria uma nova se não existir
    
    Args:
        log_dir: Diretório para logs
        log_level: Nível de logging
        
    Returns:
        PerformanceMonitor: Instância do monitor
    """
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = PerformanceMonitor(log_dir, log_level)
    return _GLOBAL_MONITOR 