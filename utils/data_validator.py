import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta

class DataValidator:
    """
    Classe responsável por validar e garantir a integridade dos dados de preço e indicadores.
    """
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.data_hash = None
        self.anomaly_detection_enabled = True
        
    def validate_price_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Valida DataFrame de preços verificando valores ausentes, anomalias e integridade.
        
        Args:
            df: DataFrame com dados de preço
            required_columns: Lista de colunas obrigatórias
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame limpo e dicionário com métricas de validação
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            
        result = {
            'valid': True,
            'issues': [],
            'rows_before': len(df),
            'rows_after': 0,
            'data_hash': '',
            'anomalies_detected': 0
        }
        
        # 1. Verificar colunas necessárias
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            result['valid'] = False
            result['issues'].append(f"Colunas ausentes: {missing_cols}")
            self.logger.error(f"Validação falhou: colunas ausentes {missing_cols}")
            return df, result
            
        # 2. Ordenar por timestamp se existir
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.sort_values('timestamp').reset_index(drop=True)
            else:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                except:
                    result['issues'].append("Não foi possível converter 'timestamp' para datetime")
                    
        # 3. Verificar valores ausentes
        na_counts = df[required_columns].isna().sum()
        has_na = na_counts.sum() > 0
        
        if has_na:
            self.logger.warning(f"Valores ausentes detectados: {na_counts[na_counts > 0].to_dict()}")
            result['issues'].append(f"Valores ausentes: {na_counts[na_counts > 0].to_dict()}")
            
            # Interpolação linear para preços
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in required_columns]
            df[price_cols] = df[price_cols].interpolate(method='linear', limit=3)
            
            # Verificar se ainda existem NaNs após interpolação
            remaining_na = df[required_columns].isna().sum().sum()
            if remaining_na > 0:
                self.logger.warning(f"Removendo {remaining_na} linhas com NaN restantes após interpolação")
                df = df.dropna(subset=required_columns)
                result['issues'].append(f"Removidas {remaining_na} linhas com valores ausentes não interpoláveis")
                
        # 4. Verificar valores negativos/zero em preços
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in required_columns]
        for col in price_cols:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                self.logger.warning(f"Valores inválidos (<=0) em {col}: {invalid_count}")
                result['issues'].append(f"Valores inválidos em {col}: {invalid_count}")
                df = df[df[col] > 0]
        
        # 5. Verificar inconsistências OHLC
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Verifica se high < low
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                self.logger.warning(f"Inconsistência OHLC: high < low em {invalid_hl} linhas")
                result['issues'].append(f"Inconsistência high < low: {invalid_hl} linhas")
                
                # Correção: troca high e low onde necessário
                invalid_idx = df[df['high'] < df['low']].index
                df.loc[invalid_idx, ['high', 'low']] = df.loc[invalid_idx, ['low', 'high']].values
                
            # Verifica se high < max(open, close) ou low > min(open, close)
            invalid_range = ((df['high'] < df[['open', 'close']].max(axis=1)) | 
                            (df['low'] > df[['open', 'close']].min(axis=1))).sum()
            if invalid_range > 0:
                self.logger.warning(f"Inconsistência OHLC: preço fora do range em {invalid_range} linhas")
                result['issues'].append(f"Preço fora do range OHLC: {invalid_range} linhas")
                
                # Correção: ajusta high/low quando necessário
                df['high'] = df[['high', 'open', 'close']].max(axis=1)
                df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # 6. Verificar gaps excessivos
        if 'timestamp' in df.columns and len(df) > 1:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp_diff'] = df['timestamp'].diff()
                # Identifica gaps maiores que 3 vezes o intervalo médio
                mean_interval = df['timestamp_diff'].median()
                large_gaps = df[df['timestamp_diff'] > mean_interval * 3].index.tolist()
                if large_gaps:
                    self.logger.warning(f"Detectados {len(large_gaps)} gaps temporais excessivos")
                    result['issues'].append(f"Gaps temporais: {len(large_gaps)} detectados")
                df = df.drop(columns=['timestamp_diff'])
        
        # 7. Detecção de anomalias (picos extremos)
        if self.anomaly_detection_enabled and len(df) > 20:
            for col in price_cols:
                # Calcula média móvel e desvio padrão
                rolling_mean = df[col].rolling(window=20, min_periods=1).mean()
                rolling_std = df[col].rolling(window=20, min_periods=1).std()
                
                # Detecta valores além de 5 desvios padrão da média
                threshold = 5
                anomalies = df[(df[col] > rolling_mean + threshold * rolling_std) | 
                              (df[col] < rolling_mean - threshold * rolling_std)].index
                
                if len(anomalies) > 0:
                    self.logger.warning(f"Detectadas {len(anomalies)} anomalias em {col}")
                    result['anomalies_detected'] += len(anomalies)
                    
                    # Suaviza anomalias substituindo por média móvel
                    df.loc[anomalies, col] = rolling_mean.loc[anomalies]
        
        # 8. Calcular hash dos dados para verificação de integridade
        data_json = df[required_columns].head(100).to_json()
        data_hash = hashlib.md5(data_json.encode()).hexdigest()
        self.data_hash = data_hash
        result['data_hash'] = data_hash
        
        # 9. Resultado final
        result['rows_after'] = len(df)
        result['valid'] = (len(result['issues']) == 0) and (result['anomalies_detected'] == 0)
        if not result['valid']:
            self.logger.warning(f"Validação completada com {len(result['issues'])} problemas e {result['anomalies_detected']} anomalias")
        else:
            self.logger.info(f"Validação completada com sucesso: {result['rows_after']} linhas válidas")
            
        return df, result
    
    def validate_indicators(self, df: pd.DataFrame, indicator_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Valida os indicadores técnicos no DataFrame, preenchendo valores ausentes.
        
        Args:
            df: DataFrame com indicadores
            indicator_columns: Lista de colunas de indicadores
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame com indicadores validados e estatísticas
        """
        result = {
            'valid': True,
            'issues': [],
            'fixed_indicators': 0
        }
        
        # Verifica existência das colunas
        missing_cols = [col for col in indicator_columns if col not in df.columns]
        if missing_cols:
            result['valid'] = False
            result['issues'].append(f"Indicadores ausentes: {missing_cols}")
            self.logger.error(f"Validação falhou: indicadores ausentes {missing_cols}")
            return df, result
            
        # Conta valores ausentes
        na_counts = df[indicator_columns].isna().sum()
        total_na = na_counts.sum()
        
        if total_na > 0:
            self.logger.warning(f"Valores ausentes em indicadores: {na_counts[na_counts > 0].to_dict()}")
            result['issues'].append(f"Valores ausentes: {na_counts[na_counts > 0].to_dict()}")
            result['fixed_indicators'] = total_na
            
            # Preenche NaNs com interpolação
            df[indicator_columns] = df[indicator_columns].interpolate(method='linear')
            
            # Preenche valores ainda ausentes nas extremidades
            df[indicator_columns] = df[indicator_columns].fillna(method='bfill').fillna(method='ffill')
            
            # Qualquer NaN restante vira 0
            df[indicator_columns] = df[indicator_columns].fillna(0)
            
        # Verifica valores infinitos
        inf_mask = np.isinf(df[indicator_columns])
        total_inf = inf_mask.sum().sum()
        
        if total_inf > 0:
            self.logger.warning(f"Valores infinitos em indicadores: {total_inf} detectados")
            result['issues'].append(f"Valores infinitos: {total_inf}")
            result['fixed_indicators'] += total_inf
            
            # Substitui infinitos pela mediana da coluna
            for col in indicator_columns:
                if inf_mask[col].any():
                    median_val = df.loc[~np.isinf(df[col]), col].median()
                    df.loc[np.isinf(df[col]), col] = median_val
        
        # Substituir outliers extremos em indicadores osciladores
        oscillator_cols = [col for col in indicator_columns if 'rsi' in col or 'stoch' in col]
        for col in oscillator_cols:
            if col in df.columns:
                # Limita osciladores a [0, 100]
                invalid_count = ((df[col] < 0) | (df[col] > 100)).sum()
                if invalid_count > 0:
                    self.logger.warning(f"Valores fora do range [0,100] em {col}: {invalid_count}")
                    result['issues'].append(f"Valores fora do range em {col}: {invalid_count}")
                    result['fixed_indicators'] += invalid_count
                    df[col] = df[col].clip(0, 100)
        
        return df, result
    
    def check_data_integrity(self, current_hash: str) -> bool:
        """
        Verifica se o hash dos dados atuais corresponde ao hash anterior.
        Útil para detectar manipulação de dados durante execução.
        
        Args:
            current_hash: Hash MD5 atual dos dados
            
        Returns:
            bool: True se a integridade for confirmada
        """
        if self.data_hash is None:
            # Primeiro uso, armazena o hash
            self.data_hash = current_hash
            return True
            
        # Verifica se o hash corresponde ao hash anteriormente armazenado
        is_valid = current_hash == self.data_hash
        if not is_valid:
            self.logger.error(f"Violação de integridade de dados detectada! Hash anterior: {self.data_hash}, Hash atual: {current_hash}")
            
        return is_valid