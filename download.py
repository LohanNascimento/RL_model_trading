import ccxt
import pandas as pd
import time
import os
import logging
import argparse
import hashlib
import json
from datetime import datetime, timedelta
from utils.data_validator import DataValidator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Configura os argumentos de linha de comando"""
    parser = argparse.ArgumentParser(description='Download de dados OHLCV')
    parser.add_argument('--asset', default='AVAX', help='Ativo (ex: BTC, ETH, AVAX)')
    parser.add_argument('--quote', default='USDT', help='Moeda de cotação (ex: USDT, BUSD)')
    parser.add_argument('--exchange', default='binance', help='Exchange (ex: binance, ftx)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (ex: 1m, 5m, 1h, 1d)')
    parser.add_argument('--start_date', default='2022-01-01', help='Data inicial (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=None, help='Data final (YYYY-MM-DD)')
    parser.add_argument('--output_dir', default='data', help='Diretório de saída')
    parser.add_argument('--validate', action='store_true', help='Validar dados após download')
    return parser.parse_args()

def fetch_ohlcv(exchange_id, symbol, timeframe, since, limit=1000, retry_count=3):
    """
    Baixa dados OHLCV com tratamento de erros e retry
    
    Args:
        exchange_id: ID da exchange (ex: 'binance')
        symbol: Par de trading (ex: 'BTC/USDT')
        timeframe: Intervalo de tempo (ex: '1h')
        since: Timestamp inicial em milissegundos
        limit: Número máximo de candles por requisição
        retry_count: Número de tentativas em caso de erro
        
    Returns:
        list: Dados OHLCV
    """
    for attempt in range(retry_count):
        try:
            exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            return ohlcv
        except ccxt.NetworkError as e:
            logger.warning(f"Erro de rede ({attempt+1}/{retry_count}): {str(e)}")
            if attempt == retry_count - 1:
                logger.error(f"Falha após {retry_count} tentativas")
                raise
            time.sleep(5 * (attempt + 1))  # Backoff exponencial
        except ccxt.ExchangeError as e:
            logger.error(f"Erro da exchange: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Erro desconhecido: {str(e)}")
            raise

def fetch_all_ohlcv(exchange_id, symbol, timeframe, start_date, end_date=None, batch_size=1000):
    """
    Baixa todos os dados OHLCV em um intervalo de tempo
    
    Args:
        exchange_id: ID da exchange
        symbol: Par de trading
        timeframe: Intervalo de tempo
        start_date: Data inicial no formato 'YYYY-MM-DD'
        end_date: Data final (opcional)
        batch_size: Tamanho do lote
        
    Returns:
        DataFrame: Dados OHLCV
    """
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })
    
    # Converte datas para timestamp
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    
    if end_date:
        until = exchange.parse8601(f"{end_date}T23:59:59Z")
    else:
        until = int(datetime.now().timestamp() * 1000)
    
    logger.info(f"Baixando dados de {symbol} ({timeframe}) de {start_date} até {end_date or 'agora'}")
    
    all_ohlcv = []
    current_since = since
    
    # Calcula tamanho em dias para barra de progresso
    total_days = (datetime.fromtimestamp(until/1000) - datetime.fromtimestamp(since/1000)).days
    progress_step = max(1, total_days // 10)
    last_progress = 0
    
    while current_since < until:
        try:
            ohlcv = fetch_ohlcv(exchange_id, symbol, timeframe, current_since, batch_size)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"Sem dados para {symbol} após {datetime.fromtimestamp(current_since/1000)}")
                break
                
            # Adiciona ao resultado
            all_ohlcv.extend(ohlcv)
            
            # Atualiza timestamp para próximo lote
            current_since = ohlcv[-1][0] + 1
            
            # Mostra progresso a cada 10%
            current_days = (datetime.fromtimestamp(current_since/1000) - datetime.fromtimestamp(since/1000)).days
            if current_days - last_progress >= progress_step:
                progress_pct = min(100, int((current_days / total_days) * 100))
                logger.info(f"Progresso: {progress_pct}% ({current_days}/{total_days} dias)")
                last_progress = current_days
                
            # Previne rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logger.error(f"Erro ao baixar dados: {str(e)}")
            # Se temos dados parciais, continuamos com o que temos
            if all_ohlcv:
                logger.warning(f"Continuando com {len(all_ohlcv)} candles obtidos antes do erro")
                break
            else:
                raise
    
    # Converte para DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicatas
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp')
    
    logger.info(f"Download concluído: {len(df)} candles baixados")
    return df

def calculate_data_hash(df):
    """Calcula hash MD5 dos dados para verificação de integridade"""
    data_json = df.head(100).to_json()
    return hashlib.md5(data_json.encode()).hexdigest()

def save_with_metadata(df, filepath, metadata=None):
    """Salva DataFrame com metadados para rastreabilidade"""
    if metadata is None:
        metadata = {}
        
    # Adiciona metadados básicos
    metadata.update({
        'data_hash': calculate_data_hash(df),
        'rows': len(df),
        'date_range': [df['timestamp'].min().strftime('%Y-%m-%d'), 
                       df['timestamp'].max().strftime('%Y-%m-%d')],
        'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Salva metadados
    metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salva dados
    df.to_csv(filepath, index=False)
    logger.info(f"Dados salvos em {filepath}")
    logger.info(f"Metadados salvos em {metadata_path}")

def main():
    args = setup_argparse()
    
    # Configura parâmetros
    asset = args.asset
    quote = args.quote
    symbol = f'{asset}/{quote}'
    timeframe = args.timeframe
    exchange_id = args.exchange
    
    # Configura saída
    csv_filename = f'{asset}{quote}_{timeframe}.csv'
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, csv_filename)
    
    try:
        # Baixa dados
        df = fetch_all_ohlcv(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Validação de dados
        if args.validate:
            logger.info("Validando dados baixados...")
            validator = DataValidator()
            df, validation_result = validator.validate_price_data(df)
            
            if not validation_result['valid']:
                logger.warning(f"Validação detectou {len(validation_result['issues'])} problemas:")
                for issue in validation_result['issues']:
                    logger.warning(f"- {issue}")
                    
                if validation_result['rows_before'] != validation_result['rows_after']:
                    logger.warning(f"Removidas {validation_result['rows_before'] - validation_result['rows_after']} linhas com problemas")
            else:
                logger.info("Validação concluída: dados OK")
        
        # Salva com metadados
        save_with_metadata(df, csv_path, {
            'asset': asset,
            'quote': quote,
            'exchange': exchange_id,
            'timeframe': timeframe,
            'start_date': args.start_date,
            'end_date': args.end_date or datetime.now().strftime('%Y-%m-%d'),
            'validated': args.validate
        })
        
        logger.info(f"Concluído! {len(df)} candles salvos em {csv_path}")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()
