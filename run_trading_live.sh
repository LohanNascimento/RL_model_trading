#!/bin/bash
# Script para executar o sistema de trading em modo produção
# Executa o modelo treinado na Binance Testnet com monitoramento contínuo

# Configura variáveis de ambiente (preencha com suas credenciais)
export BINANCE_TESTNET_KEY="sua_api_key_aqui"
export BINANCE_TESTNET_SECRET="sua_api_secret_aqui"

# Escolha o modo de posição (one-way ou hedge)
POSITION_MODE="one-way"  # Recomendado para iniciantes (one-way ou hedge)

# Cria diretório para logs se não existir
mkdir -p logs

# Data e hora atual para o arquivo de log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/trading_live_${TIMESTAMP}.log"

echo "Iniciando sistema de trading em $(date)"
echo "Logs serão salvos em: ${LOG_FILE}"

# Verifica o modo de posição atual
echo "Verificando modo de posição..."
python set_position_mode.py --mode $POSITION_MODE --test

# Se quiser configurar o modo, descomente a linha abaixo
# python set_position_mode.py --mode $POSITION_MODE

# Verifica saldo e posições
echo "Verificando conta e posições atuais..."
python run_testnet.py --check_balance

# Executa o sistema de trading
# Opções recomendadas:
# --interval 3600: verifica a cada 1 hora (timeframe é 1h)
# --model_path checkpoints/best_model.zip: usa o melhor modelo treinado
# --symbols BTCUSDT ETHUSDT DOTUSDT: monitora apenas os ativos mais líquidos
# --log_level INFO: nível de log detalhado

echo "Iniciando monitoramento e trading..."
python run_testnet.py \
  --interval 3600 \
  --model_path checkpoints/best_model.zip \
  --symbols BTCUSDT ETHUSDT DOTUSDT ADAUSDT SOLUSDT \
  --log_level INFO \
  2>&1 | tee -a "${LOG_FILE}"

# Nota: Para um trading real, remova a flag --monitor_only abaixo
# python run_testnet.py \
#   --interval 3600 \
#   --model_path checkpoints/best_model.zip \
#   --symbols BTCUSDT ETHUSDT DOTUSDT ADAUSDT SOLUSDT \
#   --log_level INFO \
#   --monitor_only \
#   2>&1 | tee -a "${LOG_FILE}" 