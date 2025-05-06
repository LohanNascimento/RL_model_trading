@echo off
REM Script para executar o sistema de trading em modo produção no Windows
REM Executa o modelo treinado na Binance Testnet com monitoramento contínuo

REM Configura variáveis de ambiente (preencha com suas credenciais)
set BINANCE_TESTNET_KEY=af48d3f1963b76c51f52066079b70af894b059d230ef2a850001f4f7e9431327
set BINANCE_TESTNET_SECRET=5169ec900dc7bb90ef5c28ab5db6ccd1eb1584080efca2ec4b41cd305b690a8b

REM Cria diretório para logs se não existir
if not exist logs mkdir logs

REM Data e hora atual para o arquivo de log
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"
set "LOG_FILE=logs\trading_live_%TIMESTAMP%.log"

echo Iniciando sistema de trading em %date% %time%
echo Logs serão salvos em: %LOG_FILE%

REM Executa o sistema de trading
REM Opções recomendadas:
REM --interval 3600: verifica a cada 1 hora (timeframe é 1h)
REM --model_path checkpoints/best_model.zip: usa o melhor modelo treinado
REM --symbols BTCUSDT ETHUSDT DOTUSDT: monitora apenas os ativos mais líquidos
REM --log_level INFO: nível de log detalhado

python run_testnet.py ^
  --interval 3600 ^
  --model_path checkpoints/best_model.zip ^
  --symbols BTCUSDT ETHUSDT DOTUSDT ADAUSDT SOLUSDT ^
  --log_level INFO

REM Para um trading real, remova a linha acima e descomente a linha abaixo (sem --monitor_only)
REM python run_testnet.py ^
REM   --interval 3600 ^
REM   --model_path checkpoints/best_model.zip ^
REM   --symbols BTCUSDT ETHUSDT DOTUSDT ADAUSDT SOLUSDT ^
REM   --log_level INFO ^
REM   --monitor_only 