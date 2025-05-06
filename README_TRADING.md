# Sistema de Trading Automatizado com RL

Este sistema permite executar um modelo de trading baseado em Reinforcement Learning (RL) na Binance Futures Testnet, simulando operações reais sem arriscar capital.

## Visão Geral

O sistema foi projetado para monitorar múltiplos ativos de criptomoedas simultaneamente e tomar decisões de compra/venda baseadas no modelo treinado. Ele inclui:

- Monitoramento contínuo de múltiplos símbolos
- Gerenciamento de risco por operação
- Limites de operações por dia por símbolo
- Logs detalhados e relatórios de desempenho
- Modos de execução: monitoramento (sem operações) ou trading completo

## Pré-requisitos

- Python 3.8+
- Biblioteca CCXT instalada (`pip install ccxt`)
- Modelo treinado (normalmente em `checkpoints/best_model.zip`)
- Credenciais da Binance Testnet (opcional para testes)

## Configuração

1. Obtenha credenciais da Binance Testnet:
   - Visite https://testnet.binancefuture.com/
   - Registre-se e obtenha suas chaves de API e Secret

2. Configure as chaves:
   ```
   export BINANCE_TESTNET_KEY="sua_api_key"
   export BINANCE_TESTNET_SECRET="sua_api_secret"
   ```
   No Windows, use `set` em vez de `export`.

3. Verifique a configuração:
   ```
   python run_testnet.py --check_balance
   ```

## Modos de Execução

### Verificação de Saldo e Posições

```
python run_testnet.py --check_balance
```

### Teste de Trade (envia e fecha uma ordem de teste)

```
python run_testnet.py --test_trade --symbols BTCUSDT
```

### Modo de Monitoramento (sem executar trades)

```
python run_testnet.py --monitor_only --symbols BTCUSDT ETHUSDT DOTUSDT
```

### Execução Única (um ciclo apenas)

```
python run_testnet.py --run_once --symbols BTCUSDT ETHUSDT
```

### Execução Contínua (trading real)

```
python run_testnet.py --symbols BTCUSDT ETHUSDT DOTUSDT --interval 3600
```

## Parâmetros Importantes

- `--symbols`: Lista de símbolos a monitorar (ex: `BTCUSDT ETHUSDT`)
- `--interval`: Intervalo entre verificações em segundos (padrão: 60)
- `--model_path`: Caminho para o modelo treinado
- `--config`: Caminho para o arquivo de configuração do ambiente
- `--risk_config`: Caminho para o arquivo de configuração de risco
- `--log_level`: Nível de detalhamento dos logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--monitor_only`: Apenas monitorar sem executar trades
- `--run_once`: Executar apenas um ciclo e sair

## Scripts Automatizados

Para execução contínua, use os scripts incluídos:

- Linux/Mac: `./run_trading_live.sh`
- Windows: `run_trading_live.bat`

Estes scripts já vêm configurados com parâmetros otimizados.

## Gerenciamento de Risco

O sistema é configurado para:
- Limitar risco por operação a 2% (configurável)
- Aplicar relação risco:recompensa de 1:2 (configurável)
- Limitar a exposição por ativo a 10% do capital
- Permitir apenas 1 operação por dia por símbolo

## Logs e Monitoramento

Os logs são gravados no diretório `dryrun/logs/` com timestamps para facilitar análise posterior.

## Segurança

- **IMPORTANTE**: Teste sempre na Testnet antes de aplicar em capital real.
- Verifique a interação do sistema com a corretora antes de deixar em execução contínua.
- Valide o modelo em dados históricos recentes antes de colocá-lo em produção.
- Comece com um pequeno capital ao usar em ambiente real.

## Solução de Problemas

- **"Erro de timestamp"**: O sistema tenta sincronizar o timestamp com o servidor Binance. Se persistir, configure seu computador para sincronizar com servidores NTP.
- **"Erro de observação"**: Verifique se o formato das observações (features) corresponde ao que o modelo espera.
- **"Quantidade mínima"**: A Binance exige que ordens sejam de pelo menos 100 USDT. O sistema já ajusta isso automaticamente.
- **"Order's position side does not match user's setting"**: Este erro ocorre quando o modo de posição da conta está configurado incorretamente. Use o script `set_position_mode.py` para ajustar:
  ```
  python set_position_mode.py --mode one-way  # Para modo One-way
  python set_position_mode.py --mode hedge    # Para modo Hedge
  ```

## Modos de Posição na Binance

A Binance Futures Testnet suporta dois modos de operação:

1. **Modo One-way (Único)**: 
   - Você só pode ter uma posição por símbolo
   - Mudar de direção (long para short) requer fechar a posição atual primeiro
   - Mais simples e recomendado para iniciantes

2. **Modo Hedge (Dual)**:
   - Permite posições long e short simultaneamente no mesmo símbolo
   - Requer especificação explícita de `positionSide` (LONG ou SHORT) nas ordens
   - Mais flexível, mas também mais complexo

Para configurar o modo de posição:
```
python set_position_mode.py --mode one-way  # ou 'hedge'
```

Para apenas verificar a configuração atual:
```
python set_position_mode.py --mode one-way --test
```

O sistema foi adaptado para funcionar com qualquer um dos modos, detectando automaticamente a configuração da sua conta.

## Limitações

- Trading em timeframe de 1h (não otimizado para outros timeframes)
- API da Binance pode limitar o número de requisições por minuto
- Modo Testnet pode ter diferenças de liquidez em relação à conta real

## Dicas de Uso

- Sempre comece com `--monitor_only` para verificar o comportamento do modelo
- Monitore os logs regularmente para identificar padrões ou problemas
- Considere ajustar os parâmetros de risco conforme o desempenho observado 