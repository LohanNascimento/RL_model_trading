import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime


def generate_performance_report(trade_log_path, output_path_html='relatorio_desempenho.html', 
                              output_path_img='saldo_evolucao.png', output_path_dist='distribuicao_pnl.png'):
    """
    Gera um relatório HTML moderno e atraente de desempenho do modelo de trading baseado no trade_log.csv.
    Exibe métricas importantes, gráfico de saldo ao longo do tempo e distribuição de PNL.
    """
    # Carrega o trade log
    trades = pd.read_csv(trade_log_path)
    
    # Filtra apenas trades completos (com pnl e saldo)
    trades = trades.dropna(subset=['pnl', 'saldo'])
    
    # Converte a coluna datetime para formato datetime
    if 'datetime' in trades.columns:
        trades['datetime'] = pd.to_datetime(trades['datetime'])
    
    # Métricas básicas
    lucro_total = trades['pnl'].sum()
    n_trades = len(trades)
    n_ganhos = (trades['pnl'] > 0).sum()
    n_perdas = (trades['pnl'] < 0).sum()
    taxa_acerto = n_ganhos / n_trades if n_trades > 0 else 0
    pnl_medio = trades['pnl'].mean()
    pnl_medio_ganhos = trades.loc[trades['pnl'] > 0, 'pnl'].mean() if n_ganhos > 0 else 0
    pnl_medio_perdas = trades.loc[trades['pnl'] < 0, 'pnl'].mean() if n_perdas > 0 else 0
    max_drawdown = (trades['saldo'].cummax() - trades['saldo']).max()
    sharpe = trades['pnl'].mean() / (trades['pnl'].std() + 1e-8) * np.sqrt(len(trades))
    
    # Calcular mais métricas
    profit_factor = abs(trades.loc[trades['pnl'] > 0, 'pnl'].sum() / trades.loc[trades['pnl'] < 0, 'pnl'].sum()) if trades.loc[trades['pnl'] < 0, 'pnl'].sum() != 0 else float('inf')
    
    # Criar gráfico de saldo com estilo moderno
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    # Plotar saldo com linha mais grossa e cor atrativa
    plt.plot(trades['datetime'], trades['saldo'], linewidth=2, color='#3498db')
    
    # Adicionar linha de tendência
    z = np.polyfit(range(len(trades)), trades['saldo'], 1)
    p = np.poly1d(z)
    plt.plot(trades['datetime'], p(range(len(trades))), linestyle='--', color='#2ecc71', alpha=0.8)
    
    # Melhorar aparência do gráfico
    plt.grid(True, alpha=0.3)
    plt.xlabel('Data/Hora', fontsize=12)
    plt.ylabel('Saldo', fontsize=12)
    plt.title('Evolução do Saldo', fontsize=16, fontweight='bold')
    
    # Adicionar anotação do lucro total
    plt.annotate(f'Lucro Total: {lucro_total:.2f}', 
                xy=(0.02, 0.95), 
                xycoords='axes fraction', 
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path_img, dpi=300, bbox_inches='tight')
    
    # Gráfico de distribuição de PNL
    plt.figure(figsize=(10, 5))
    sns.histplot(trades['pnl'], kde=True, bins=20, color='#9b59b6')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('PNL por Trade', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição de PNL', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path_dist, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Data atual para o relatório
    report_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Definir cores para métricas baseadas em valores
    sharpe_color = '#27ae60' if sharpe > 1 else '#e74c3c'
    acerto_color = '#27ae60' if taxa_acerto > 0.5 else '#e74c3c'
    
    # Gera HTML moderno com Bootstrap
    with open(output_path_html, 'w', encoding='utf-8') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório de Desempenho do Modelo</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    color: #343a40;
                    padding-top: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #3498db, #9b59b6);
                    color: white;
                    padding: 30px 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .card {{
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    margin-bottom: 20px;
                    transition: transform 0.3s;
                }}
                .card:hover {{
                    transform: translateY(-5px);
                }}
                .card-header {{
                    font-weight: bold;
                    background-color: #f1f3f5;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                }}
                .metric-label {{
                    color: #6c757d;
                    font-size: 14px;
                }}
                .positive {{
                    color: #28a745;
                }}
                .negative {{
                    color: #dc3545;
                }}
                .image-container {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header text-center">
                    <h1>Relatório de Desempenho do Modelo</h1>
                    <p class="lead">Análise Completa de Trading</p>
                    <p class="small">Gerado em: {report_date}</p>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Resumo Financeiro</div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col-6 mb-3">
                                        <div class="metric-value {'positive' if lucro_total > 0 else 'negative'}">{lucro_total:.2f}</div>
                                        <div class="metric-label">Lucro Total</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-value {'positive' if pnl_medio > 0 else 'negative'}">{pnl_medio:.2f}</div>
                                        <div class="metric-label">PNL Médio</div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-value">{max_drawdown:.2f}</div>
                                        <div class="metric-label">Max Drawdown</div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                                        <div class="metric-label">Sharpe Ratio</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Estatísticas dos Trades</div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col-6 mb-3">
                                        <div class="metric-value">{n_trades}</div>
                                        <div class="metric-label">Total de Trades</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-value" style="color: {acerto_color}">{taxa_acerto:.2%}</div>
                                        <div class="metric-label">Taxa de Acerto</div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-value positive">{pnl_medio_ganhos:.2f}</div>
                                        <div class="metric-label">PNL Médio Ganhos</div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-value negative">{pnl_medio_perdas:.2f}</div>
                                        <div class="metric-label">PNL Médio Perdas</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Detalhes dos Trades</div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <div class="d-flex justify-content-between">
                                                <span>Trades Ganhos:</span>
                                                <span class="fw-bold">{n_ganhos}</span>
                                            </div>
                                            <div class="progress">
                                                <div class="progress-bar bg-success" role="progressbar" style="width: {taxa_acerto*100}%" 
                                                     aria-valuenow="{taxa_acerto*100}" aria-valuemin="0" aria-valuemax="100"></div>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <div class="d-flex justify-content-between">
                                                <span>Trades Perdidos:</span>
                                                <span class="fw-bold">{n_perdas}</span>
                                            </div>
                                            <div class="progress">
                                                <div class="progress-bar bg-danger" role="progressbar" style="width: {(1-taxa_acerto)*100}%" 
                                                     aria-valuenow="{(1-taxa_acerto)*100}" aria-valuemin="0" aria-valuemax="100"></div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-2">
                                            <div class="d-flex justify-content-between">
                                                <span>Profit Factor:</span>
                                                <span class="fw-bold {'positive' if profit_factor > 1 else 'negative'}">{profit_factor:.2f}</span>
                                            </div>
                                        </div>
                                        <div class="mb-2">
                                            <div class="d-flex justify-content-between">
                                                <span>Razão Ganho/Perda:</span>
                                                <span class="fw-bold">{abs(pnl_medio_ganhos/pnl_medio_perdas):.2f}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Evolução do Saldo</div>
                            <div class="card-body">
                                <div class="image-container text-center">
                                    <img src="{output_path_img}" class="img-fluid" alt="Evolução do Saldo">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Distribuição de PNL</div>
                            <div class="card-body">
                                <div class="image-container text-center">
                                    <img src="{output_path_dist}" class="img-fluid" alt="Distribuição de PNL">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>© {datetime.now().year} - Relatório de Trading Automatizado</p>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """)
    print(f"Relatório moderno salvo em {output_path_html}")


if __name__ == "__main__":
    # Exemplo de uso
    generate_performance_report('trade_log.csv')