TREND_CONFIG = {
    'entry_rules': {
        'bos_lookback': 5,       # Velas para trás para detectar BOS
        'fvg_size': 0.0075,      # 0.75% mínimo para considerar FVG
        'max_retrace': 0.382     # Retração máxima permitida ao FVG
    },
    'exit_rules': {
        'risk_reward': 2.0,
        'trailing_stop': False
    }
}