import json
import yaml

def load_config(path):
    """Carrega configuração de arquivo .json ou .yaml/.yml"""
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError('Formato de configuração não suportado: ' + path)
