import os
import yaml

def load_config(config_paths):
  
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    cfg = {}
    for path in config_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file does not exist: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg_part = yaml.safe_load(f)
        cfg = _deep_update(cfg, cfg_part)
    return cfg


def _deep_update(base, updates):
 
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base
