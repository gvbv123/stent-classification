import os
import yaml

def load_config(config_paths):
    """
    加载配置（支持多个yaml，后者覆盖前者）
    Args:
        config_paths: list[str] or str
    Returns:
        dict 配置
    """
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    cfg = {}
    for path in config_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg_part = yaml.safe_load(f)
        cfg = _deep_update(cfg, cfg_part)
    return cfg


def _deep_update(base, updates):
    """
    递归地更新字典，后者覆盖前者
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base
