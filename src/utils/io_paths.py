import os
import datetime

def make_output_dirs(cfg):
    """
    根据配置创建带时间戳的输出目录
    返回字典：{runs, outputs, figures}
    """
    root = cfg["OUTPUT"]["ROOT"]
    run_name = cfg["OUTPUT"]["RUN_NAME"]
    use_ts = cfg["OUTPUT"].get("USE_TIMESTAMP", True)

    if use_ts:
        fmt = cfg["OUTPUT"].get("TIMESTAMP_FMT", "%Y%m%d-%H%M%S")
        ts = datetime.datetime.now().strftime(fmt)
        run_dir = os.path.join(root, run_name, ts)
    else:
        run_dir = os.path.join(root, run_name)

    dirs = {
        "root": run_dir,
        "runs": os.path.join(run_dir, "runs"),
        "outputs": os.path.join(run_dir, "outputs"),
        "figures": os.path.join(run_dir, "figures"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs
