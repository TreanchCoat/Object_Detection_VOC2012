#!/usr/bin/env python
"""Config-driven YOLO baseline training on PASCAL VOC 2012.

Usage:
    python prepare_voc.py --config configs/baseline_det.yaml     # once per config (det and seg use separate dataset dirs)
    python train.py --config configs/baseline_det.yaml
    python train.py --config configs/baseline_det.yaml --set train.epochs=1 --set train.fraction=0.02   # smoke test

Per run, <project>/<run_id>/ gets:
    epoch_metrics.csv   per-epoch train losses + val metrics (cleaned copy of Ultralytics' results.csv)
    run_metadata.csv    key,value: config, overrides, environment, dataset stats, timing, final/best metrics
    config_resolved.yaml  the exact config used (after --set overrides)
    + Ultralytics' own outputs (weights/, results.csv, args.yaml, plots)
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml

AUG_KEYS = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "erasing"]
# Which results.csv column defines the "best" epoch: boxes for detect, masks for segment
BEST_METRIC = {"detect": "metrics/mAP50-95(B)", "segment": "metrics/mAP50-95(M)"}


# ----------------------------------------------------------------------------- helpers
def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got: {item}")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def fmt(v) -> str:
    return json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)


def write_kv_csv(path: Path, meta: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in meta.items():
            w.writerow([k, fmt(v)])


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def environment_info() -> dict:
    import torch
    import ultralytics

    info = {
        "env.python": platform.python_version(),
        "env.platform": platform.platform(),
        "env.hostname": platform.node(),
        "env.torch": torch.__version__,
        "env.ultralytics": ultralytics.__version__,
        "env.cuda_available": torch.cuda.is_available(),
        "env.cuda_version": torch.version.cuda,
        "env.git_commit": sh(["git", "rev-parse", "HEAD"]) or "n/a",
        "env.git_dirty": bool(sh(["git", "status", "--porcelain"])),
    }
    if torch.cuda.is_available():
        info["env.gpu"] = torch.cuda.get_device_name(0)
        info["env.gpu_count"] = torch.cuda.device_count()
    return info


def check_train_args(train_args: dict, strict_no_aug: bool) -> None:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    unknown = sorted(k for k in train_args if k not in DEFAULT_CFG_DICT)
    if unknown:
        raise SystemExit(f"Unknown Ultralytics argument(s) in config: {unknown}")

    # Augmentation audit: any aug key missing from the config silently falls back to a non-zero default.
    active = {}
    for k in AUG_KEYS:
        if k in DEFAULT_CFG_DICT:
            val = train_args.get(k, DEFAULT_CFG_DICT[k])
            if val not in (0, 0.0, None, False):
                active[k] = val
    if active:
        msg = f"Augmentation is not fully disabled: {active}"
        if strict_no_aug:
            raise SystemExit(msg + "\nSet them to 0.0 in the config (or run.strict_no_aug: false).")
        print("WARNING:", msg)


def export_epoch_metrics(run_dir: Path) -> list[dict]:
    src = run_dir / "results.csv"
    if not src.exists():
        return []
    with open(src, newline="", encoding="utf-8") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if rows:
        with open(run_dir / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize(rows: list[dict], metric: str) -> dict:
    if not rows:
        return {}
    out = {"result.epochs_completed": len(rows)}
    out.update({f"final.{k}": v for k, v in rows[-1].items()})
    scored = [r for r in rows if r.get(metric)]
    if scored:
        best = max(scored, key=lambda r: float(r[metric]))
        out["result.best_metric"] = metric
        out.update({f"best.{k}": v for k, v in best.items()})
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_det.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value",
                    help="override a config value (repeatable)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    apply_overrides(cfg, args.set)

    run_cfg = cfg["run"]
    run_id = f"{run_cfg['name']}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project = Path(run_cfg.get("project", "runs")).expanduser().resolve()
    run_dir = project / run_id

    task = cfg["data"].get("task", "detect")
    if task not in BEST_METRIC:
        raise SystemExit(f"data.task must be one of {list(BEST_METRIC)}, got {task!r}")
    if (task == "segment") != ("seg" in Path(cfg["model"]["weights"]).name):
        raise SystemExit(f"data.task={task!r} does not match model.weights={cfg['model']['weights']!r} "
                         "(segment needs a *-seg model, detect needs a plain one)")

    yolo_root = Path(cfg["data"]["yolo_root"]).expanduser().resolve()
    dataset_yaml = yolo_root / "voc2012.yaml"
    if not dataset_yaml.exists():
        raise SystemExit(f"{dataset_yaml} not found - run prepare_voc.py first.")

    train_args = {**cfg["train"], **cfg.get("augmentation", {})}
    check_train_args(train_args, bool(run_cfg.get("strict_no_aug", True)))
    train_args.update(data=str(dataset_yaml), project=str(project), name=run_id,
                      exist_ok=True, seed=run_cfg.get("seed", 0))

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_dump(cfg, sort_keys=False)
    (run_dir / "config_resolved.yaml").write_text(resolved)

    meta = {
        "run_id": run_id,
        "status": "running",
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config_file": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(resolved.encode()).hexdigest(),
        "cli_overrides": args.set,
        **environment_info(),
        **{f"config.{k}": v for k, v in flatten(cfg).items()},
    }
    stats_file = yolo_root / "stats.json"
    if stats_file.exists():
        meta.update({f"dataset.{k}": v for k, v in flatten(json.loads(stats_file.read_text())).items()})
    write_kv_csv(run_dir / "run_metadata.csv", meta)

    from ultralytics import YOLO

    t0, status, error, model = time.time(), "finished", "", None
    try:
        model = YOLO(cfg["model"]["weights"])
        model.train(**train_args)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as e:
        status, error = "failed", repr(e)
        raise
    finally:
        meta["status"] = status
        meta["error"] = error
        meta["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        meta["duration_sec"] = round(time.time() - t0, 1)
        meta.update(summarize(export_epoch_metrics(run_dir), BEST_METRIC[task]))
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            meta.update({f"effective_args.{k}": v for k, v in vars(trainer.args).items()})
        write_kv_csv(run_dir / "run_metadata.csv", meta)
        print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":  # guard required for dataloader workers on Windows/macOS
    main()
