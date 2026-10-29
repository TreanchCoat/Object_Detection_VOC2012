#!/usr/bin/env python
"""Evaluate every detection/segmentation weight (.pt or Ultralytics-exported .onnx) in a folder on VOC2007.

    python evaluate.py --weights-dir weights --voc2007 S:/Datasets/VOCdevkit

Detection weights ("detect" task) are evaluated on every annotated image in VOC2007
(Annotations/*.xml with a matching JPEG), converted once into <data-root> (default datasets/voc2007_eval).

Segmentation weights ("segment" task) are evaluated on the VOC2007 segmentation subset only (the images that
have SegmentationObject/ + SegmentationClass/ masks, ~200 images). The polygon labels are built lazily, the
first time a segmentation weight is found. Box and mask metrics are both reported.
NOTE: seg results (small subset, mask-derived boxes) are NOT directly comparable with detection results.

For each weight file in --weights-dir, results go to <out>/<name>/ (for .onnx files: <out>/<name>_onnx/) :
    <name>_metrics.csv     key,value: overall metrics, speed, parameters used, dataset stats, versions, file hash
    <name>_per_class.csv   per-class instances, precision, recall, mAP50, mAP50-95 (+ mask_* columns for seg models)
    + Ultralytics plots (PR/F1 curves, confusion matrices, example predictions)
and <out>/summary.csv has one row per weight file (rebuilt from all the *_metrics.csv files).

Notes
  - imgsz defaults to "auto": the image size stored in each checkpoint at training time.
  - Only detect/segment models with the 20 VOC class names are evaluated; others are recorded as "skipped".
  - .onnx files must be exported by Ultralytics (task/names/imgsz are read from the embedded metadata). A static-shape
    ONNX is always evaluated at its export size and batch 1. Use --no-rect to evaluate .pt and .onnx identically.
  - Difficult objects are dropped by default for the box dataset, matching how the training data was converted.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from prepare_voc import VOC_CLASSES, convert_annotation, link_or_copy

SUMMARY_KEYS = ["status", "reason", "model.format", "model.task", "eval.imgsz", "metric.mAP50-95", "metric.mAP50",
                "metric.precision", "metric.recall", "metric.mask.mAP50-95", "metric.mask.mAP50",
                "metric.mask.precision", "metric.mask.recall", "model.params", "speed.inference_ms",
                "eval.dataset", "eval.n_images", "weights_path"]


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- dataset
def find_voc2007(root: Path) -> Path:
    for cand in (root / "VOC2007", root / "VOCdevkit" / "VOC2007", root):
        if (cand / "JPEGImages").is_dir() and (cand / "Annotations").is_dir():
            return cand
    raise SystemExit(f"Could not find VOC2007 (JPEGImages/ + Annotations/) under {root}")


def build_dataset(voc: Path, data_root: Path, keep_difficult: bool, force_copy: bool, rebuild: bool):
    marker = data_root / "stats.json"
    if data_root.exists() and any(data_root.iterdir()):
        if not marker.exists():
            raise SystemExit(f"{data_root} exists and was not created by evaluate.py - choose another --data-root")
        old = json.loads(marker.read_text())
        if old.get("voc_dir") == str(voc) and old.get("keep_difficult") == keep_difficult and not rebuild:
            print(f"Reusing converted VOC2007 at {data_root} ({old['images']} images)")
            return data_root / "voc2007_eval.yaml", old
        for sub in ("images", "labels"):  # derived data only; never touches the original VOC files
            shutil.rmtree(data_root / sub, ignore_errors=True)

    ids = sorted(p.stem for p in (voc / "Annotations").glob("*.xml") if (voc / "JPEGImages" / f"{p.stem}.jpg").exists())
    if not ids:
        raise SystemExit(f"No annotated images found in {voc}")
    img_dir, lbl_dir = data_root / "images" / "all", data_root / "labels" / "all"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    per_class, n_lines, n_empty, sk_diff, sk_deg = Counter(), 0, 0, 0, 0
    for i in ids:
        src = voc / "JPEGImages" / f"{i}.jpg"
        lines, pc, a, b = convert_annotation(voc / "Annotations" / f"{i}.xml", src, keep_difficult)
        (lbl_dir / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        link_or_copy(src, img_dir / f"{i}.jpg", force_copy)
        per_class.update(pc)
        n_lines += len(lines)
        n_empty += not lines
        sk_diff += a
        sk_deg += b

    stats = {"voc_dir": str(voc), "keep_difficult": keep_difficult, "images": len(ids), "label_lines": n_lines,
             "images_without_objects": n_empty, "skipped_difficult": sk_diff, "skipped_degenerate": sk_deg,
             "lines_per_class": dict(per_class)}
    ds = {"path": str(data_root.resolve()), "train": "images/all", "val": "images/all",
          "names": {i: c for i, c in enumerate(VOC_CLASSES)}}
    (data_root / "voc2007_eval.yaml").write_text(yaml.safe_dump(ds, sort_keys=False))
    marker.write_text(json.dumps(stats, indent=2))
    print(f"Converted VOC2007: {len(ids)} images, {n_lines} objects "
          f"(skipped {sk_diff} difficult, {sk_deg} degenerate)")
    return data_root / "voc2007_eval.yaml", stats


def build_seg_dataset(voc: Path, data_root: Path, force_copy: bool, rebuild: bool):
    """Polygon labels from VOC2007 SegmentationObject/SegmentationClass (only a subset of images has them)."""
    import cv2
    import numpy as np
    from collections import Counter
    from PIL import Image

    marker = data_root / "seg_stats.json"
    img_dir, lbl_dir = data_root / "images" / "seg", data_root / "labels" / "seg"
    yaml_path = data_root / "voc2007_seg_eval.yaml"
    if marker.exists() and img_dir.is_dir() and lbl_dir.is_dir() and yaml_path.exists() and not rebuild:
        old = json.loads(marker.read_text())
        if old.get("voc_dir") == str(voc):
            print(f"Reusing converted VOC2007 seg subset ({old['images']} images)")
            return yaml_path, old
    for d in (img_dir, lbl_dir):
        shutil.rmtree(d, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in (voc / "SegmentationObject").glob("*.png")
                 if (voc / "SegmentationClass" / f"{p.stem}.png").exists()
                 and (voc / "JPEGImages" / f"{p.stem}.jpg").exists())
    if not ids:
        raise SystemExit(f"No SegmentationObject/SegmentationClass masks found in {voc}")

    per_class, n_lines = Counter(), 0
    for i in ids:
        obj = np.array(Image.open(voc / "SegmentationObject" / f"{i}.png"))
        cls = np.array(Image.open(voc / "SegmentationClass" / f"{i}.png"))
        h, w = obj.shape
        lines = []
        for inst in np.unique(obj):
            if inst in (0, 255):  # background / void border
                continue
            m = obj == inst
            vals = cls[m]
            vals = vals[(vals > 0) & (vals <= 20)]
            if vals.size == 0:
                continue
            cid = int(np.bincount(vals).argmax()) - 1
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
            if len(c) < 3:
                continue
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in c)
            lines.append(f"{cid} {coords}")
            per_class[VOC_CLASSES[cid]] += 1
        (lbl_dir / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        link_or_copy(voc / "JPEGImages" / f"{i}.jpg", img_dir / f"{i}.jpg", force_copy)
        n_lines += len(lines)

    stats = {"voc_dir": str(voc), "images": len(ids), "label_lines": n_lines, "lines_per_class": dict(per_class)}
    ds = {"path": str(data_root), "train": "images/seg", "val": "images/seg",
          "names": {i: c for i, c in enumerate(VOC_CLASSES)}}
    yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
    marker.write_text(json.dumps(stats, indent=2))
    print(f"Converted VOC2007 seg subset: {len(ids)} images, {n_lines} instances")
    return yaml_path, stats


# ----------------------------------------------------------------------------- csv helpers
def write_kv(path: Path, meta: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in meta.items():
            w.writerow([k, json.dumps(v, default=str) if isinstance(v, (dict, list)) else v])


def read_kv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["key"]: r["value"] for r in csv.DictReader(f)}


def rebuild_summary(out_root: Path) -> Path:
    rows = []
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        m = d / f"{d.name}_metrics.csv"
        if m.exists():
            kv = read_kv(m)
            rows.append({"weights": d.name, **{k: kv.get(k, "") for k in SUMMARY_KEYS}})
    path = out_root / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["weights"] + SUMMARY_KEYS)
        w.writeheader()
        w.writerows(rows)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_tag(w: Path) -> str:
    """Folder/file label: 'final' for final.pt, 'final_onnx' for final.onnx (so both can live in one folder)."""
    return w.stem if w.suffix.lower() == ".pt" else f"{w.stem}_{w.suffix.lower().lstrip('.')}"


def onnx_info(path: Path) -> dict:
    """Task / names / imgsz / batch from the metadata Ultralytics embeds in an ONNX export."""
    import ast
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    md = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map

    def parse(key):
        try:
            return ast.literal_eval(md[key]) if key in md else None
        except Exception:
            return None

    imgsz = parse("imgsz")
    return {"task": md.get("task"), "names": parse("names"), "batch": parse("batch"),
            "imgsz": int(imgsz[0]) if isinstance(imgsz, (list, tuple)) and imgsz else None}


def onnx_param_count(path: Path):
    """Parameter count from the graph initializers (BN is usually folded, so it differs slightly from the .pt count)."""
    try:
        import numpy as np
        import onnx
        m = onnx.load(str(path), load_external_data=False)
        return int(sum(int(np.prod(t.dims)) for t in m.graph.initializer))
    except Exception:
        return ""


# ----------------------------------------------------------------------------- evaluation
def evaluate_one(wpath: Path, args, data_yaml: Path, stats: dict, out_dir: Path, meta: dict, seg_provider) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO

    is_onnx = wpath.suffix.lower() == ".onnx"
    meta["model.format"] = "onnx" if is_onnx else "pt"
    if is_onnx:
        info = onnx_info(wpath)
        if not info["task"] or not info["names"]:
            raise Skip("ONNX file has no Ultralytics metadata (task/names); export it with model.export(format='onnx')")
        task, names = info["task"], list(info["names"].values())
        meta["model.export_batch"] = info["batch"]
        model = YOLO(str(wpath), task=task)  # explicit task: older Ultralytics versions cannot guess it from an .onnx path
    else:
        model = YOLO(str(wpath))
        task, names = model.task, list(model.names.values())
    meta["model.task"] = task
    if task not in ("detect", "segment"):
        raise Skip(f"model task is '{task}'; only detect and segment models are evaluated")
    if names != VOC_CLASSES:
        raise Skip("class names are not the 20 VOC classes (e.g. an un-fine-tuned COCO model); fine-tune on VOC first")
    is_seg = task == "segment"
    if is_seg:  # masks exist only for a subset of VOC2007, so seg models use that subset
        data_yaml, stats = seg_provider()

    train_imgsz = info["imgsz"] if is_onnx else model.overrides.get("imgsz")
    meta["model.train_imgsz"] = train_imgsz
    if args.imgsz == "auto":
        imgsz = int(train_imgsz) if isinstance(train_imgsz, (int, float)) else 640
    else:
        imgsz = int(args.imgsz)
    if is_onnx:
        meta["model.params"] = onnx_param_count(wpath)
        meta["model.params_source"] = "onnx graph initializers"
    else:
        meta["model.params"] = sum(p.numel() for p in model.model.parameters())
    meta.update({"eval.imgsz": imgsz, "eval.conf": args.conf, "eval.iou": args.iou, "eval.batch": args.batch,
                 "eval.device": args.device or "auto", "eval.n_images": stats["images"],
                 "eval.keep_difficult": stats.get("keep_difficult", "n/a (instance masks)"),
                 "eval.dataset": "VOC2007 segmentation subset" if is_seg else "VOC2007 (all annotated images)",
                 "eval.rect": "off (--no-rect)" if args.no_rect else "ultralytics default",
                 "env.ultralytics": ultralytics.__version__, "env.torch": torch.__version__})

    extra = {"rect": False} if args.no_rect else {}
    metrics = model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=args.batch, conf=args.conf,
                        iou=args.iou, device=args.device, workers=args.workers, plots=True,
                        project=str(out_dir.parent), name=out_dir.name, exist_ok=True, verbose=False, **extra)

    b = metrics.box
    meta.update({"metric.precision": round(float(b.mp), 5), "metric.recall": round(float(b.mr), 5),
                 "metric.mAP50": round(float(b.map50), 5), "metric.mAP50-95": round(float(b.map), 5)})
    if is_seg:
        s = metrics.seg
        meta.update({"metric.mask.precision": round(float(s.mp), 5), "metric.mask.recall": round(float(s.mr), 5),
                     "metric.mask.mAP50": round(float(s.map50), 5), "metric.mask.mAP50-95": round(float(s.map), 5)})
    meta.update({f"speed.{k}_ms": round(float(v), 3) for k, v in metrics.speed.items()})

    counts = stats.get("lines_per_class", {})
    header = ["class_id", "class_name", "instances", "precision", "recall", "mAP50", "mAP50-95"]
    if is_seg:
        header += ["mask_precision", "mask_recall", "mask_mAP50", "mask_mAP50-95"]
    with open(out_dir / f"{out_dir.name}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k, cid in enumerate(b.ap_class_index):
            p, r, ap50, ap = b.class_result(k)
            name = VOC_CLASSES[int(cid)]
            row = [int(cid), name, counts.get(name, ""), round(float(p), 5), round(float(r), 5),
                   round(float(ap50), 5), round(float(ap), 5)]
            if is_seg:
                row += [round(float(v), 5) for v in metrics.seg.class_result(k)]
            w.writerow(row)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", required=True, help="folder with .pt and/or .onnx files")
    ap.add_argument("--voc2007", required=True, help="folder containing VOC2007/ (or VOC2007 itself)")
    ap.add_argument("--out", default="evaluations")
    ap.add_argument("--data-root", default="datasets/voc2007_eval", help="where the converted VOC2007 is written")
    ap.add_argument("--imgsz", default="auto", help='"auto" = size stored in each checkpoint, or an integer')
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.001, help="Ultralytics validation default")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU, same as validation during training")
    ap.add_argument("--device", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-difficult", action="store_true", help="keep objects flagged difficult (default: drop)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--rebuild", action="store_true", help="re-convert VOC2007 even if already converted")
    ap.add_argument("--skip-existing", action="store_true", help="skip weights that already have a finished result")
    ap.add_argument("--recursive", action="store_true", help="also search sub-folders of --weights-dir")
    ap.add_argument("--no-rect", action="store_true",
                    help="force square letterboxing (rect=False) so .pt and .onnx are evaluated identically")
    args = ap.parse_args()

    wdir = Path(args.weights_dir).expanduser()
    found = wdir.rglob("*") if args.recursive else wdir.glob("*")
    weights = sorted(p for p in found if p.is_file() and p.suffix.lower() in (".pt", ".onnx"))
    if not weights:
        raise SystemExit(f"No .pt or .onnx files found in {wdir}")
    names = [weight_tag(w) for w in weights]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"Duplicate weight file names (results folders would collide): {sorted(dup)}")

    voc = find_voc2007(Path(args.voc2007).expanduser())
    data_root = Path(args.data_root).expanduser().resolve()
    data_yaml, stats = build_dataset(voc, data_root, args.keep_difficult, args.copy, args.rebuild)
    seg_cache = {}

    def seg_provider():
        if "v" not in seg_cache:
            seg_cache["v"] = build_seg_dataset(voc, data_root, args.copy, args.rebuild)
        return seg_cache["v"]

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for n, w in enumerate(weights, 1):
        tag = weight_tag(w)
        out_dir = out_root / tag
        done = out_dir / f"{tag}_metrics.csv"
        if args.skip_existing and done.exists() and read_kv(done).get("status") == "evaluated":
            print(f"[{n}/{len(weights)}] {w.name}: already evaluated, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{n}/{len(weights)}] {w.name}")
        meta = {"weights_file": w.name, "weights_path": str(w.resolve()), "weights_sha256": sha256(w),
                "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"), "status": "evaluated", "reason": ""}
        used = stats
        try:
            used = evaluate_one(w, args, data_yaml, stats, out_dir, meta, seg_provider)
            print(f"    mAP50-95 {meta['metric.mAP50-95']}   mAP50 {meta['metric.mAP50']}   "
                  f"P {meta['metric.precision']}   R {meta['metric.recall']}"
                  + (f"   mask mAP50-95 {meta['metric.mask.mAP50-95']}" if "metric.mask.mAP50-95" in meta else ""))
        except Skip as e:
            meta.update(status="skipped", reason=str(e))
            print(f"    skipped: {e}")
        except Exception as e:  # keep going with the remaining weights
            meta.update(status="failed", reason=repr(e))
            print(f"    FAILED: {e!r}")
        meta.update({f"dataset.{k}": v for k, v in used.items() if k != "voc_dir"})
        write_kv(done, meta)
        rebuild_summary(out_root)

    print(f"\nSummary: {rebuild_summary(out_root)}")


if __name__ == "__main__":  # guard needed for dataloader workers on Windows
    main()