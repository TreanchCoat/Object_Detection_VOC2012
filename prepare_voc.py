#!/usr/bin/env python
"""Convert PASCAL VOC 2012 to Ultralytics YOLO format, for either task.

data.task: detect   -> boxes from Annotations/*.xml          (ImageSets/<imageset>/<split>.txt)
data.task: segment  -> polygons from SegmentationObject/Class PNGs (ImageSets/Segmentation recommended)

Writes into <yolo_root>:
  images/{train,val}/   symlinks to the originals (copies if symlinks fail)
  labels/{train,val}/   YOLO txt labels
  voc2012.yaml          dataset file for Ultralytics
  stats.json            counts, read by train.py for run metadata
"""
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
CLASS_TO_ID = {c: i for i, c in enumerate(VOC_CLASSES)}


def find_voc2012(voc_root: Path) -> Path:
    for cand in (voc_root / "VOC2012", voc_root / "VOCdevkit" / "VOC2012", voc_root):
        if (cand / "JPEGImages").is_dir() and (cand / "Annotations").is_dir():
            return cand
    raise SystemExit(f"Could not find VOC2012 (JPEGImages/ + Annotations/) under {voc_root}")


# ----------------------------------------------------------------------------- detection
def convert_annotation(xml_path: Path, img_path: Path, keep_difficult: bool):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text)) if size is not None else 0
    h = int(float(size.find("height").text)) if size is not None else 0
    if w <= 0 or h <= 0:  # a few VOC annotations have missing/zero size
        with Image.open(img_path) as im:
            w, h = im.size

    lines, per_class, skipped_difficult, skipped_degenerate = [], Counter(), 0, 0
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_ID:
            continue
        diff = obj.find("difficult")
        if diff is not None and int(diff.text) == 1 and not keep_difficult:
            skipped_difficult += 1
            continue
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        xmin, xmax = max(0.0, xmin), min(float(w), xmax)
        ymin, ymax = max(0.0, ymin), min(float(h), ymax)
        if xmax <= xmin or ymax <= ymin:
            skipped_degenerate += 1
            continue
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        per_class[name] += 1
    return lines, per_class, skipped_difficult, skipped_degenerate


# ----------------------------------------------------------------------------- segmentation
def convert_segmentation(obj_path: Path, cls_path: Path, min_area: float):
    """One YOLO-seg line per contour of every instance in SegmentationObject.

    Instance class = majority class (SegmentationClass) inside the instance mask.
    Pixel values: 0 = background, 255 = void/boundary band (ignored), 1..20 = classes / instance ids.
    An instance split into several fragments by occlusion yields one line per fragment
    (same behaviour as Ultralytics' own mask->YOLO converter).
    """
    import cv2
    import numpy as np

    obj = np.array(Image.open(obj_path))
    cls = np.array(Image.open(cls_path))
    if obj.ndim != 2 or cls.ndim != 2:
        raise ValueError(f"Expected palette/indexed PNGs, got shapes {obj.shape} / {cls.shape}: {obj_path.name}")
    h, w = obj.shape

    lines, per_class, skipped_small = [], Counter(), 0
    for inst_id in np.unique(obj):
        if inst_id in (0, 255):
            continue
        mask = obj == inst_id
        vals = cls[mask]
        vals = vals[(vals > 0) & (vals < 255)]
        if vals.size == 0:
            continue
        class_idx = int(np.bincount(vals).argmax()) - 1  # PNG classes are 1..20
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3 or cv2.contourArea(c) < min_area:
                skipped_small += 1
                continue
            pts = c.reshape(-1, 2).astype(float)
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts)
            lines.append(f"{class_idx} {coords}")
            per_class[VOC_CLASSES[class_idx]] += 1
    return lines, per_class, 0, skipped_small


# ----------------------------------------------------------------------------- misc
def link_or_copy(src: Path, dst: Path, force_copy: bool):
    if dst.exists() or dst.is_symlink():
        return
    if not force_copy:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:  # e.g. Windows without symlink privilege
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_det.yaml")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["data"]
    task = cfg.get("task", "detect")
    if task not in ("detect", "segment"):
        raise SystemExit(f"data.task must be 'detect' or 'segment', got {task!r}")
    imageset = cfg.get("imageset", "Main")
    min_area = float(cfg.get("min_contour_area", 16))
    keep_difficult = bool(cfg.get("keep_difficult", False))
    use_all = bool(cfg.get("use_all_annotated", False))
    if use_all and task != "detect":
        raise SystemExit("data.use_all_annotated only applies to task: detect (masks exist for a subset only)")

    voc = find_voc2012(Path(cfg["voc_root"]).expanduser())
    out = Path(cfg["yolo_root"]).expanduser().resolve()

    stats = {"task": task, "imageset": imageset, "keep_difficult": keep_difficult, "use_all_annotated": use_all, "voc_dir": str(voc), "splits": {}}
    for split_key, out_name in (("train_split", "train"), ("val_split", "val")):
        split = cfg[split_key]
        list_file = voc / "ImageSets" / imageset / f"{split}.txt"
        ids = list_file.read_text().split()
        official_ids = set(ids)
        if use_all and out_name == "train":
            # every image that has an XML + JPEG, minus the official val images (val stays identical to the
            # standard baseline so results remain comparable)
            val_ids = set((voc / "ImageSets" / imageset / f"{cfg['val_split']}.txt").read_text().split())
            ids = sorted(p.stem for p in (voc / "Annotations").glob("*.xml")
                         if (voc / "JPEGImages" / f"{p.stem}.jpg").exists() and p.stem not in val_ids)
        img_dir, lbl_dir = out / "images" / out_name, out / "labels" / out_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        per_class, n_inst, n_empty, sk_a, sk_b = Counter(), 0, 0, 0, 0
        extra_class, n_extra, n_extra_empty = Counter(), 0, 0  # images outside the official list
        for i in ids:
            src_img = voc / "JPEGImages" / f"{i}.jpg"
            if task == "detect":
                lines, pc, a, b = convert_annotation(voc / "Annotations" / f"{i}.xml", src_img, keep_difficult)
            else:
                lines, pc, a, b = convert_segmentation(
                    voc / "SegmentationObject" / f"{i}.png", voc / "SegmentationClass" / f"{i}.png", min_area)
            (lbl_dir / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            link_or_copy(src_img, img_dir / f"{i}.jpg", args.copy)
            per_class.update(pc)
            if i not in official_ids:
                n_extra += 1
                n_extra_empty += not lines
                extra_class.update(pc)
            n_inst += len(lines)
            n_empty += not lines
            sk_a += a
            sk_b += b

        stats["splits"][out_name] = {
            "voc_split": split, "images": len(ids), "label_lines": n_inst, "images_without_objects": n_empty,
            "skipped_difficult": sk_a,
            "skipped_degenerate_or_small": sk_b,
            "lines_per_class": dict(per_class),
        }
        if use_all and out_name == "train":
            stats["splits"][out_name].update({
                "official_list_images": len(official_ids), "extra_images": n_extra,
                "extra_images_without_objects": n_extra_empty, "extra_lines_per_class": dict(extra_class),
            })
            print(f"[train] {n_extra} extra images beyond the official list; "
                  f"objects in extras: {dict(extra_class.most_common())}")
        print(f"[{out_name}] task={task} imageset={imageset}/{split}: {len(ids)} images, {n_inst} label lines "
              f"(skipped: {sk_a} difficult, {sk_b} degenerate/small)")

    ds = {"path": str(out), "train": "images/train", "val": "images/val",
          "names": {i: c for i, c in enumerate(VOC_CLASSES)}}
    (out / "voc2012.yaml").write_text(yaml.safe_dump(ds, sort_keys=False))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Dataset ready: {out / 'voc2012.yaml'}")


if __name__ == "__main__":
    main()
