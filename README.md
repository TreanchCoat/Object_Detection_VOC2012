# Object_Detection_VOC2012

## Setup

```bash
pip install -r requirements.txt
```

Download [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 2012 for training and 2007 for evaluation, so that your `VOCdevkit` folder contains `VOC2012/` and `VOC2007/`.

## 1. Configure

Create a YAML file, for example `configs/baseline_seg.yaml`:

```yaml
run:
  name: baseline_seg          # run folder is <project>/<name>_<timestamp>
  project: runs
  seed: 0
  strict_no_aug: true         # abort if any augmentation is left enabled

data:
  task: segment               # "detect" or "segment"
  voc_root: /path/to/VOCdevkit
  yolo_root: datasets/voc2012_seg
  imageset: Segmentation      # use "Main" for detection
  train_split: train
  val_split: val
  # optional: keep_difficult, use_all_annotated (detect only), min_contour_area (segment only)

model:
  weights: yolo11n-seg.pt     # segment needs a "-seg" model, detect needs a plain one

train:                        # any Ultralytics training argument
  epochs: 20
  imgsz: 640
  batch: 16

augmentation:                 # keys not listed here fall back to Ultralytics defaults
  hsv_h: 0.0
  hsv_s: 0.0
  hsv_v: 0.0
  degrees: 0.0
  translate: 0.0
  scale: 0.0
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.0
  mosaic: 0.0
  mixup: 0.0
  copy_paste: 0.0
  erasing: 0.0
```

## 2. Prepare the dataset (once per config)

```bash
python prepare_voc.py --config configs/baseline_seg.yaml
```


## 3. Train

```bash
python train.py --config configs/baseline_seg.yaml

# override any config value from the command line
python train.py --config configs/baseline_seg.yaml --set train.epochs=1 --set train.fraction=0.02
python train.py --config configs/baseline_seg.yaml --set augmentation.hsv_s=0.7 --set run.name=hsv_s07
```

Each run writes to `<project>/<run_id>/`:

- `weights/best.pt`, `weights/last.pt` and the usual Ultralytics outputs
- `epoch_metrics.csv`: per-epoch losses and validation metrics
- `run_metadata.csv`: config, overrides, environment, dataset stats, timing, best and final metrics
- `config_resolved.yaml`: the exact config used

## 4. Evaluate on VOC2007

Copy the weights you want to compare into one folder (rename them to meaningful names), then:

```bash
python evaluate.py --weights-dir eval_weights --voc2007 /path/to/VOCdevkit
```

Outputs go to `evaluations/`:


## 5. Run the demo app (GUI and REST API)

`app/app.py` serves a Gradio interface and a REST API from a single FastAPI server.

1. Put your trained `.pt` files in `app/weights/`. The folder is created on first start. If it is empty, the app falls back to `yolo11n.pt`, which Ultralytics downloads. Only `.pt` files are listed, so ONNX files are not picked up.
2. Start the server from inside the `app` folder, because the `weights/` path is relative to the working directory:

```bash
cd app
python app.py                                   # starts with auto-reload
# or
uvicorn app:app --host 0.0.0.0 --port 7860
```

3. Open http://localhost:7860.