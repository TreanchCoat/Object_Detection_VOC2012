import datetime as dt
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# --- Model Discovery & Caching ---
def get_available_weights() -> list[str]:
    weights_dir = Path("weights")
    if not weights_dir.exists():
        weights_dir.mkdir(exist_ok=True)
    
    pt_files = sorted([str(p) for p in weights_dir.glob("*.pt")])
    if not pt_files:
        pt_files = ["yolo11n.pt"]
    return pt_files

MODEL_CACHE = {}

def get_model(weights_path: str) -> YOLO:
    if weights_path not in MODEL_CACHE:
        print(f"Loading YOLO model from: {weights_path}")
        MODEL_CACHE[weights_path] = YOLO(weights_path)
    return MODEL_CACHE[weights_path]

initial_weights = get_available_weights()[0]
default_model = get_model(initial_weights)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="YOLO Object Detection & Segmentation API",
    description="REST API and interactive GUI for running trained YOLO models.",
    version="1.0.0"
)

# --- Helper Functions ---
def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def result_to_detections(r, with_masks: bool) -> list[dict]:
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    xyxy, conf, cls = to_np(r.boxes.xyxy), to_np(r.boxes.conf), to_np(r.boxes.cls).astype(int)
    polys = r.masks.xy if (with_masks and getattr(r, "masks", None) is not None) else None
    
    for i in range(len(cls)):
        d = {
            "class_id": int(cls[i]),
            "class_name": r.names[int(cls[i])],
            "confidence": round(float(conf[i]), 4),
            "box_xyxy": [round(float(v), 1) for v in xyxy[i]],
        }
        if polys is not None:
            poly = np.asarray(polys[i], dtype=float)
            d["mask_area_px"] = round(polygon_area(poly), 1)
            d["mask_polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
        dets.append(d)
    return dets

def render_visualization(r, task: str, show_orig: bool, show_boxes: bool, show_masks: bool):
    if r is None:
        return None
    
    base_img = r.orig_img.copy() if show_orig else np.zeros_like(r.orig_img)
    
    orig_backup = r.orig_img
    r.orig_img = base_img
    
    draw_masks = show_masks if (task == "segment") else False
    vis = r.plot(masks=draw_masks, boxes=show_boxes)
    
    r.orig_img = orig_backup
    return Image.fromarray(vis[..., ::-1])


# --- REST API Endpoints ---
@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    weights: str = Form(initial_weights),
    max_det: int = Form(300),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    model = get_model(weights)
    task = model.task

    # imgsz is statically set to 640
    results = model.predict(
        source=image,
        imgsz=640,
        max_det=max_det,
        verbose=False
    )
    r = results[0]
    
    draw_masks = (task == "segment")
    dets = result_to_detections(r, with_masks=draw_masks)
    
    response_data = {
        "filename": file.filename,
        "model_weights": weights,
        "model_task": task,
        "image_size_hw": list(r.orig_shape),
        "n_detections": len(dets),
        "detections": dets,
        "timestamp": dt.datetime.now().isoformat()
    }
    return JSONResponse(content=response_data)


# --- Gradio GUI Integration ---
import gradio as gr

def predict_gui(weights_path, image, show_orig, show_boxes, show_masks):
    if image is None:
        return None, "Please upload an image.", None
    
    model = get_model(weights_path)
    task = model.task
    
    # imgsz is statically set to 640
    results = model.predict(
        source=image,
        imgsz=640,
        verbose=False
    )
    r = results[0]
    
    vis_pil = render_visualization(r, task, show_orig, show_boxes, show_masks)
    dets = result_to_detections(r, with_masks=(task == "segment"))
    formatted_output = json.dumps(dets, indent=2)
    
    return vis_pil, formatted_output, r

def update_toggles(r_state, weights_path, show_orig, show_boxes, show_masks):
    if r_state is None:
        return None
    model = get_model(weights_path)
    return render_visualization(r_state, model.task, show_orig, show_boxes, show_masks)

def on_model_change(weights_path):
    try:
        model = get_model(weights_path)
        is_seg = (model.task == "segment")
        return gr.update(visible=is_seg, value=is_seg)
    except Exception:
        return gr.update(visible=False, value=False)


# Build Custom Gradio Blocks UI
with gr.Blocks(title="YOLO Model Demo & Model Selector") as demo:
    result_state = gr.State(value=None)
    
    gr.Markdown("# YOLO Model Demo & Selector")
    gr.Markdown("Select a trained model from your `weights/` folder and upload an image to run predictions (Image size locked to 640).")
    
    available_weights = get_available_weights()
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=available_weights,
                value=available_weights[0],
                label="Select Model Weights",
                info="Choose from available .pt files in the weights/ directory"
            )
            image_input = gr.Image(type="pil", label="Input Image")
            
            gr.Markdown("### Visualization Toggles")
            toggle_orig = gr.Checkbox(value=True, label="Show Original Image")
            toggle_boxes = gr.Checkbox(value=True, label="Show Bounding Boxes")
            
            initial_is_seg = (get_model(available_weights[0]).task == "segment")
            toggle_masks = gr.Checkbox(
                value=initial_is_seg, 
                label="Show Segmentation Masks", 
                visible=initial_is_seg
            )
            
            submit_btn = gr.Button("Run Prediction", variant="primary")
            
        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="Prediction Visualization", interactive=False)
            
            with gr.Accordion("Show More: Detection Results (JSON)", open=False):
                json_output = gr.Code(
                    language="json", 
                    label="Raw Detections", 
                    lines=10, 
                    max_lines=15
                )

    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[toggle_masks]
    )

    submit_btn.click(
        fn=predict_gui,
        inputs=[model_dropdown, image_input, toggle_orig, toggle_boxes, toggle_masks],
        outputs=[image_output, json_output, result_state]
    )

    toggle_inputs = [result_state, model_dropdown, toggle_orig, toggle_boxes, toggle_masks]
    toggle_orig.change(fn=update_toggles, inputs=toggle_inputs, outputs=image_output)
    toggle_boxes.change(fn=update_toggles, inputs=toggle_inputs, outputs=image_output)
    toggle_masks.change(fn=update_toggles, inputs=toggle_inputs, outputs=image_output)

# Mount Gradio app onto FastAPI at the root URL
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)