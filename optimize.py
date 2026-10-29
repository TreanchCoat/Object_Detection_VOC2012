from ultralytics import YOLO

m = YOLO("final.pt")

m.export(format="onnx", imgsz=640, simplify=True)            # FP32 ONNX
m.export(format="onnx", imgsz=640, simplify=True, half=True)  # FP16 (may need a GPU at export time)

# INT8 needs a calibration dataset; supported formats include OpenVINO, TFLite and TensorRT
m.export(format="openvino", imgsz=640, int8=True, data="your_data.yaml")