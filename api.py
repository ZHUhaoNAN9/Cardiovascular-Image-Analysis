#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# api.py
import threading
from typing import Optional
from io import BytesIO
import base64

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

# ============ 路径（按你自己的实际文件位置） ============
YOLO_WEIGHTS = "./yolo_pre_exp_gpt/stageB_finetune_stable/weights/best.pt"
SAM_BASE_CKPT = "sam_vit_b_01ec64.pth"
MEDSAM_FINETUNE = "medsam_fusion_best_gpt.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "calcification", 1: "fibre", 2: "lipid"}

# ============ 推理参数 ============
YOLO_CONF = 0.01
YOLO_IOU  = 0.40
YOLO_IMGSZ = 640
YOLO_MAXDET = 30

MIN_PAD = 10
MAX_PAD = 80
MIN_W = 6
MIN_H = 6
KEEP_COMPONENT_AT_CENTER = True

# ============ 加载模型（进程启动时只加载一次） ============
yolo = YOLO(YOLO_WEIGHTS)

sam = sam_model_registry["vit_b"](checkpoint=SAM_BASE_CKPT)
state = torch.load(MEDSAM_FINETUNE, map_location="cpu")
sam.load_state_dict(state, strict=True)
sam.to(DEVICE).eval()
predictor = SamPredictor(sam)

# predictor.set_image 不是线程安全：并发必须加锁
predictor_lock = threading.Lock()


def pad_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad = int(np.clip(0.10 * max(bw, bh), MIN_PAD, MAX_PAD))
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(W, int(x2 + pad))
    y2 = min(H, int(y2 + pad))
    if x2 <= x1 + 1: x2 = min(W, x1 + 2)
    if y2 <= y1 + 1: y2 = min(H, y1 + 2)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def keep_component_center(mask_bin: np.ndarray, cx: int, cy: int):
    m = mask_bin.astype(np.uint8)
    num, lab = cv2.connectedComponents(m)
    if num <= 2:
        return mask_bin

    cx = int(np.clip(cx, 0, m.shape[1]-1))
    cy = int(np.clip(cy, 0, m.shape[0]-1))
    cid = lab[cy, cx]

    if cid != 0:
        return (lab == cid).astype(np.uint8)

    areas = [(lab == k).sum() for k in range(1, num)]
    kmax = 1 + int(np.argmax(areas))
    return (lab == kmax).astype(np.uint8)


def sam_segment_with_refine(predictor: SamPredictor, box_xyxy: np.ndarray, W: int, H: int):
    # 第一次分割
    masks, _, _ = predictor.predict(box=box_xyxy[None, :], multimask_output=False)
    m1 = masks[0].astype(np.uint8)

    ys, xs = np.where(m1 > 0)
    if len(xs) < 30:
        return m1

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    pad = int(0.05 * max(x2-x1, y2-y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

    box2 = np.array([x1, y1, x2, y2], dtype=np.int32)

    masks2, _, _ = predictor.predict(box=box2[None, :], multimask_output=False)
    return masks2[0].astype(np.uint8)


def yolo_predict_two_pass_on_array(yolo, img_rgb: np.ndarray):
    # Pass-1
    res1 = yolo.predict(
        source=img_rgb,
        conf=0.01,
        iou=0.40,
        imgsz=640,
        augment=False,
        max_det=30,
        verbose=False
    )[0]
    if res1.boxes is not None and len(res1.boxes) > 0:
        return res1

    # Pass-2 fallback
    res2 = yolo.predict(
        source=img_rgb,
        conf=0.005,
        iou=0.60,
        imgsz=768,
        augment=True,
        max_det=80,
        verbose=False
    )[0]
    return res2


def process_image(img_rgb: np.ndarray):
    H, W = img_rgb.shape[:2]

    # YOLO
    res = yolo_predict_two_pass_on_array(yolo, img_rgb)
    has_box = (res.boxes is not None and len(res.boxes) > 0)
    if not has_box:
        return np.zeros((H, W), dtype=np.uint8)

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    union_mask = np.zeros((H, W), dtype=np.uint8)

    # SAM：set_image + predict 需要锁，避免并发互相覆盖
    with predictor_lock:
        predictor.set_image(img_rgb)

        for box, s in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = box

            # 超大框 & 低置信度过滤
            area = float((x2 - x1) * (y2 - y1))
            if area > 0.35 * W * H and float(s) < 0.2:
                continue

            # 小框过滤
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue

            pbox = pad_box_xyxy(box, W, H)
            px1, py1, px2, py2 = pbox.tolist()

            m = sam_segment_with_refine(predictor, pbox, W, H)

            if KEEP_COMPONENT_AT_CENTER:
                cx = int((px1 + px2) * 0.5)
                cy = int((py1 + py2) * 0.5)
                m = keep_component_center(m, cx, cy)

            clipped = np.zeros_like(m, dtype=np.uint8)
            clipped[py1:py2, px1:px2] = m[py1:py2, px1:px2]
            union_mask = np.maximum(union_mask, clipped)

    # 你原来的 mask 是 0/1，这里转成 0/255 更适合 png
    union_mask = (union_mask > 0).astype(np.uint8) * 255
    return union_mask


# 你的类别映射（按你训练时的 class id）
CLASS_NAMES = {0: "calcification", 1: "fibre", 2: "lipid"}

def process_image_with_meta(img_rgb: np.ndarray):
    H, W = img_rgb.shape[:2]

    res = yolo_predict_two_pass_on_array(yolo, img_rgb)
    has_box = (res.boxes is not None and len(res.boxes) > 0)
    if not has_box:
        mask = np.zeros((H, W), dtype=np.uint8)
        return mask, []

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    clses = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    detections = []
    for box, c, s in zip(boxes_xyxy, clses, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "class_id": int(c),
            "class_name": CLASS_NAMES.get(int(c), str(int(c))),
            "conf": float(s),
            "box_xyxy": [x1, y1, x2, y2],
        })

    # 生成 union_mask（用你现有的逻辑）
    mask = process_image(img_rgb)  # 你现有的返回 0/255 的 mask
    return mask, detections




@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(img)

    mask = process_image(img_rgb)

    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        return Response(content=b"", media_type="application/octet-stream", status_code=500)

    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...), class_id: Optional[int] = None):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(img)
    H, W = img_rgb.shape[:2]

    # YOLO 推理（得到所有 detections）
    res = yolo_predict_two_pass_on_array(yolo, img_rgb)
    has_box = (res.boxes is not None and len(res.boxes) > 0)

    if not has_box:
        mask = np.zeros((H, W), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return {"mask_png_base64": b64, "detections": []}

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    clses = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    detections = []
    for box, c, s in zip(boxes_xyxy, clses, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "class_id": int(c),
            "class_name": CLASS_NAMES.get(int(c), str(int(c))),
            "conf": float(s),
            "box_xyxy": [x1, y1, x2, y2],
        })

    # ====== 按 class_id 过滤，用于生成 mask（不影响 detections 输出）======
    if class_id is None:
        keep = np.ones_like(clses, dtype=bool)
    else:
        keep = (clses == int(class_id))

    f_boxes = boxes_xyxy[keep]
    f_confs = confs[keep]

    # ====== 用过滤后的 boxes 做 SAM 分割，生成 union mask ======
    union_mask = np.zeros((H, W), dtype=np.uint8)

    with predictor_lock:
        predictor.set_image(img_rgb)

        for box, s in zip(f_boxes, f_confs):
            x1, y1, x2, y2 = box

            area = float((x2 - x1) * (y2 - y1))
            if area > 0.35 * W * H and float(s) < 0.2:
                continue
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue

            pbox = pad_box_xyxy(box, W, H)
            px1, py1, px2, py2 = pbox.tolist()

            m = sam_segment_with_refine(predictor, pbox, W, H)
            if KEEP_COMPONENT_AT_CENTER:
                cx = int((px1 + px2) * 0.5)
                cy = int((py1 + py2) * 0.5)
                m = keep_component_center(m, cx, cy)

            clipped = np.zeros_like(m, dtype=np.uint8)
            clipped[py1:py2, px1:px2] = m[py1:py2, px1:px2]
            union_mask = np.maximum(union_mask, clipped)

    union_mask = (union_mask > 0).astype(np.uint8) * 255

    ok, encoded = cv2.imencode(".png", union_mask)
    if not ok:
        return {"error": "encode failed"}

    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return {"mask_png_base64": b64, "detections": detections}
