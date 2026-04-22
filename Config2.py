import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ============ 你需要改的路径 ============
DATA_ROOT = Path("/root/Pure_Data_82_NoDamage")
VAL_IMG_DIR = DATA_ROOT / "images/val"
VAL_MSK_DIR = DATA_ROOT / "masks/val"

YOLO_WEIGHTS = Path("./yolo_pre_exp_gpt/stageB_finetune_stable/weights/best.pt")  # ✅ 用 StageB
SAM_BASE_CKPT = Path("sam_vit_b_01ec64.pth")  # SAM 原始权重
MEDSAM_FINETUNE = Path("medsam_fusion_best_gpt.pth")  # ✅ 你训练出的权重

OUT_DIR = Path("./fusion_debug_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 融合推荐推理参数 ============
YOLO_CONF = 0.01
YOLO_IOU = 0.40
YOLO_IMGSZ = 640
YOLO_MAXDET = 30

# box 扩框（建议与你训练一致：10%并夹住范围）
MIN_PAD = 0
MAX_PAD = 0

# 小框过滤阈值（别用 15，会误杀 calcification）
MIN_W = 6
MIN_H = 6

# 是否只保留“包含 box 中心点”的连通域（减少漂移）
KEEP_COMPONENT_AT_CENTER = False


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pred_bin = pred_bin.astype(np.uint8)
    gt_bin = gt_bin.astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum() + 1e-6
    return float((2 * inter + 1e-6) / denom)


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
    """只保留包含 (cx,cy) 的连通域；若中心落在背景则保留最大连通域"""
    m = mask_bin.astype(np.uint8)
    num, lab = cv2.connectedComponents(m)
    if num <= 2:
        return mask_bin

    cx = int(np.clip(cx, 0, m.shape[1] - 1))
    cy = int(np.clip(cy, 0, m.shape[0] - 1))
    cid = lab[cy, cx]

    if cid != 0:
        return (lab == cid).astype(np.uint8)

    # 中心点落在背景：取最大连通域
    areas = [(lab == k).sum() for k in range(1, num)]
    kmax = 1 + int(np.argmax(areas))
    return (lab == kmax).astype(np.uint8)


def overlay_vis(img_bgr, union_mask, boxes, clses, confs, class_names):
    vis = img_bgr.copy()
    # mask 红色叠加
    m = (union_mask > 0).astype(np.uint8) * 255
    colored = np.zeros_like(vis)
    colored[:, :, 2] = m
    vis = cv2.addWeighted(vis, 1.0, colored, 0.35, 0)

    # 画框
    for (b, c, s) in zip(boxes, clses, confs):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{class_names.get(int(c), str(int(c)))} {float(s):.2f}"
        cv2.putText(vis, txt, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


# ===================== 新增：YOLO Two-pass 推理 =====================
def yolo_predict_two_pass(yolo, img_path):
    # Pass-1：正常（你当前最优点）
    res1 = yolo.predict(
        source=str(img_path),
        conf=0.01,
        iou=0.40,
        imgsz=640,
        augment=False,
        max_det=30,
        verbose=False
    )[0]

    if res1.boxes is not None and len(res1.boxes) > 0:
        return res1

    # Pass-2：兜底（只在 boxes=0 时触发）
    res2 = yolo.predict(
        source=str(img_path),
        conf=0.005,
        iou=0.60,  # NMS 放松一点，保留更多候选
        imgsz=768,
        augment=True,
        max_det=80,
        verbose=False
    )[0]
    return res2


def sam_segment_with_refine(predictor, box_xyxy, W, H):
    # 第一次分割
    masks, _, _ = predictor.predict(box=box_xyxy[None, :], multimask_output=False)
    m1 = masks[0].astype(np.uint8)

    ys, xs = np.where(m1 > 0)
    if len(xs) < 30:  # 太小就不 refine
        return m1

    # 用第一次 mask 的外接框 refine（再 pad 一点点）
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    pad = int(0.05 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad);
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad);
    y2 = min(H, y2 + pad)

    box2 = np.array([x1, y1, x2, y2], dtype=np.int32)
    return m1

    # # 第二次分割（更紧的框）
    # masks2, _, _ = predictor.predict(box=box2[None, :], multimask_output=False)
    # return masks2[0].astype(np.uint8)


def main():
    # 1) Load YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))

    # 2) Load SAM + load finetune weights
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_BASE_CKPT))
    state = torch.load(str(MEDSAM_FINETUNE), map_location="cpu")
    sam.load_state_dict(state, strict=True)
    sam.to(DEVICE).eval()

    predictor = SamPredictor(sam)

    class_names = {0: "calcification", 1: "fibre", 2: "lipid"}

    img_files = sorted([p for p in VAL_IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    dices = []
    bad_cases = []

    # ===================== 新增：pos/neg dice & 漏检统计 =====================
    pos_dices = []
    neg_dices = []
    pos_no_box = 0
    pos_count = 0
    neg_count = 0

    for img_path in img_files:
        stem = img_path.stem
        msk_path = VAL_MSK_DIR / f"{stem}.png"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        gt = cv2.imread(str(msk_path), 0)
        if gt is None:
            continue
        gt_bin = (gt > 0).astype(np.uint8)

        # ===================== 新增：正负样本判定 =====================
        is_pos = (gt_bin.sum() > 0)

        # ===================== 改：YOLO two-pass 推理 =====================
        res = yolo_predict_two_pass(yolo, img_path)

        has_box = (res.boxes is not None and len(res.boxes) > 0)
        if is_pos:
            pos_count += 1
            if not has_box:
                pos_no_box += 1
        else:
            neg_count += 1

        if not has_box:
            pred_union = np.zeros((H, W), dtype=np.uint8)
            d = dice_score(pred_union, gt_bin)
            dices.append(d)

            # ===================== 新增：记录 pos/neg dices =====================
            if is_pos:
                pos_dices.append(d)
            else:
                neg_dices.append(d)

            if d < 0.20:
                bad_cases.append((img_path.name, d, 0))
            continue

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        # set image once
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        union_mask = np.zeros((H, W), dtype=np.uint8)

        kept_boxes = []
        kept_clses = []
        kept_confs = []

        for box, c, s in zip(boxes_xyxy, clses, confs):
            x1, y1, x2, y2 = box
            # if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
            #     continue

            # ===== 超大框轻度约束：面积过大且低置信度的框直接跳过 =====
            area = float((x2 - x1) * (y2 - y1))
            if area > 0.35 * W * H and float(s) < 0.2:
                continue

            # 小框过滤
            if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                continue

            pbox = pad_box_xyxy(box, W, H)  # (int) padded box
            px1, py1, px2, py2 = pbox.tolist()

            # masks, scores, _ = predictor.predict(
            #     box=pbox[None, :],
            #     multimask_output=False
            # )
            # m = masks[0].astype(np.uint8)  # (H,W)

            m = sam_segment_with_refine(predictor, pbox, W, H)

            # 可选：只保留包含 box 中心的连通域，减少漂移
            if KEEP_COMPONENT_AT_CENTER:
                cx = int((px1 + px2) * 0.5)
                cy = int((py1 + py2) * 0.5)
                m = keep_component_center(m, cx, cy)

            # box clipping（与训练监督一致）
            clipped = np.zeros_like(m, dtype=np.uint8)
            clipped[py1:py2, px1:px2] = m[py1:py2, px1:px2]

            union_mask = np.maximum(union_mask, clipped)

            kept_boxes.append(pbox)
            kept_clses.append(c)
            kept_confs.append(s)

        d = dice_score(union_mask, gt_bin)
        dices.append(d)

        # ===================== 新增：记录 pos/neg dices =====================
        if is_pos:
            pos_dices.append(d)
        else:
            neg_dices.append(d)

        # 保存低分可视化
        if d < 0.20:
            bad_cases.append((img_path.name, d, len(kept_boxes)))
            vis = overlay_vis(img_bgr, union_mask, kept_boxes, kept_clses, kept_confs, class_names)
            cv2.imwrite(str(OUT_DIR / f"BAD_{stem}_dice_{d:.3f}.jpg"), vis)

    mean_dice = float(np.mean(dices)) if len(dices) else 0.0
    print(f"\n🏁 Fusion Val Mean Dice = {mean_dice:.4f} (N={len(dices)})")

    bad_cases.sort(key=lambda x: x[1])
    print("\nWorst cases (dice < 0.20):")
    for name, d, nb in bad_cases[:30]:
        print(f"  {name} | Dice={d:.4f} | boxes={nb}")

    # ===================== 新增：打印 pos-only / neg-only / 漏检率 =====================
    print("\n========== Breakdown ==========")
    print("Mean Dice (all):", float(np.mean(dices)) if len(dices) else None)
    print("Mean Dice (pos only):", float(np.mean(pos_dices)) if len(pos_dices) else None)
    print("Mean Dice (neg only):", float(np.mean(neg_dices)) if len(neg_dices) else None)
    print("Pos frames:", pos_count, "Neg frames:", neg_count)
    print("Pos no-box (FN frames):", pos_no_box, "Rate:", float(pos_no_box / (pos_count + 1e-9)))


if __name__ == "__main__":
    main()