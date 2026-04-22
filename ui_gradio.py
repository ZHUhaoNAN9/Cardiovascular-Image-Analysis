import base64
import io
import os
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import gradio as gr

FASTAPI_JSON = "http://127.0.0.1:8004/predict_json"

CLASS_MAP = {
    "all": None,
    "calcification": 0,
    "fibre": 1,
    "lipid": 2,
}

CLASS_ORDER = ["calcification", "fibre", "lipid"]

CLASS_COLORS = {
    "calcification": (255, 100, 100),
    "fibre": (100, 255, 100),
    "lipid": (100, 100, 255),
}


def decode_mask_png(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("L")


def confidence_status(detections):
    if not detections:
        return "No detection"

    max_conf = max(d["conf"] for d in detections)

    if max_conf >= 0.3:
        level = "High"
    elif max_conf >= 0.15:
        level = "Medium"
    else:
        level = "Low / Uncertain"

    return f"{level} (max_conf = {max_conf:.4f})"


def make_overlay(
    img: Image.Image,
    mask: Image.Image,
    detections,
    show_mask: bool,
    alpha: int,
    show_boxes: bool,
    mask_color: str = "cyan",
):
    img = img.convert("RGB")
    out = img.copy()

    if show_mask and mask is not None:
        if mask_color == "auto":
            mask_arr = np.array(mask).astype(np.uint8)
            color_layer = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
            
            for d in detections:
                cls_name = d["class_name"]
                x1, y1, x2, y2 = [int(v) for v in d["box_xyxy"]]
                color = CLASS_COLORS.get(cls_name, (0, 255, 255))
                region_mask = mask_arr[y1:y2, x1:x2]
                color_layer[y1:y2, x1:x2][region_mask > 0] = color
            
            img_arr = np.array(out).astype(np.uint8)
            a = (alpha / 255.0) * (mask_arr > 0).astype(np.float32)
            a = a[..., None]
            blended = img_arr.astype(np.float32) * (1 - a) + color_layer.astype(np.float32) * a
            out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        else:
            color_map = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "cyan": (0, 255, 255),
                "yellow": (255, 255, 0),
            }
            color = color_map.get(mask_color, (0, 255, 255))
            
            img_arr = np.array(out).astype(np.uint8)
            mask_arr = np.array(mask).astype(np.uint8)
            
            color_layer = np.zeros_like(img_arr, dtype=np.uint8)
            color_layer[..., 0] = color[0]
            color_layer[..., 1] = color[1]
            color_layer[..., 2] = color[2]
            
            a = (alpha / 255.0) * (mask_arr > 0).astype(np.float32)
            a = a[..., None]
            
            blended = img_arr.astype(np.float32) * (1 - a) + color_layer.astype(np.float32) * a
            out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    if show_boxes and detections:
        draw = ImageDraw.Draw(out)
        max_idx = int(np.argmax([d["conf"] for d in detections]))

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["box_xyxy"]
            name = d["class_name"]
            conf = d["conf"]

            if i == max_idx:
                box_color = (255, 255, 0)
                text_color = (255, 255, 0)
            else:
                box_color = CLASS_COLORS.get(name, (0, 255, 0))
                text_color = CLASS_COLORS.get(name, (0, 255, 0))

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            draw.text((x1, max(0, y1 - 18)), f"{name} {conf:.2f}", fill=text_color)

    return out


def stats_by_class(detections):
    rows = []
    for cls_name in CLASS_ORDER:
        confs = [d["conf"] for d in detections if d["class_name"] == cls_name]
        if len(confs) == 0:
            rows.append({
                "Type": cls_name,
                "Count": 0,
                "Mean Conf": "-",
                "Max Conf": "-",
            })
        else:
            rows.append({
                "Type": cls_name,
                "Count": len(confs),
                "Mean Conf": round(float(np.mean(confs)), 4),
                "Max Conf": round(float(np.max(confs)), 4),
            })
    return pd.DataFrame(rows)


def det_table(detections, class_filter_name):
    if not detections:
        return pd.DataFrame(columns=["Type", "Conf", "X1", "Y1", "X2", "Y2"])

    rows = []
    for d in detections:
        if class_filter_name != "all" and d["class_name"] != class_filter_name:
            continue
        rows.append({
            "Type": d["class_name"],
            "Conf": round(d["conf"], 4),
            "X1": round(d["box_xyxy"][0], 1),
            "Y1": round(d["box_xyxy"][1], 1),
            "X2": round(d["box_xyxy"][2], 1),
            "Y2": round(d["box_xyxy"][3], 1),
        })
    return pd.DataFrame(rows)


def create_detection_chart(detections):
    if not detections:
        return None
    
    counts = {cls: 0 for cls in CLASS_ORDER}
    for d in detections:
        cls_name = d["class_name"]
        if cls_name in counts:
            counts[cls_name] += 1
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.keys(), counts.values(), color=[f"rgb({c[0]},{c[1]},{c[2]})" for c in CLASS_COLORS.values()])
    ax.set_ylabel("Count")
    ax.set_title("Detections by Class")
    ax.set_ylim(0, max(counts.values()) + 2 if counts.values() else 1)
    
    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf).convert("RGB")


def create_conf_chart(detections):
    if not detections:
        return None
    
    confs_by_class = {cls: [] for cls in CLASS_ORDER}
    for d in detections:
        cls_name = d["class_name"]
        if cls_name in confs_by_class:
            confs_by_class[cls_name].append(d["conf"])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    data = []
    labels = []
    colors = []
    for cls in CLASS_ORDER:
        if confs_by_class[cls]:
            data.append(confs_by_class[cls])
            labels.append(cls)
            colors.append(f"rgb({CLASS_COLORS[cls][0]},{CLASS_COLORS[cls][1]},{CLASS_COLORS[cls][2]})")
    
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence Distribution by Class")
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf).convert("RGB")


def infer_once(image_path, class_filter):
    if image_path is None:
        empty_df = pd.DataFrame(columns=["Type", "Count", "Mean Conf", "Max Conf"])
        empty_df2 = pd.DataFrame(columns=["Type", "Conf", "X1", "Y1", "X2", "Y2"])
        return (
            "Please upload an image first.",
            None, None, None,
            empty_df, empty_df2,
            None, None, None, None,
            None, None, None, None,
        )

    class_id = CLASS_MAP[class_filter]

    try:
        with open(image_path, "rb") as f:
            params = {}
            if class_id is not None:
                params["class_id"] = class_id

            r = requests.post(
                FASTAPI_JSON,
                params=params,
                files={"file": f},
                timeout=600
            )

        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return (
            "Request timeout! Please try a smaller image or check the server.",
            None, None, None,
            pd.DataFrame(), pd.DataFrame(),
            None, None, None, None,
            None, None, None, None,
        )
    except Exception as e:
        return (
            f"Error: {str(e)}",
            None, None, None,
            pd.DataFrame(), pd.DataFrame(),
            None, None, None, None,
            None, None, None, None,
        )

    img = Image.open(image_path).convert("RGB")
    mask = decode_mask_png(data["mask_png_base64"])
    dets_all = data.get("detections", [])

    if class_filter == "all":
        dets_view = dets_all
    else:
        dets_view = [d for d in dets_all if d["class_name"] == class_filter]

    status_text = confidence_status(dets_view)
    stats_df = stats_by_class(dets_all)
    dets_df = det_table(dets_all, class_filter)

    overlay = make_overlay(
        img=img,
        mask=mask,
        detections=dets_view,
        show_mask=True,
        alpha=160,
        show_boxes=True,
        mask_color="cyan",
    )

    chart1 = create_detection_chart(dets_all)
    chart2 = create_conf_chart(dets_all)

    return (
        status_text, 
        overlay, img, mask,
        stats_df, dets_df,
        chart1, chart2,
        dets_all, dets_view,
    )


def render_only(state_img, state_mask, state_dets_view, show_mask, alpha, show_boxes, mask_color):
    if state_img is None:
        return None

    overlay = make_overlay(
        img=state_img,
        mask=state_mask,
        detections=state_dets_view,
        show_mask=show_mask,
        alpha=alpha,
        show_boxes=show_boxes,
        mask_color=mask_color,
    )
    return overlay


def export_results(state_img, state_mask, state_dets_all, export_format):
    if state_img is None or state_dets_all is None:
        return None, "No results to export"
    
    try:
        if export_format == "zip":
            temp_dir = tempfile.mkdtemp()
            base_path = os.path.join(temp_dir, "result")
            
            state_img.save(f"{base_path}_original.png")
            state_mask.save(f"{base_path}_mask.png")
            
            overlay = make_overlay(
                img=state_img,
                mask=state_mask,
                detections=state_dets_all,
                show_mask=True,
                alpha=200,
                show_boxes=True,
                mask_color="auto",
            )
            overlay.save(f"{base_path}_overlay.png")
            
            df = pd.DataFrame(state_dets_all)
            df.to_csv(f"{base_path}_detections.csv", index=False)
            
            zip_path = f"{temp_dir}_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(f"{base_path}_original.png", "original.png")
                zf.write(f"{base_path}_mask.png", "mask.png")
                zf.write(f"{base_path}_overlay.png", "overlay.png")
                zf.write(f"{base_path}_detections.csv", "detections.csv")
            
            with open(zip_path, "rb") as f:
                return f.read(), "Export successful"
        else:
            return None, "Unsupported format"
    except Exception as e:
        return None, f"Export failed: {str(e)}"


def batch_process(image_paths, class_filter):
    if not image_paths:
        return "Please upload at least one image.", None, None, None, None, None
    
    results = []
    successful = 0
    failed = 0
    
    empty_df = pd.DataFrame(columns=["Type", "Count", "Mean Conf", "Max Conf"])
    
    for img_path in image_paths:
        try:
            class_id = CLASS_MAP[class_filter]
            with open(img_path, "rb") as f:
                params = {}
                if class_id is not None:
                    params["class_id"] = class_id
                r = requests.post(FASTAPI_JSON, params=params, files={"file": f}, timeout=600)
                r.raise_for_status()
                data = r.json()
            
            dets = data.get("detections", [])
            results.append({
                "filename": os.path.basename(img_path),
                "total_detections": len(dets),
                "status": "Success"
            })
            successful += 1
        except Exception as e:
            results.append({
                "filename": os.path.basename(img_path),
                "total_detections": 0,
                "status": f"Failed: {str(e)[:50]}"
            })
            failed += 1
    
    summary = f"Processed {len(image_paths)} images: {successful} successful, {failed} failed"
    results_df = pd.DataFrame(results)
    
    return summary, results_df, empty_df, empty_df, None, None


with gr.Blocks(
    title="Cardiovascular Segmentation",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: bold;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        background: #f0f9ff;
        border: 1px solid #bae6fd;
    }
    """
) as demo:
    gr.HTML('<div class="main-header">🫀 YOLO + SAM Cardiovascular Segmentation</div>')
    gr.Markdown("Upload cardiovascular images for automatic segmentation and detection using YOLO + SAM models.")

    state_img = gr.State()
    state_mask = gr.State()
    state_dets_all = gr.State()
    state_dets_view = gr.State()

    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Image")
                inp = gr.Image(type="filepath", label="Upload Image", height=300)
                
                with gr.Accordion("Detection Settings", open=True):
                    class_filter = gr.Dropdown(
                        choices=["all", "calcification", "fibre", "lipid"],
                        value="all",
                        label="Class Filter"
                    )
                    gr.Markdown("*Note: Changing filter requires re-inference*")
                
                btn = gr.Button("🚀 Start Inference", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                out_overlay = gr.Image(label="Overlay Result", height=300)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Display Options")
                with gr.Row():
                    show_mask = gr.Checkbox(value=True, label="Show Mask")
                    show_boxes = gr.Checkbox(value=True, label="Show Boxes")
                
                alpha = gr.Slider(0, 255, value=160, step=5, label="Mask Transparency")
                mask_color = gr.Dropdown(
                    choices=["cyan", "green", "yellow", "red", "auto"],
                    value="auto",
                    label="Mask Color"
                )

        gr.Markdown("---")
        gr.Markdown("### Analysis")
        
        with gr.Row():
            with gr.Column():
                out_status = gr.Textbox(label="Detection Status", lines=2)
            with gr.Column():
                out_stats = gr.Dataframe(label="Statistics by Class")

        with gr.Row():
            out_table = gr.Dataframe(label="Detection Details")
            out_chart1 = gr.Image(label="Detection Count Chart")
            out_chart2 = gr.Image(label="Confidence Distribution")

        gr.Markdown("---")
        gr.Markdown("### Export")
        with gr.Row():
            export_btn = gr.Button("📥 Export Results (ZIP)")
            export_format = gr.Dropdown(choices=["zip"], value="zip", visible=False)
            export_output = gr.File(label="Download")
            export_msg = gr.Textbox(label="Export Status", lines=1)

    with gr.Tab("Batch Processing"):
        gr.Markdown("### Batch Processing")
        gr.Markdown("Process multiple images at once. Results will be summarized.")
        
        with gr.Row():
            batch_inp = gr.File(file_count="multiple", file_types=["image"], label="Upload Multiple Images")
            batch_class = gr.Dropdown(
                choices=["all", "calcification", "fibre", "lipid"],
                value="all",
                label="Class Filter"
            )
        
        batch_btn = gr.Button("🚀 Process Batch", variant="primary")
        
        gr.Markdown("### Batch Results")
        batch_summary = gr.Textbox(label="Summary")
        batch_results = gr.Dataframe(label="Per-Image Results")
        batch_stats = gr.Dataframe(label="Aggregated Statistics")

    btn.click(
        infer_once,
        inputs=[inp, class_filter],
        outputs=[
            out_status,
            out_overlay,
            state_img,
            state_mask,
            out_stats,
            out_table,
            out_chart1,
            out_chart2,
            state_dets_all,
            state_dets_view,
        ]
    ).then(
        lambda: gr.update(visible=True),
        outputs=export_format
    )

    export_btn.click(
        export_results,
        inputs=[state_img, state_mask, state_dets_all, export_format],
        outputs=[export_output, export_msg]
    )

    alpha.change(
        render_only,
        inputs=[state_img, state_mask, state_dets_view, show_mask, alpha, show_boxes, mask_color],
        outputs=out_overlay
    )

    show_mask.change(
        render_only,
        inputs=[state_img, state_mask, state_dets_view, show_mask, alpha, show_boxes, mask_color],
        outputs=out_overlay
    )

    show_boxes.change(
        render_only,
        inputs=[state_img, state_mask, state_dets_view, show_mask, alpha, show_boxes, mask_color],
        outputs=out_overlay
    )

    mask_color.change(
        render_only,
        inputs=[state_img, state_mask, state_dets_view, show_mask, alpha, show_boxes, mask_color],
        outputs=out_overlay
    )

    batch_btn.click(
        batch_process,
        inputs=[batch_inp, batch_class],
        outputs=[batch_summary, batch_results, batch_stats, out_stats, out_chart1, out_chart2]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
