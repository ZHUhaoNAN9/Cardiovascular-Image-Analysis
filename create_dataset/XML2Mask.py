import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ================= 配置区域 =================
XML_FILE = r'E:\Cardio_Data\Data18\18XML\annotations.xml'
OUTPUT_ROOT = r'E:\Cardio_Data\Data18\my_dataset'

# 您的实际标签映射
CLASS_MAP = {
    "calcification": 0,
    "fibre": 1,
    "lipid": 2,
    "damage": 3
}


# ===========================================

def convert_cvat_xml_robust(xml_path, out_root):
    mask_dir = os.path.join(out_root, 'masks')
    labels_dir = os.path.join(out_root, 'labels')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ 读取 XML 失败: {e}")
        return

    print(f"🚀 开始转换 (带自动修复功能): {xml_path}")
    count = 0

    for image in root.findall('image'):
        file_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        yolo_lines = []
        mask = np.zeros((height, width), dtype=np.uint8)
        has_valid_obj = False

        # 查找所有 Polyline 和 Polygon
        all_shapes = list(image.findall('polygon')) + list(image.findall('polyline'))

        for shape in all_shapes:
            label = shape.get('label')
            if label not in CLASS_MAP: continue

            class_id = CLASS_MAP[label]
            points_str = shape.get('points')

            points = []
            try:
                for p in points_str.split(';'):
                    x_s, y_s = p.split(',')
                    points.append([float(x_s), float(y_s)])
            except ValueError:
                continue

            points_np = np.array(points, dtype=np.int32)
            points_float = np.array(points, dtype=np.float32)

            # --- A. 制作 Mask (并修复) ---
            # 1. 先用 fillPoly 填充（如果有交叉，可能会有空洞）
            # 注意：这里我们在一个临时的小图层上画，防止不同斑块粘连
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [points_np], color=1)  # 先填成1

            # 【核心修复代码】形态学闭运算
            # 定义一个 5x5 的核，像熨斗一样把空洞熨平
            kernel = np.ones((5, 5), np.uint8)
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)

            # 将修复好的临时 Mask 合并到总 Mask 上
            # 只有当 temp_mask > 0 的地方，才涂上对应的 class_id + 1
            mask[temp_mask > 0] = class_id + 1

            # --- B. 制作 YOLO ---
            # 使用 points 的极值计算外接矩形，这不受交叉影响，非常稳
            x_min = np.min(points_float[:, 0])
            y_min = np.min(points_float[:, 1])
            x_max = np.max(points_float[:, 0])
            y_max = np.max(points_float[:, 1])

            dw = 1.0 / width
            dh = 1.0 / height
            w_box = (x_max - x_min) * dw
            h_box = (y_max - y_min) * dh
            x_center = ((x_min + x_max) / 2.0) * dw
            y_center = ((y_min + y_max) / 2.0) * dh

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")
            has_valid_obj = True

        if has_valid_obj:
            # 保存 Mask (PNG)
            mask_name = os.path.splitext(file_name)[0] + ".png"
            cv2.imwrite(os.path.join(mask_dir, mask_name), mask)

            # 保存 YOLO (TXT)
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            with open(os.path.join(labels_dir, txt_name), 'w') as f:
                f.write('\n'.join(yolo_lines))

            count += 1

    print(f"✅ 转换并修复完成！共处理 {count} 张图片。")


if __name__ == "__main__":
    convert_cvat_xml_robust(XML_FILE, OUTPUT_ROOT)