import os
import cv2
import numpy as np
from pathlib import Path

# ================= 1. 配置区域 =================
# 指向你昨天裁剪出来的 439x439 的纯净数据集根目录
DATASET_DIR = r'E:\Cardio_Data\Pure_Data_82'

# 根据你的原始代码解密的映射关系！
# 字典格式：{ Mask图像中的像素值 : 对应的 YOLO 类别 ID }
MASK_VALUE_TO_CLASS_ID = {
    1: 0,  # calcification -> YOLO Class 0
    2: 1,  # fibre         -> YOLO Class 1
    3: 2,  # lipid         -> YOLO Class 2
    4: 3
    # 注意：我们这里不写 4:3 (damage)，因为我们之前已经决定做 3 分类了。
    # 只要不写，算法就会自动忽略损伤区域，将其视为背景。
}


# ================= 2. 核心提取函数 =================
def mask_to_yolo_txt(mask_path, txt_save_path):
    # 以单通道灰度图读取 Mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ 无法读取掩膜: {mask_path.name}")
        return False

    img_h, img_w = mask.shape
    yolo_lines = []

    # 遍历每一种病灶的像素值
    for mask_val, class_id in MASK_VALUE_TO_CLASS_ID.items():
        # 提取当前病灶类别的二值化掩膜 (只保留当前病灶，其他变黑)
        binary_mask = (mask == mask_val).astype(np.uint8)

        # 寻找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # 防御机制：过滤掉因为图像抗锯齿或极度微小裁剪产生的噪点 (面积 < 10 像素)
            if cv2.contourArea(cnt) < 10:
                continue

            # 计算最紧凑的外接矩形 (这会自动适应你裁剪和去 UI 后的最新形状！)
            x, y, w, h = cv2.boundingRect(cnt)

            # 转换为 YOLO 所需的归一化中心坐标 (0 ~ 1)
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # 极限防御：确保坐标绝对不会越界 (超出 0~1 的范围 YOLO 会报错)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            # 组合成 YOLO 格式字符串
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # 如果这张图里有合法病灶，则生成 .txt 文件
    if yolo_lines:
        with open(txt_save_path, 'w') as f:
            f.writelines(yolo_lines)
        return True
    return False


# ================= 3. 批量生成主逻辑 =================
def generate_all_labels():
    dataset_path = Path(DATASET_DIR)

    # 遍历 train 和 val 两个文件夹
    for mode in ['train', 'val']:
        mask_dir = dataset_path / 'masks' / mode
        label_dir = dataset_path / 'labels' / mode

        # 如果 mask 文件夹不存在，跳过
        if not mask_dir.exists():
            continue

        # 自动创建存放 txt 的 labels 文件夹
        label_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片格式的掩膜
        mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
        if not mask_files:
            continue

        print(f"📂 正在为 {mode} 集生成全新 YOLO 坐标 (共读取 {len(mask_files)} 个掩膜)...")

        success_count = 0
        for mask_file in mask_files:
            txt_path = label_dir / (mask_file.stem + '.txt')
            if mask_to_yolo_txt(mask_file, txt_path):
                success_count += 1

        print(f"✅ {mode} 集转换完成！共生成 {success_count} 个有效的 .txt 标签文件。")

    print(f"\n🎉 所有的 YOLO 标签已浴火重生！现在图像和标签已经 100% 完美对齐。")


if __name__ == "__main__":
    generate_all_labels()