import os
import cv2
import numpy as np
from pathlib import Path

# ================= 1. 配置区域 =================
# 【待修改】你的原始彩色数据集的根目录
INPUT_DIR = r'E:\Cardio_Data\Chunked_82_Dataset_NoCrop'

# 【待修改】生成的终极纯净彩色数据集的输出目录 (用来重新训练)
OUTPUT_DIR = r'E:\Cardio_Data\Pure_Data_82'

# 我们的“黄金物理参数”
CROP_Y1, CROP_Y2 = 0, 439
CROP_X1, CROP_X2 = 129, 568
UI_CUTOFF_Y = 362


# ================= 2. 核心手术函数 =================
def perform_surgery(image_path, is_mask=False):
    """
    对单张图像或掩膜执行精确裁剪和物理屏蔽
    """
    # 掩膜必须以灰度图读取，彩色原图以 BGR 读取
    if is_mask:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        return None

    # 第一刀：切出 439x439 的黄金正方形
    cropped_img = img[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    # 防御性检查：确保切出来的尺寸是对的
    if cropped_img.shape[0] != (CROP_Y2 - CROP_Y1) or cropped_img.shape[1] != (CROP_X2 - CROP_X1):
        print(f"⚠️ 警告: {image_path.name} 尺寸异常，跳过。")
        return None

    # 第二刀：一刀切除底部 UI 面板 (变成纯黑 0)
    # 无论是 3 通道的彩色图还是单通道的掩膜，这行代码都能完美生效
    cropped_img[UI_CUTOFF_Y:, :] = 0

    return cropped_img


# ================= 3. 批量处理主逻辑 =================
def process_dataset():
    input_base = Path(INPUT_DIR)
    output_base = Path(OUTPUT_DIR)

    print(f"🚀 启动终极数据重塑流水线...")
    print(f"📏 裁剪参数: X[{CROP_X1}:{CROP_X2}], Y[{CROP_Y1}:{CROP_Y2}]")
    print(f"🎯 屏蔽参数: 涂黑 Y >= {UI_CUTOFF_Y} 的所有区域")

    # 遍历四大核心文件夹
    sub_dirs = [
        ('images/train', False),
        ('images/val', False),
        ('masks/train', True),
        ('masks/val', True)
    ]

    for sub_dir, is_mask in sub_dirs:
        input_folder = input_base / sub_dir
        output_folder = output_base / sub_dir

        if not input_folder.exists():
            print(f"⚠️ 文件夹 {input_folder} 不存在，已跳过。")
            continue

        output_folder.mkdir(parents=True, exist_ok=True)
        files = sorted([f for f in input_folder.glob('*') if f.suffix.lower() in ['.jpg', '.png']])

        if not files: continue

        print(f"📂 正在处理 {sub_dir} ({len(files)} 个文件)...")
        success_count = 0

        for file_path in files:
            processed_img = perform_surgery(file_path, is_mask)

            if processed_img is not None:
                # 掩膜强烈建议存为 PNG 防止压缩改变标签值
                save_ext = '.png' if is_mask else file_path.suffix
                save_path = output_folder / (file_path.stem + save_ext)
                cv2.imwrite(str(save_path), processed_img)
                success_count += 1

        print(f"✅ 完成! 成功处理 {success_count}/{len(files)} 个文件。")

    print(f"\n🎉 终极纯净彩色数据集生成完毕！")
    print(f"📂 请立即将模型训练路径指向: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    process_dataset()