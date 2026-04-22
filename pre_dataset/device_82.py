import os
import yaml
import shutil
from pathlib import Path

# ================= 1. 核心配置区域 =================
ALL_PATIENTS = [
    r'E:\Cardio_Data\Orginal_Data\Data1',
    r'E:\Cardio_Data\Orginal_Data\Data9',
    r'E:\Cardio_Data\Orginal_Data\Data12',
    r'E:\Cardio_Data\Orginal_Data\Data18'
]

OUTPUT_DIR = r'E:\Cardio_Data\Chunked_82_Dataset_NoCrop'
CLASS_NAMES = ['calcification', 'fibre', 'lipid', 'damage']

# 分块采样配置
CHUNK_SIZE = 25
BUFFER_SIZE = 2


# ================= 2. 内部函数 =================
def setup_directories(base_dir):
    paths = {
        'img_train': Path(base_dir) / 'images' / 'train', 'img_val': Path(base_dir) / 'images' / 'val',
        'lbl_train': Path(base_dir) / 'labels' / 'train', 'lbl_val': Path(base_dir) / 'labels' / 'val',
        'msk_train': Path(base_dir) / 'masks' / 'train', 'msk_val': Path(base_dir) / 'masks' / 'val'
    }
    if Path(base_dir).exists(): shutil.rmtree(base_dir)
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths


def process_image_set(file_list, split_name, patient_id, lbl_dir, msk_dir, out_paths):
    count = 0
    for img_path in file_list:
        base_name = img_path.stem
        new_base_name = f"{patient_id}_{base_name}"

        # 定义源路径和目标路径
        src_lbl = lbl_dir / f"{base_name}.txt"
        src_msk = msk_dir / f"{base_name}.png"

        if not src_lbl.exists(): continue

        # 1. 拷贝图像
        shutil.copy2(img_path, out_paths[f'img_{split_name}'] / f"{new_base_name}{img_path.suffix}")

        # 2. 拷贝标签 (不进行裁剪计算，直接拷贝)
        shutil.copy2(src_lbl, out_paths[f'lbl_{split_name}'] / f"{new_base_name}.txt")

        # 3. 拷贝 Mask (如果存在)
        if src_msk.exists():
            shutil.copy2(src_msk, out_paths[f'msk_{split_name}'] / f"{new_base_name}.png")

        count += 1
    return count


# ================= 3. 主流程 =================
if __name__ == "__main__":
    out_paths = setup_directories(OUTPUT_DIR)
    t_train, t_val, t_drop = 0, 0, 0

    print(f"🚀 开始进行 8:2 分块分类 (原始尺寸)...")

    for folder_path in ALL_PATIENTS:
        folder = Path(folder_path)
        img_dir, lbl_dir, msk_dir = folder / 'images', folder / 'my_dataset' / 'labels', folder / 'my_dataset' / 'masks'

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"⚠️ 跳过 {folder.name}: 找不到图像或标签文件夹")
            continue

        img_files = sorted([f for f in img_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

        train_list, val_list = [], []
        for i in range(0, len(img_files), CHUNK_SIZE):
            chunk = img_files[i: i + CHUNK_SIZE]
            chunk_idx = i // CHUNK_SIZE

            # 8:2 逻辑：每 5 个块中，前 4 个为训练，第 5 个为验证
            if chunk_idx % 5 == 4:
                if len(chunk) > BUFFER_SIZE * 2:
                    val_list.extend(chunk[BUFFER_SIZE: -BUFFER_SIZE])
                    t_drop += (BUFFER_SIZE * 2)
                else:
                    t_drop += len(chunk)
            else:
                train_list.extend(chunk)

        t_train += process_image_set(train_list, 'train', folder.name, lbl_dir, msk_dir, out_paths)
        t_val += process_image_set(val_list, 'val', folder.name, lbl_dir, msk_dir, out_paths)
        print(f"📦 病人 {folder.name} 处理完毕")

    # 生成 YAML
    yaml_data = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: n for i, n in enumerate(CLASS_NAMES)}
    }
    with open(Path(OUTPUT_DIR) / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print("\n" + "=" * 30)
    print(f"✅ 处理完成！")
    print(f"📊 训练集: {t_train}")
    print(f"📊 验证集: {t_val}")
    print(f"🛡️ 缓冲区丢弃: {t_drop}")
    print("=" * 30)