import os
import shutil
import random
import yaml

# ==================== 请修改这里的路径 ====================
# 1. 你的原图在哪里？(CVAT 导出时解压出来的图片文件夹)
# 也就是包含 frame_000000.png, frame_000001.png... 的那个文件夹
SRC_IMAGES_DIR = r'E:\Cardio_Data\Data18\images'

# 2. 你的生成的标签在哪里？(运行 convert_fix.py 后生成的 my_dataset 文件夹)
SRC_LABELS_DIR = r'E:\Cardio_Data\Data18\my_dataset\labels'
SRC_MASKS_DIR = r'E:\Cardio_Data\Data18\my_dataset\masks'

# 3. 你想把整理好的数据集存到哪里？(最后打包上传这个文件夹)
OUTPUT_DIR = r'E:\Cardio_Data\Data18\final_dataset_for_kaggle'

# 4. 你的类别名称 (必须和 convert_fix.py 里的 CLASS_MAP 一致)
CLASS_NAMES = ['calcification', 'fibre', 'lipid', 'damage']


# ========================================================

def organize_dataset():
    # 1. 创建目标文件夹结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'masks', split), exist_ok=True)

    # 2. 获取所有有标签的文件名 (以 .txt 为准)
    # 因为有些图可能没标，我们只取有标签的图
    files = [f for f in os.listdir(SRC_LABELS_DIR) if f.endswith('.txt')]
    random.seed(42)  # 保证每次切分结果一样
    random.shuffle(files)

    # 3. 切分训练集和验证集 (80% : 20%)
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    print(f"📊 共找到 {len(files)} 组数据。训练集: {len(train_files)}, 验证集: {len(val_files)}")

    # 4. 移动文件的函数
    def move_files(file_list, split_name):
        for filename in file_list:
            base_name = os.path.splitext(filename)[0]

            # A. 复制 YOLO 标签 (.txt)
            src_txt = os.path.join(SRC_LABELS_DIR, filename)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_name, filename)
            shutil.copy(src_txt, dst_txt)

            # B. 复制 MedSAM 掩膜 (.png)
            # 注意：mask 是 png 格式
            src_mask = os.path.join(SRC_MASKS_DIR, base_name + '.png')
            dst_mask = os.path.join(OUTPUT_DIR, 'masks', split_name, base_name + '.png')
            if os.path.exists(src_mask):
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"⚠️ 警告: 找不到对应的 Mask: {src_mask}")

            # C. 复制 原图 (.png 或 .jpg)
            # 尝试寻找对应的图片文件
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                src_img = os.path.join(SRC_IMAGES_DIR, base_name + ext)
                if os.path.exists(src_img):
                    dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, base_name + ext)
                    shutil.copy(src_img, dst_img)
                    image_found = True
                    break

            if not image_found:
                print(f"❌ 错误: 找不到对应的原图: {base_name}")

    print("🚀 正在复制文件...")
    move_files(train_files, 'train')
    move_files(val_files, 'val')

    # 5. 生成 data.yaml (YOLO 需要)
    yaml_content = {
        'path': '../input/final-dataset/final_dataset_for_kaggle',  # Kaggle 上的挂载路径 (预测值)
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }

    # 同时也生成一个本地用的绝对路径 yaml，方便你在本地测试
    yaml_local = yaml_content.copy()
    yaml_local['path'] = OUTPUT_DIR

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"✅ 整理完成！数据集已保存在: {OUTPUT_DIR}")
    print(f"📄 已生成配置文件: {os.path.join(OUTPUT_DIR, 'data.yaml')}")
    print("👉 请将这个文件夹压缩为 zip，上传到 Kaggle Dataset。")


if __name__ == "__main__":
    organize_dataset()