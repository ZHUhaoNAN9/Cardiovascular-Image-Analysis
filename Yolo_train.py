import os
import yaml
from ultralytics import YOLO
import numpy

# 1. 获取数据集的绝对路径 (最稳妥的方式)
# 假设您的 final_dataset_for_kaggle 文件夹在当前脚本的同级目录下
current_dir = os.getcwd()
dataset_root_name = r'/root/Pure_Data_82_NoDamage'
dataset_abs_path = os.path.join(current_dir, dataset_root_name)

# 检查一下路径是否存在，避免乌龙
if not os.path.exists(dataset_abs_path):
    print(f"❌ 错误：找不到数据集文件夹: {dataset_abs_path}")
else:
    print(f"✅ 数据集路径锁定: {dataset_abs_path}")

# 2. 动态生成 data.yaml
data_yaml_content = {
    'path': dataset_abs_path,  # 根目录 (使用绝对路径)
    'train': 'images/train',  # 相对路径 (相对于上面的 path)
    'val': 'images/val',  # 相对路径 (相对于上面的 path)

    # 注意：这里务必确认您的 names 和标注时的 ID 是否一致
    # 'names': {0: 'calcification', 1: 'fibre', 2: 'lipid', 3: 'damage'}
    'names': {0: 'calcification', 1: 'fibre', 2: 'lipid'}
}

# yaml 文件保存在数据集根目录下
yaml_save_path = os.path.join(dataset_abs_path, 'data.yaml')

with open(yaml_save_path, 'w') as f:
    # 【修正点】这里要 dump 字典内容(data_yaml_content)，而不是路径字符串
    yaml.dump(data_yaml_content, f, sort_keys=False)

print(f"📄 配置文件已生成: {yaml_save_path}")

# 3. 开始训练
model = YOLO('yolov8m.pt')

# 开始训练
results = model.train(

    data=yaml_save_path,  # 指向 data.yaml

    # === 1. 时间策略：给足上限，交给早停来把控 ===
    epochs=200,  # 上限放宽到 200 轮
    patience=50,  # ⚠️ 修正：这才是真正的开启早停机制！

    # === 2. 硬件榨取 ===
    imgsz=448,
    batch=16,  # 🚀 火力全开：榨干 4080 SUPER 的 16GB 显存

    # === 3. 温和且必备的医学增强 (保持原样，极其优秀) ===
    mosaic=0.0,
    mixup=0.0,
    degrees=15.0,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.0,
    hsv_s=0.1,
    hsv_v=0.1,

    # === 4. 高级优化器策略 ===
    cos_lr=True,
    lr0=0.001,

    project='yolo_pre_exp',
    name='Pure_run_Final_NoDamage'  # 换个新名字，纪念这次严谨的实验
)