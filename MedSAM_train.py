import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
import copy  # 用于保存最佳模型状态

# ================= 1. 全局配置区域 =================
# 确保这里指向你清洗掉 damage 后的 3 分类数据集
DATA_ROOT = '/root/Pure_Data_82_NoDamage'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'images/train')
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'masks/train')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'images/val')  # 新增：验证集图片
VAL_MASK_DIR = os.path.join(DATA_ROOT, 'masks/val')  # 新增：验证集掩膜

CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4  # 32GB 显存建议开到 4


# ================= 2. 数据处理与增强 =================
def augment_data(image, mask):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


class OCTDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 1

        # 仅在训练时进行翻转增强
        if self.is_train:
            image, mask = augment_data(image, mask)

        target_size = 1024
        image = cv2.resize(image, (target_size, target_size))
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # 🌟 优化：引入包含正负偏移的高级框抖动，模拟更真实的 YOLO 误差
            if self.is_train:
                # -10代表框可能会偏小切到斑块，40代表框可能会画得很宽松
                x_min = max(0, x_min - np.random.randint(-10, 40))
                y_min = max(0, y_min - np.random.randint(-10, 40))
                x_max = min(target_size, x_max + np.random.randint(-10, 40))
                y_max = min(target_size, y_max + np.random.randint(-10, 40))
            else:
                # 验证时不加剧烈抖动，或者只加固定的小边距
                x_min = max(0, x_min - 10)
                y_min = max(0, y_min - 10)
                x_max = min(target_size, x_max + 10)
                y_max = min(target_size, y_max + 10)

            box = np.array([x_min, y_min, x_max, y_max])
        else:
            box = np.array([0, 0, target_size, target_size])

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        box_tensor = torch.from_numpy(box).float()

        return image_tensor, mask_tensor, box_tensor


def compute_dice(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# ================= 3. 核心训练逻辑 =================
def train_medsam():
    # 🌟 新增：分别初始化 Train 和 Val 数据加载器
    train_dataset = OCTDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, is_train=True)
    val_dataset = OCTDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🚀 正在初始化 MedSAM 模型...")
    sam_model = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)

    # 深度解冻策略保持不变...
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    for param in sam_model.image_encoder.blocks[11].parameters():
        param.requires_grad = True
    for param in sam_model.image_encoder.neck.parameters():
        param.requires_grad = True
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=1e-5)

    train_loss_history, train_dice_history = [], []
    val_dice_history = []  # 新增：记录验证集分数

    best_val_dice = 0.0
    best_weights = None

    print(f"📈 开始深度微调 (Total Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")

    for epoch in range(EPOCHS):
        # ================= 训练阶段 =================
        sam_model.train()
        epoch_loss = 0
        epoch_dice = 0
        steps = 0

        for image, gt_mask, box in train_loader:
            image, gt_mask, box = image.to(DEVICE), gt_mask.to(DEVICE), box.to(DEVICE)
            if len(box.shape) == 2:
                box = box[:, None, :]

            image_embeddings = sam_model.image_encoder(image)

            low_res_masks_list = []
            for i in range(image.shape[0]):
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box[i:i + 1], masks=None
                )
                low_res_mask, _ = sam_model.mask_decoder(
                    image_embeddings=image_embeddings[i:i + 1],
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                low_res_masks_list.append(low_res_mask)

            low_res_masks = torch.cat(low_res_masks_list, dim=0)

            pred_mask = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
            pred_mask_sigmoid = torch.sigmoid(pred_mask)

            loss_bce = F.binary_cross_entropy(pred_mask_sigmoid, gt_mask)
            pred_binary = (pred_mask_sigmoid > 0.5).float()
            dice_val = compute_dice(pred_binary, gt_mask)

            loss = loss_bce + (1 - dice_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_val.item()
            steps += 1

        avg_train_loss = epoch_loss / steps
        avg_train_dice = epoch_dice / steps
        train_loss_history.append(avg_train_loss)
        train_dice_history.append(avg_train_dice)

        # ================= 验证阶段 =================
        sam_model.eval()
        val_epoch_dice = 0
        val_steps = 0

        with torch.no_grad():
            for image, gt_mask, box in val_loader:
                image, gt_mask, box = image.to(DEVICE), gt_mask.to(DEVICE), box.to(DEVICE)
                if len(box.shape) == 2: box = box[:, None, :]

                image_embeddings = sam_model.image_encoder(image)
                low_res_masks_list = []
                for i in range(image.shape[0]):
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None, boxes=box[i:i + 1], masks=None
                    )
                    low_res_mask, _ = sam_model.mask_decoder(
                        image_embeddings=image_embeddings[i:i + 1],
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    low_res_masks_list.append(low_res_mask)

                low_res_masks = torch.cat(low_res_masks_list, dim=0)
                pred_mask = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_mask_sigmoid = torch.sigmoid(pred_mask)
                pred_binary = (pred_mask_sigmoid > 0.5).float()

                val_epoch_dice += compute_dice(pred_binary, gt_mask).item()
                val_steps += 1

        avg_val_dice = val_epoch_dice / val_steps
        val_dice_history.append(avg_val_dice)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f}")

        # 🌟 核心：保存验证集分数最高的模型权重
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_weights = copy.deepcopy(sam_model.state_dict())
            print(f"   🌟 发现最佳模型！Val Dice 提升至: {best_val_dice:.4f}")

    print("\n💾 训练完成！正在将最佳权重保存到本地...")
    torch.save(best_weights, 'medsam_ultimate_best_Pure.pth')
    print(f"✅ 权重已保存为: medsam_ultimate_best_Pure.pth (Best Val Dice: {best_val_dice:.4f})")

    return train_loss_history, train_dice_history, val_dice_history


# ================= 4. 绘制训练曲线 =================
def plot_training_curves(loss_data, train_dice_data, val_dice_data):
    epochs = range(1, len(loss_data) + 1)

    plt.figure(figsize=(15, 6))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_data, 'b-', label='Training Loss', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 绘制 Dice 曲线 (同时包含 Train 和 Val)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice_data, 'r-', label='Train Dice', linewidth=2, alpha=0.6)
    plt.plot(epochs, val_dice_data, 'g-', label='Val Dice', linewidth=2)
    plt.title('Segmentation Accuracy (Dice)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)

    max_val_dice = max(val_dice_data)
    max_epoch = val_dice_data.index(max_val_dice) + 1
    plt.annotate(f'Best Val: {max_val_dice:.4f}',
                 xy=(max_epoch, max_val_dice),
                 xytext=(max_epoch, max_val_dice - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    plt.legend()

    plt.tight_layout()
    plt.savefig('medsam_ultimate_curves.png', dpi=300)
    print("📊 训练曲线已保存为: medsam_ultimate_curves.png")


if __name__ == "__main__":
    loss_curve, train_dice_curve, val_dice_curve = train_medsam()
    plot_training_curves(loss_curve, train_dice_curve, val_dice_curve)