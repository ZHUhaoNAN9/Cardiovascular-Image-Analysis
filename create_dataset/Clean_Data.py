import os
import shutil
from pathlib import Path


def sync_dataset(img_dir, txt_dir, img_extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    """
    核对图片和txt文件夹，并交互式将不匹配的文件移动到特定的回收站文件夹中。
    """
    img_path = Path(img_dir)
    txt_path = Path(txt_dir)

    if not img_path.exists() or not txt_path.exists():
        print("错误：输入的文件夹路径不存在，请检查路径。")
        return

    # 定义回收站主文件夹，默认放在图片文件夹的上一级目录中
    trash_base_dir = img_path.parent / "trash_pic_txt"
    trash_pic_dir = trash_base_dir / "trash_pic"
    trash_txt_dir = trash_base_dir / "trash_txt"

    print(f"正在扫描文件夹...\n 图片目录: {img_dir}\n 标签目录: {txt_dir}")
    print("-" * 50)

    # 1. 获取所有文件的 {文件名(不含后缀) : 完整路径}
    img_files = {f.stem: f for f in img_path.iterdir() if f.suffix.lower() in img_extensions}
    txt_files = {f.stem: f for f in txt_path.iterdir() if f.suffix.lower() == '.txt'}

    img_stems = set(img_files.keys())
    txt_stems = set(txt_files.keys())

    # 2. 找出差异
    images_without_txt = img_stems - txt_stems
    txt_without_images = txt_stems - img_stems

    # --- 处理情况 A: 图片多余 ---
    if images_without_txt:
        print(f"\n发现 {len(images_without_txt)} 张图片缺少对应的 .txt 文件：")
        for i, stem in enumerate(list(images_without_txt)[:10]):
            print(f" - {img_files[stem].name}")
        if len(images_without_txt) > 10:
            print(f" ... 以及其他 {len(images_without_txt) - 10} 个文件")

        user_input = input(
            f"\n>>> 是否将这 {len(images_without_txt)} 张多余的图片移动到回收站？(输入 y 确认移动，其他键跳过): ").lower()

        if user_input == 'y':
            # 创建对应的回收站文件夹 (如果不存在)
            trash_pic_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for stem in images_without_txt:
                try:
                    # 使用 shutil.move 移动文件
                    src_file = img_files[stem]
                    dst_file = trash_pic_dir / src_file.name
                    shutil.move(str(src_file), str(dst_file))
                    count += 1
                except Exception as e:
                    print(f"移动失败 {stem}: {e}")
            print(f"已移动 {count} 张图片到:\n {trash_pic_dir}")
        else:
            print("已跳过移动图片。")
    else:
        print("\n✔ 所有图片都有对应的 .txt 文件。")

    print("-" * 30)

    # --- 处理情况 B: Txt 多余 ---
    if txt_without_images:
        print(f"\n发现 {len(txt_without_images)} 个 .txt 文件缺少对应的图片：")
        for i, stem in enumerate(list(txt_without_images)[:10]):
            print(f" - {txt_files[stem].name}")
        if len(txt_without_images) > 10:
            print(f" ... 以及其他 {len(txt_without_images) - 10} 个文件")

        user_input = input(
            f"\n>>> 是否将这 {len(txt_without_images)} 个多余的 .txt 文件移动到回收站？(输入 y 确认移动，其他键跳过): ").lower()

        if user_input == 'y':
            # 创建对应的回收站文件夹 (如果不存在)
            trash_txt_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for stem in txt_without_images:
                try:
                    # 使用 shutil.move 移动文件
                    src_file = txt_files[stem]
                    dst_file = trash_txt_dir / src_file.name
                    shutil.move(str(src_file), str(dst_file))
                    count += 1
                except Exception as e:
                    print(f"移动失败 {stem}: {e}")
            print(f"已移动 {count} 个 .txt 文件到:\n {trash_txt_dir}")
        else:
            print("已跳过移动 .txt 文件。")
    else:
        print("\n✔ 所有 .txt 文件都有对应的图片。")

    print("\n" + "=" * 50)
    print("核对与清洗完成！")


# ---在此处修改你的路径---
# 示例：
image_folder = r"E:\Cardio_Data\Merged_Dataset_for_Kaggle\images"
txt_folder = r"E:\Cardio_Data\Merged_Dataset_for_Kaggle\labels"

# 运行函数
sync_dataset(image_folder, txt_folder)