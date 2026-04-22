import cv2
import os
import numpy as np  # 需要导入 numpy


def extract_frames(video_path, output_folder, file_prefix="frame_", extension=".jpg"):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建文件夹: {output_folder}")

    # 读取视频
    # 注意：cv2.VideoCapture 有时对中文路径也不友好，如果读取失败，也需要类似处理
    # 但根据你的日志，视频似乎读取成功了（算出了375帧），所以这里先不动
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件，请尝试将视频路径改为纯英文")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"开始处理，视频总帧数: {total_frames}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        file_name = f"{file_prefix}{str(frame_count).zfill(6)}{extension}"
        output_path = os.path.join(output_folder, file_name)

        # --- 修改开始：使用支持中文路径的保存方式 ---
        try:
            # 1. 先对图片进行编码
            # 2. 转换为 numpy 数组
            # 3. 使用 tofile 写入文件
            cv2.imencode(extension, frame)[1].tofile(output_path)
        except Exception as e:
            print(f"保存失败: {output_path}, 错误: {e}")
        # --- 修改结束 ---

        if frame_count % 100 == 0:
            print(f"已处理: {frame_count}/{total_frames}")

        frame_count += 1

    cap.release()
    print(f"处理完成！请检查文件夹: {output_folder}")


# --- 使用示例 ---
video_file = r"E:\Cardio_Data\Video_Datasets\RAW\18\18.mp4"  # 确保这里也是你的实际视频路径
save_dir = r"E:\Cardio_Data\Data18\images"  # 保持你的中文路径

extract_frames(video_file, save_dir, file_prefix="frame_")