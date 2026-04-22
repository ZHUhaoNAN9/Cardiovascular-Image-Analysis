import cv2
import numpy as np

# ================= 1. 配置区域 =================
# 替换为你本地测试用的原始 OCT 彩色图片路径
IMAGE_PATH = r'E:\Cardio_Data\Data1\images\frame_000307.jpg'

# 你刚才锁定的黄金正方形坐标
CROP_Y1, CROP_Y2 = 0, 439
CROP_X1, CROP_X2 = 129, 568

img_cropped = None


# ================= 2. 鼠标回调：水平标尺逻辑 =================
def mouse_callback(event, x, y, flags, param):
    global img_cropped

    # 每次移动鼠标，都基于干净的裁剪图重新画线
    img_display = img_cropped.copy()

    # 画一条红色的水平游标卡尺线
    cv2.line(img_display, (0, y), (img_display.shape[1], y), (0, 0, 255), 1)
    # 在线旁边显示当前的 Y 坐标值
    cv2.putText(img_display, f"Y Cutoff: {y}", (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('UI Ruler', img_display)

    # 点击左键，锁定坐标并打印代码
    if event == cv2.EVENT_LBUTTONDOWN:
        print("\n" + "=" * 50)
        print(f"🎯 底部 UI 屏蔽线已锁定！")
        print(f"📏 Y 轴切割坐标: {y}")
        print(f"✂️ 请在未来的全自动清洗流水线中，加入这一行代码:")
        print(f"cropped_img[{y}:, :] = 0  # 将 Y={y} 以下的 UI 区域全部涂黑")
        print("=" * 50 + "\n")


# ================= 3. 启动程序 =================
def main():
    global img_cropped
    orig_img = cv2.imread(IMAGE_PATH)

    if orig_img is None:
        print("❌ 无法读取图片，请检查路径是否正确！")
        return

    # 第一步：先执行你的黄金裁剪
    img_cropped = orig_img[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    cv2.namedWindow('UI Ruler')
    cv2.setMouseCallback('UI Ruler', mouse_callback)

    print("👉 操作说明：")
    print("1. 移动鼠标，将红线对准底部 UI 的最上边缘。")
    print("2. 点击左键锁定坐标。")
    print("3. 按 'ESC' 键退出。")

    # 初始化显示
    cv2.imshow('UI Ruler', img_cropped)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # 27 是 ESC 键的 ASCII 码
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()