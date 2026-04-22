import cv2
import numpy as np
import math

# ================= 1. 配置区域 =================
# 替换为你本地任意一张原始 OCT 彩色图片的路径
IMAGE_PATH = r'E:\Cardio_Data\Data1\images\frame_000307.jpg'

# ================= 2. 全局状态变量 =================
# 状态机：'IDLE' (空闲), 'DRAWING' (画新框), 'MOVING' (移动框), 'RESIZING' (缩放框)
state = 'IDLE'

# 正在画新框时的起点坐标
ix, iy = -1, -1

# 当前锁定框的参数：左上角坐标 (cx, cy) 和 边长 (csize)
cx, cy, csize = -1, -1, -1

# 拖动框时的鼠标偏移量
offset_x, offset_y = 0, 0

# 记录当前抓取的是哪个顶点 (TL:左上, TR:右上, BL:左下, BR:右下)
active_corner = None
GRAB_TOLERANCE = 15  # 鼠标靠近顶点多少像素内可以抓取

original_img = None


# ================= 3. 辅助函数 =================
def get_hovered_corner(x, y, cx, cy, csize):
    """检测鼠标是否悬停在四个顶点附近"""
    if csize <= 0: return None
    corners = {
        'TL': (cx, cy),
        'TR': (cx + csize, cy),
        'BL': (cx, cy + csize),
        'BR': (cx + csize, cy + csize)
    }
    for name, (px, py) in corners.items():
        # 利用勾股定理计算距离
        if math.hypot(x - px, y - py) <= GRAB_TOLERANCE:
            return name
    return None


def draw_box_with_handles(img, x, y, size, color=(0, 0, 255), thickness=2):
    """绘制正方形以及四个角落的抓取手柄"""
    cv2.rectangle(img, (int(x), int(y)), (int(x + size), int(y + size)), color, thickness)
    # 绘制四个角的圆形手柄
    corners = [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]
    for (px, py) in corners:
        cv2.circle(img, (int(px), int(py)), 5, (255, 0, 0), -1)  # 蓝色小圆点


# ================= 4. 核心鼠标交互逻辑 =================
def mouse_callback(event, x, y, flags, param):
    global state, ix, iy, cx, cy, csize, offset_x, offset_y, active_corner, original_img

    img_h, img_w = original_img.shape[:2]
    img_display = original_img.copy()

    # --- 1. 鼠标左键按下 ---
    if event == cv2.EVENT_LBUTTONDOWN:
        # 优先检测是否抓住了顶点 (进入缩放模式)
        corner = get_hovered_corner(x, y, cx, cy, csize)
        if corner:
            state = 'RESIZING'
            active_corner = corner
        # 其次检测是否点击在框内部 (进入移动模式)
        elif csize > 0 and (cx <= x <= cx + csize) and (cy <= y <= cy + csize):
            state = 'MOVING'
            offset_x = x - cx
            offset_y = y - cy
        # 否则是在空白处点击 (进入画新框模式)
        else:
            state = 'DRAWING'
            ix, iy = x, y
            cx, cy, csize = -1, -1, -1

    # --- 2. 鼠标移动 ---
    elif event == cv2.EVENT_MOUSEMOVE:
        if state == 'DRAWING':
            side_length = max(abs(x - ix), abs(y - iy))
            dir_x = 1 if x > ix else -1
            dir_y = 1 if y > iy else -1
            temp_x2 = ix + dir_x * side_length
            temp_y2 = iy + dir_y * side_length

            x_min, y_min = min(ix, temp_x2), min(iy, temp_y2)
            cv2.rectangle(img_display, (x_min, y_min), (x_min + side_length, y_min + side_length), (0, 255, 0), 2)
            cv2.imshow('Ultimate Square Cropper', img_display)

        elif state == 'MOVING':
            new_cx = x - offset_x
            new_cy = y - offset_y
            # 碰撞检测：防越界
            cx = max(0, min(new_cx, img_w - csize))
            cy = max(0, min(new_cy, img_h - csize))

            draw_box_with_handles(img_display, cx, cy, csize, color=(0, 255, 255))  # 黄色代表移动中
            cv2.imshow('Ultimate Square Cropper', img_display)

        elif state == 'RESIZING':
            # 缩放逻辑：固定对角锚点，根据鼠标位置计算新边长
            min_size = 20  # 限制最小尺寸

            if active_corner == 'TL':
                anchor_x, anchor_y = cx + csize, cy + csize
                new_s = max(abs(anchor_x - x), abs(anchor_y - y))
                new_s = max(min_size, min(new_s, anchor_x, anchor_y))  # 防越界
                csize, cx, cy = new_s, anchor_x - new_s, anchor_y - new_s

            elif active_corner == 'TR':
                anchor_x, anchor_y = cx, cy + csize
                new_s = max(abs(x - anchor_x), abs(anchor_y - y))
                new_s = max(min_size, min(new_s, img_w - anchor_x, anchor_y))
                csize, cx, cy = new_s, anchor_x, anchor_y - new_s

            elif active_corner == 'BL':
                anchor_x, anchor_y = cx + csize, cy
                new_s = max(abs(anchor_x - x), abs(y - anchor_y))
                new_s = max(min_size, min(new_s, anchor_x, img_h - anchor_y))
                csize, cx, cy = new_s, anchor_x - new_s, anchor_y

            elif active_corner == 'BR':
                anchor_x, anchor_y = cx, cy
                new_s = max(abs(x - anchor_x), abs(y - anchor_y))
                new_s = max(min_size, min(new_s, img_w - anchor_x, img_h - anchor_y))
                csize, cx, cy = new_s, anchor_x, anchor_y

            draw_box_with_handles(img_display, cx, cy, csize, color=(255, 0, 255))  # 紫色代表缩放中
            cv2.imshow('Ultimate Square Cropper', img_display)

        elif state == 'IDLE' and csize > 0:
            # 悬停提示：当鼠标靠近顶点时，画一个高亮圆圈提示可以抓取
            hover_corner = get_hovered_corner(x, y, cx, cy, csize)
            draw_box_with_handles(img_display, cx, cy, csize)
            if hover_corner:
                corners_dict = {'TL': (cx, cy), 'TR': (cx + csize, cy),
                                'BL': (cx, cy + csize), 'BR': (cx + csize, cy + csize)}
                hx, hy = corners_dict[hover_corner]
                cv2.circle(img_display, (int(hx), int(hy)), 8, (0, 255, 0), 2)  # 绿色空心圆高亮
            cv2.imshow('Ultimate Square Cropper', img_display)

    # --- 3. 鼠标左键松开 ---
    elif event == cv2.EVENT_LBUTTONUP:
        if state == 'DRAWING':
            side_length = max(abs(x - ix), abs(y - iy))
            if side_length > 20:  # 忽略极小的误触
                dir_x = 1 if x > ix else -1
                dir_y = 1 if y > iy else -1
                temp_x2 = ix + dir_x * side_length
                temp_y2 = iy + dir_y * side_length
                cx, cy = min(ix, temp_x2), min(iy, temp_y2)
                csize = side_length

                # 越界修剪
                cx, cy = max(0, cx), max(0, cy)
                csize = min(csize, img_w - cx, img_h - cy)
            else:
                cx, cy, csize = -1, -1, -1  # 取消

        state = 'IDLE'
        active_corner = None

        # 刷新并打印
        if csize > 0:
            draw_box_with_handles(img_display, cx, cy, csize)
            cv2.imshow('Ultimate Square Cropper', img_display)
            print_output(cx, cy, csize)


def print_output(x, y, size):
    # 强制转为整型，防止切片报错
    x, y, size = int(x), int(y), int(size)
    print("\n" + "=" * 50)
    print(f"✅ 完美正方形区域已锁定！")
    print(f"📏 边长尺寸: {size} x {size}")
    print(f"✂️ 请在你的预处理代码中使用以下切片:")
    print(f"cropped_img = orig_img[{y}:{y + size}, {x}:{x + size}]")
    print("=" * 50)


# ================= 5. 启动程序 =================
def main():
    global original_img
    original_img = cv2.imread(IMAGE_PATH)

    if original_img is None:
        print("❌ 无法读取图片，请检查路径是否正确！")
        return

    cv2.namedWindow('Ultimate Square Cropper')
    cv2.setMouseCallback('Ultimate Square Cropper', mouse_callback)

    print("👉 终极操作说明：")
    print("1. 【画框】：在空白处按住左键拖动。")
    print("2. 【移动】：按住框内部区域拖动（框变黄）。")
    print("3. 【缩放】：把鼠标放在四个角的蓝点上，按住拖动（框变紫）。")
    print("4. 【退出】：按 'q' 键或 'ESC' 键退出。")

    cv2.imshow('Ultimate Square Cropper', original_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()