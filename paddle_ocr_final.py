from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
from datetime import datetime

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 常量定义
VIDEO_PATH = 'video/1.2.mp4'
ENDING_LOG_KEYWORDS = ['中国中央电视台', 'www.cctv.com']


# 初始化OCR
def initialize_ocr() -> PaddleOCR:
    return PaddleOCR(use_angle_cls=True, use_gpu=False, show_log=False)


# 处理log图像区域并进行OCR识别
def process_image_region(frame, ocr) -> list:
    height, width, _ = frame.shape
    x_start, x_end = int(width / 3), int(2 * width / 3)
    y_start, y_end = int(3 * height / 4), int(5 * height / 6)
    roi = frame[y_start:y_end, x_start:x_end]
    result = ocr.ocr(roi)
    return result


# 保存帧到文件夹
def save_frame(frame, save_folder, frame_count) -> None:
    save_path = os.path.join(save_folder, f"frame_{frame_count}.png")
    cv2.imwrite(save_path, frame)
    print(f"Saved frame to: {save_path}")


# draw_ocr画图
def draw_ocr_results(image, result) -> None:
    boxes = []
    txts = []
    scores = []

    for line in result:
        for line1 in line:
            # 坐标还原，框出log中的文字需要还原坐标
            line1[0] = [[x + 640, y + 810] for x, y in line1[0]]

            # 获取OCR结果的相关信息
            boxes.append(line1[0])
            txts.append(line1[1][0])
            scores.append(line1[1][1])

    # 在图像上画出OCR结果
    draw_ocr(image, boxes, txts, scores)


# 主处理循环
def process_video(video_path, save_folder, ocr) -> None:
    video_capture = None
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise Exception('Failed to open video')
    except Exception as e:
        print(f'Error: {e}')
        exit(-1)

    frame_count = 0
    start_time = datetime.now()

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame is not None:
            # 处理log图像区域并进行OCR识别
            result = process_image_region(frame, ocr)

            if result == [None]:
                frame_count += 1
                continue

            # ocr识别结果包含坐标、文字和置信率。获取文字列表target
            target = [line1[1][0] for line in result for line1 in line]

            print(f'第{frame_count}帧, 识别到{target}')

            if any(keyword in target for keyword in ENDING_LOG_KEYWORDS):
                save_frame(frame, save_folder, frame_count)

                # 将相关信息传递给draw_ocr_results函数
                draw_ocr_results(frame, result)
                cv2.imshow("result", frame)
                cv2.waitKey(1)  # 1毫秒的等待时间，避免阻塞

                # # 按 'q' 键退出循环
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f'average time per frame: {elapsed_time / frame_count}')


if __name__ == "__main__":
    # 初始化OCR
    ocr_instance = initialize_ocr()

    # 获取当前日期
    current_date = datetime.now().strftime("%m%d%H%M")

    # 创建保存图像的文件夹
    save_folder = f'save_{current_date}'
    os.makedirs(save_folder, exist_ok=True)

    # 处理视频
    process_video(VIDEO_PATH, save_folder, ocr_instance)
