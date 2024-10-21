

#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from ultralytics.utils.plotting import save_one_box  # 假设save_one_box位于此模块中
# from pathlib import Path
#
#
# def load_model(model_path):
#     # 加载模型权重
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#     model.eval()  # 设置模型为评估模式
#     return model
#
#
# def draw_boxes_and_calculate_dimensions(frame, model, save_crops=True):
#     # 将 OpenCV 图像转换为 PIL 图像
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # 使用模型进行推理
#     results = model(pil_image, size=640)  # 设置图像大小
#
#     # 获取检测结果
#     predictions = results.xyxy[0].tolist()
#
#     # 创建一个新的图像绘图对象
#     frame_copy = frame.copy()
#
#     # 遍历每一个检测结果
#     for idx, pred in enumerate(predictions):
#         # pred 格式为 [x1, y1, x2, y2, confidence, class_id]
#         x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
#         confidence = pred[4]
#         class_id = int(pred[5])
#
#         # 在帧上绘制边界框
#         cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         # 计算边界框的宽度和高度
#         width = x2 - x1
#         height = y2 - y1
#
#         # 构造标签字符串
#         label = (
#             f"Class ID: {class_id}, Confidence: {confidence:.2f}, "
#             f"Coordinates: (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2}), "
#             f"Width: {width} pixels, Height: {height} pixels"
#         )
#
#         # 在边界框上方写入类别ID、置信度、坐标、宽度和高度
#         text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#         text_x = max(x1, frame.shape[1] - text_size[0] - 10)
#         text_y = max(y1 - 10, text_size[1] + 10)
#
#         # 绘制背景矩形以提高可读性
#         cv2.rectangle(frame_copy, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), -1)
#         cv2.putText(frame_copy, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#         # 打印坐标、置信度、宽度和高度
#         print(label)
#
#         # # 保存裁剪的图像
#         # if save_crops:
#         #     file_name = f'crop_{idx}.jpg'
#         #     cropped_img = save_one_box(torch.tensor([x1, y1, x2, y2]), np.array(pil_image), file=Path(file_name))
#
#     return frame_copy
#
#
# # 加载预训练的YOLOv5模型
# model_path = r'E:\8.8\yolov5-master\runs\train\exp15\weights\best.pt'
# model = load_model(model_path)
#
# # 指定摄像头设备，一般内置摄像头的索引是0
# camera_index = 0
#
# # 读取摄像头
# cap = cv2.VideoCapture(camera_index)
#
# if not cap.isOpened():
#     print("无法打开摄像头")
# else:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("无法获取帧")
#             break
#
#         # 进行物体检测并绘制边界框
#         annotated_frame = draw_boxes_and_calculate_dimensions(frame, model)
#
#         # 显示处理后的帧
#         cv2.imshow('Camera Stream', annotated_frame)
#
#         # 按 Q 键退出循环
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# # 释放资源并关闭所有窗口
# cap.release()
# cv2.destroyAllWindows()


#
# import cv2
# import torch
# from yolov5.models.common import AutoShape  # 假设你需要 AutoShape 来处理任意尺寸的输入
# from yolov5.models.yolo import Model  # 假设你需要 Model 类来构建模型
#
# def load_model(model_path):
#     # 加载模型权重
#     state_dict = torch.load(model_path)
#
#     # 构建模型架构
#     # 注意这里需要指定一个配置文件或者直接构建模型结构
#     # 这里假设你有一个配置文件，例如 'yolov5s.yaml'
#     # 如果你知道具体的模型结构，也可以直接创建
#     model = Model(cfg=r'E:\8.8\yolov5-master\data\myvoc.yaml')  # 请替换为实际的配置文件路径
#
#     # 将权重加载到模型中
#     model.load_state_dict(state_dict, strict=False)  # strict=False 忽略大小不匹配的层
#
#     # 如果需要的话，添加 AutoShape 处理
#     model = AutoShape(model)
#
#     return model.eval()  # 设置模型为评估模式
#
# # 加载预训练的YOLOv5模型
# model_path = r'E:\8.8\yolov5-master\runs\train\exp17\weights\custom_model_two.pt'
# model = load_model(model_path)
#
# # 设置图像路径
# image_path = r"F:\image_in\Image_20240905105823067.bmp"
#
# cv2.draw_boxes(image_path, model)
#


# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def load_model(model_path):
#     # 加载模型权重
#     state_dict = torch.load(model_path)
#
#     # 获取模型架构
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#
#     return model
#
#
# def draw_boxes(image_path, model):
#     # 加载图像
#     img = Image.open(image_path)
#
#     # 使用模型进行推理
#     results = model(img, size=640)  # 设置图像大小
#
#     # 获取检测结果
#     predictions = results.xyxy[0].tolist()
#
#     # 创建一个新的图像绘图对象
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#
#     # 遍历每一个检测结果
#     for pred in predictions:
#         # pred 格式为 [x1, y1, x2, y2, confidence, class_id]
#         x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
#         confidence = pred[4]
#         class_id = int(pred[5])
#
#         # 绘制边界框
#         rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
#         ax.add_patch(rect)
#
#         # 打印坐标和置信度
#         print(
#             f"Class ID: {class_id}, Confidence: {confidence:.2f}, Coordinates: (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2})")
#
#     plt.show()
#
#
# # 加载预训练的YOLOv5模型
# model_path = r'E:\8.8\yolov5-master\runs\train\exp17\weights\custom_model.pt'
# model = load_model(model_path)
#
# # 设置图像路径
# image_path = r"F:\image_in\Image_20240905105823067.bmp"
#
# draw_boxes(image_path, model)

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.utils.plotting import save_one_box  # 假设save_one_box位于此模块中
from pathlib import Path


def load_model(model_path):
    # 加载模型权重
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.eval()  # 设置模型为评估模式
    return model


def draw_boxes_and_calculate_dimensions(image, model, save_crops=True):
    # 将 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 使用模型进行推理
    results = model(pil_image, size=640)  # 设置图像大小

    # 获取检测结果
    predictions = results.xyxy[0].tolist()

    # 创建一个新的图像绘图对象
    image_copy = image.copy()

    # 遍历每一个检测结果
    for idx, pred in enumerate(predictions):
        # pred 格式为 [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        confidence = pred[4]
        class_id = int(pred[5])

        # 在帧上绘制边界框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 计算边界框的宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 构造标签字符串
        label = (
            f"Class ID: {class_id}, Confidence: {confidence:.2f}, "
            f"Coordinates: (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2}), "
            f"Width: {width} pixels, Height: {height} pixels"
        )

        # 在边界框上方写入类别ID、置信度、坐标、宽度和高度
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = max(x1, image.shape[1] - text_size[0] - 10)
        text_y = max(y1 - 10, text_size[1] + 10)

        # 绘制背景矩形以提高可读性
        cv2.rectangle(image_copy, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), -1)
        cv2.putText(image_copy, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 打印坐标、置信度、宽度和高度
        print(label)

        # 保存裁剪的图像
        if save_crops:
            file_name = f'crop_{idx}.jpg'
            cropped_img = save_one_box(torch.tensor([x1, y1, x2, y2]), np.array(pil_image), file=Path(file_name))

    return image_copy


# 加载预训练的YOLOv5模型
model_path = r'E:\8.8\yolov5-master\runs\train\exp17\weights\custom_model_three.pt'
model = load_model(model_path)

# 指定图片路径
image_path = r'F:\image_in\Image_20240905105823067.bmp'

# 读取图片
image = cv2.imread(image_path)

if image is None:
    print("无法打开图片")
else:
    # 进行物体检测并绘制边界框
    annotated_image = draw_boxes_and_calculate_dimensions(image, model)

    # 显示处理后的帧
    cv2.imshow('Annotated Image', annotated_image)

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()