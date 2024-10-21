import cv2
import numpy as np
import torch
from utils.general import scale_coords
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import Annotator
from utils.fanzhuan import detect_and_measure

def load_model(weights=r'E:\8.8\yolov5-master\runs\train\exp14\weights\best.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    return model


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiples constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def detect_objects(model, source, img_size=640, conf_thres=0.25, iou_thres=0.45):
    img0 = cv2.imread(source)  # BGR
    assert img0 is not None, 'Image Not Found ' + source

    # Resize and pad image
    img, ratio, (dw, dh) = letterbox(img0, new_shape=img_size)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Convert to torch tensor
    img = torch.from_numpy(img).to(model.device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # add batch dimension

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    detections = []
    annotator = Annotator(img0, line_width=3, example=str(model.names))
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=[255, 0, 0])

                # Measure object size
                # 将张量从 GPU 复制到 CPU
                xyxy_cpu = tuple(int(coord.cpu()) for coord in xyxy)
                width, height = measure_object_size(img0, xyxy_cpu)
                detections.append((label, width, height))

    # 直接从 annotator 获取标注后的图像
    annotated_img = annotator.img_result()  # 注意这里可能需要根据实际实现来调整
    return annotated_img, detections


def measure_object_size(image, bbox):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    width, height = rect[1]
    return width, height


def main():
    weights = r'E:\8.8\yolov5-master\runs\train\exp14\weights\best.pt'  # or your custom model path
    source = r'E:\8.8\yolov5-master\R-C.jpg'

    model = load_model(weights)
    annotated_img, detections = detect_objects(model, source)

    if detections:
        for label, width, height in detections:
            print(f"{label}, 宽度: {width} 像素, 高度: {height} 像素")
        cv2.imshow('Detection', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("没有检测到任何物体。")


if __name__ == '__main__':
    main()