from PIL import Image, ImageDraw
import onnxruntime as ort
import numpy as np
import random

class DetectONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (640, 640)
        self.image = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.3
        self.magnification = 1

    def preprocess(self, image):
        origin_shape = image.size
        width, height = origin_shape
        max_l = max(width, height)
        if(max_l > 2880): self.input_size = (1280, 1280)

        resized_image = image.resize(self.input_size)
        np_image = np.array(resized_image).astype(np.float32) / 255.0
        np_image = np.transpose(np_image, (2, 0, 1))
        np_image = np.expand_dims(np_image, axis=0)

        return np_image, origin_shape

    def postprocess(self, output, image_shape):
        output = output[0][0]
        predictions = output.T

        boxes = []
        confidences = []

        for pred in predictions:
            x, y, w, h, conf = pred
            if conf < self.conf_threshold:
                continue

            left = int((x - w / 2) * image_shape[1] / self.input_size[0])
            top = int((y - h / 2) * image_shape[0] / self.input_size[1])
            width = int(w * image_shape[1] / self.input_size[0])
            height = int(h * image_shape[0] / self.input_size[1])

            left = max(0, left)
            top = max(0, top)
            right = min(left + width, image_shape[0])
            bottom = min(top + height, image_shape[1])

            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                continue

            boxes.append([left, top, width, height])
            confidences.append(float(conf))

        indices = self.non_max_suppression(boxes, confidences, self.iou_threshold)

        result_boxes = []
        for i in indices:
            box = boxes[i]
            if(box[2] >= 40 and box[3] >= 40):
                result_boxes.append({
                    'box': box,
                    'score': confidences[i],
                })
        return result_boxes

    def infer(self, image):
        input_tensor, orig_shape = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes = self.postprocess(outputs, orig_shape)
        return boxes

    def non_max_suppression(self, boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return []

        # xywh2xyxy
        boxes = np.array([
            [b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes
        ], dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def draw(self):
        csvList = []
        if isinstance(self.image, np.ndarray):
            img = Image.fromarray(self.image)
            image = Image.fromarray(self.image)
        else:
            if (self.magnification != 1):
                scale = self.magnification
                o_w, o_h = self.image.size
                self.image = self.image.resize((int(o_w / scale), int(o_h / scale)), Image.BILINEAR)
            img = self.image.copy()
        resultBoxes = self.infer(image)
        csvList.append(len(resultBoxes))

        draw = ImageDraw.Draw(img)
        min_side = min(img.size)
        # line_width = max(1, min_side // 333)
        line_width = 3
        color = tuple(random.randint(0, 255) for _ in range(3))

        for item in resultBoxes:
            x, y, w, h = item['box']
            draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
        img = np.array(img)
        return img, csvList

    def crop(self):
        img = self.image.copy()
        if isinstance(self.image, np.ndarray):
            image = Image.fromarray(self.image)
        resultBoxes = self.infer(image)

        crops = []
        for box in resultBoxes:
            x, y, w, h = map(int, box['box'])
            crop = img[y:y + h, x:x + w].copy()
            crops.append(crop)
        return crops, resultBoxes






