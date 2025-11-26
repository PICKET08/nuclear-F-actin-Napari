from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnxruntime as ort

class SegmentONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.threshold = 0.4
        self.magnification = 1
        self.input_name = self.session.get_inputs()[0].name
        self.image = None

    def preprocess(self, image):
        np_image = np.array(image).astype(np.float32) / 255.0
        np_image = (np_image - 0.5) / 0.5  # Normalize to [-1, 1]
        np_image = np.transpose(np_image, (2, 0, 1))  # HWC → CHW
        np_image = np.expand_dims(np_image, axis=0)  # CHW → NCHW
        return np_image

    def postprocess(self, output):
        output_sigmoid = 1 / (1 + np.exp(-output))  # apply sigmoid
        mask = (output_sigmoid[0, 0] > self.threshold).astype(np.uint8) * 255
        return mask

    def infer(self, images):
        if not isinstance(images, list):
            images = [images]

        results = []
        for img in images:
            input_tensor = self.preprocess(img)
            output = self.session.run(None, {self.input_name: input_tensor})[0]
            mask = self.postprocess(output)
            results.append(mask)

        return results


    def draw(self, images, Boxes):
        csvList = []
        masks = self.infer(images)
        image = self.image.copy()
        image_pil = Image.fromarray(image).convert("RGBA")

        if (self.magnification != 1):
            scale = self.magnification
            o_w, o_h = image_pil.size
            image_pil = image_pil.resize((int(o_w / scale), int(o_h / scale)), Image.BILINEAR)

        draw = ImageDraw.Draw(image_pil)

        color = (0, 0, 255)
        alpha = 120
        min_side = min(image_pil.size)
        # line_width = max(1, min_side // 333)
        line_width = 4
        for mask, box in zip(masks, Boxes):
            x, y, w, h = map(int, box['box'])

            mask_pil = Image.fromarray(mask).convert("L")
            mask_overlay = Image.new("RGBA", (w, h), color + (0,))
            mask_overlay.putalpha(mask_pil.point(lambda p: alpha if p > 0 else 0))
            image_pil.paste(mask_overlay, (x, y), mask_overlay)

            draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        intervals = [(0.0, 0.008), (0.008, 0.05), (0.05, 0.1),
                     (0.1, 0.2),   (0.2, 0.3),    (0.3, 0.4),
                     (0.4, 0.5),   (0.5, 1)]
        csvList = [0] * 9
        csvList[0] = len(Boxes)
        for image, mask, box in zip(images, masks, Boxes):
            x, y, w, h = map(int, box['box'])

            R, G, B = image[..., 0], image[..., 1], image[..., 2]
            gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
            binary = np.zeros_like(gray, dtype=np.uint8)
            binary[gray > np.mean(gray) * 0.8] = 255
            percent = np.count_nonzero(mask == 255) / np.count_nonzero(binary == 255)
            label = f"{percent * 100:.1f}%"

            font = ImageFont.truetype("arialbd.ttf", 32)

            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x + 2
            text_y = y + 2

            margin = 2
            draw.rectangle(
                [text_x - margin, text_y - margin, text_x + text_width + margin, text_y + text_height + margin],
                fill=color
            )
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
            for i, (start, end) in enumerate(intervals):
                if(start <= percent < end):
                    csvList[i + 1] += 1
                    break
        csvList.insert(1, csvList[0] - csvList[1])
        csvList.insert(3, csvList[1] / csvList[0])

        img = np.array(image_pil.convert("RGB"))

        return img, csvList


