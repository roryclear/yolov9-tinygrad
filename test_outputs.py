import sys
from yolov9 import preprocess, YOLOv9, SIZES, load_state_dict, safe_load, scale_boxes, draw_bounding_boxes_and_save
import time
from tinygrad import Tensor
from tinygrad.helpers import fetch
import cv2
import numpy as np
from pathlib import Path

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Error: Image URL or path not provided.")
    sys.exit(1)

  img_path = "https://www.aljazeera.com/wp-content/uploads/2022/10/2022-04-28T192650Z_1186456067_UP1EI4S1I0P14_RTRMADP_3_SOCCER-ENGLAND-MUN-CHE-REPORT.jpg"
  for yolo_variant in ["t", "s", "m", "c", "e"]:
    print(f'running inference for YOLO version {yolo_variant}')

    output_folder_path = Path('./outputs')
    output_folder_path.mkdir(parents=True, exist_ok=True)
    #absolute image path or URL
    image_location = np.frombuffer(fetch(img_path).read_bytes(), np.uint8)
    image = [cv2.imdecode(image_location, 1)]
    out_path = (output_folder_path / f"{Path(img_path).stem}_output{Path(img_path).suffix or '.png'}").as_posix()
    if not isinstance(image[0], np.ndarray):
      print('Error in image loading. Check your image file.')
      sys.exit(1)
    pre_processed_image = preprocess(image)
    yolo_infer = YOLOv9(*SIZES[yolo_variant]) if yolo_variant in SIZES else YOLOv9()
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{yolo_variant}.safetensors'))
    load_state_dict(yolo_infer, state_dict)
    st = time.time()
    pred = yolo_infer(pre_processed_image)
    pred = pred.numpy()[0]
    pred = pred[pred[:, 4] >= 0.25]
    print(f'did inference in {int(round(((time.time() - st) * 1000)))}ms')
    #v9 and v3 have same 80 class names for Object Detection
    class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
    pred = scale_boxes(pre_processed_image.shape[2:], pred, image[0].shape)
    draw_bounding_boxes_and_save(orig_img_path=image_location, output_img_path=out_path, predictions=pred, class_labels=class_labels)