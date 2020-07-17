# python main.py --image "Input_images/lawn_mower.jpg" --size "(250, 250)" --min-conf 0.95
# python main.py --image "Input_images/stingray.jpg" --size "(400, 150)" --min-conf 0.95
# python main.py --image "Input_images/hummingbird.jpg" --size "(250, 250)" --min-conf 0.95

import numpy as np
import os 
import cv2
import time
import argparse
from tensorflow.keras.applications import DenseNet121, imagenet_utils
from tensorflow.keras.applications.densenet  import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from utils import *

# Argument Parser Variables
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",	help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,	help="minimum confidence to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,	help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# Defining important constants
WIDTH = 600
PYRAMID_SCALE = 1.5
WIN_STEP = 8
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)
_, filename = os.path.split(args["image"])

model = DenseNet121(weights="imagenet", include_top=True)
original_image = cv2.imread(args["image"])
original_image = resize(original_image, width=WIDTH)
(H, W) = original_image.shape[:2]
pyramid = image_pyramid(original_image, scale_factor = PYRAMID_SCALE, min_size = ROI_SIZE)
rois, locs = [], []
start = time.time()

for image in pyramid:
    scale = W / float(image.shape[1])
    for (x, y, roi_image) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x, y, w, h = int(x * scale), int(y * scale), int(ROI_SIZE[0] * scale), int(ROI_SIZE[1] * scale)
        roi = cv2.resize(roi_image, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        if args["visualize"] > 0:
            clone = original_image.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Visualization_Window", clone)
            cv2.imshow("ROI", roi_image)
            cv2.waitKey(0)

end = time.time()
print("Total time taken for sliding window: {:.5f} seconds".format(end - start))
rois = np.array(rois, dtype="float32")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("Time taken for classification of all ROIs: {:.5f} seconds".format(end - start))
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}
for (i, p) in enumerate(preds):
    (imagenet_ID, label, prob) = p[0]
    if prob >= args["min_conf"]:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

print(len(labels))
for label in labels.keys():
    clone = original_image.copy()
    for (box, prob) in labels[label]:
        (x1, y1, x2, y2) = box
        cv2.rectangle(clone, (x1, y1), (x2, y2),(0, 255, 0), 2)

    cv2.imshow("Before_NMS", clone)
    clone = original_image.copy()
    boxes, proba = np.array([p[0] for p in labels[label]]), np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(clone, (x1, y1), (x2, y2),(0, 255, 0), 2)
        y = y1-10 if y1-10 > 10 else y1 + 10
        cv2.putText(clone, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    cv2.imshow("After_NMS", clone)
    cv2.imwrite(os.path.join("./Output_images", filename), clone)
    cv2.waitKey(0)
