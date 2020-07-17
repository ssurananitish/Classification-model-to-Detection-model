import cv2
import numpy as np

def image_pyramid(image, scale_factor = 1.5, min_size=(224,224)):
    yield image
    while True:
        scale_down = int(image.shape[1] / scale_factor)
        image = resize(image, width=scale_down)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

def sliding_window(image, step_size, window_size):
    # Function for implementing the sliding window for the image
    for y in range(0,image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1]-window_size[0], step_size):
            yield(x, y, image[y:y+window_size[0], x:x+window_size[1]])

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resizing the Image while maintaining the aspect ratio of the image
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        ar = height / float(h)
        dim = (int(w * ar), height)

    else:
        ar = width / float(w)
        dim = (width, int(h * ar))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def non_max_suppression(boxes, probs=None, overlapping_threshold=0.3):
    # Function for implementing the non max suppression and selecting the bounding box with maximum confidence
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    selected_indexes = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if probs is not None: 
        idxs = probs
    else: 
        idxs = y2

    idxs = np.argsort(idxs)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selected_indexes.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[idxs[:last]]), np.maximum(y1[i], y1[idxs[:last]])
        xx2, yy2 = np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapping_threshold)[0])))
    
    return boxes[selected_indexes].astype("int")