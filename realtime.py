import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

from deeplab import DeepLabModel
from utils import create_pascal_label_colormap, vis_segmentation, label_to_color_image

# MODEL_PATH = 'models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
MODEL_PATH = 'models/deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz'
MODEL = DeepLabModel(MODEL_PATH)
# _SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image3.jpg?raw=true')


cap = cv2.VideoCapture('walk-cropped.mp4')   # /dev/video0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_im, seg_map = MODEL.run(img)
    resized_im = cv2.cvtColor(np.array(resized_im), cv2.COLOR_RGB2BGR)

    # vis_segmentation(resized_im, seg_map)
    # thresh = np.array(seg_map)
    # thresh = thresh[thresh != 0]
    # thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    seg_im = label_to_color_image(seg_map).astype(np.uint8)
    seg_im = cv2.cvtColor(np.array(seg_im), cv2.COLOR_RGB2BGR)
    out = np.hstack((resized_im, seg_im))

    cv2.imshow('out', out)
    print(np.unique(seg_map))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()