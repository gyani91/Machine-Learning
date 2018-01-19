from library import *
from resize import *
from classify import *
from convert import *
from plot import *

CONVERT = False
CONVERT_IMAGE_PATH = "Images/*/*.*"
RESIZE = False
RESIZE_IMAGE_PATH = "Images/*/*.jpg"
DOWNLOAD_MODEL = True
MODEL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
TRAIN = False
CLASSIFY = False
CLASSIFY_IMAGE_PATH = "Classify/Panda.jpg"
PLOT = True
RESULT_FILE = 'result.txt'

images_counts = range(50, 650 + 1, 50)
class_counts= range(2, 15 + 1)

if CONVERT:
    convert(CONVERT_IMAGE_PATH)

if RESIZE:
    resize(RESIZE_IMAGE_PATH)

# Download the pretrained model (Google Inception V3 for our analysis)
if DOWNLOAD_MODEL:
    download_pretrained_model(MODEL)

if TRAIN:
    # Create graph from the pretrained model
    graph, bottleneck_tensor, image_data_tensor, resized_image_tensor = (create_graph())

    for class_count in class_counts:
        for images_count in images_counts:
            handle = open(RESULT_FILE, 'ab')
            np.savetxt(handle, [run_model(images_count, class_count, graph, bottleneck_tensor, image_data_tensor, resized_image_tensor)])
            handle.close()

if CLASSIFY:
    classify(CLASSIFY_IMAGE_PATH)

if PLOT:
    plot(RESULT_FILE)