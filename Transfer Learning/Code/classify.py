import tensorflow as tf
from resize import *
from convert import *

#IMAGE_PATH = "Classify/panda.jpg"
TRAINED_GRAPH = "sets_graph.pb"
LABELS = "label.txt"
FINAL_TENSOR_NAME = "final_tensor"

def classify(IMAGE_PATH):
    # Convert the image to JPEG
    converted_image = convert(IMAGE_PATH)

    # Resize the image
    resized_image = resize(converted_image)

    # Read the input_image
    input_image = tf.gfile.FastGFile(resized_image, 'rb').read()

    # Load labels
    class_labels = [line.rstrip() for line
                   in tf.gfile.GFile(LABELS)]

    #Load the trained model
    with tf.gfile.FastGFile(TRAINED_GRAPH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # Feed the input_image to the graph and get the prediction
        softmax_tensor = sess.graph.get_tensor_by_name(FINAL_TENSOR_NAME+':0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': input_image})

        # Sort the labels of the prediction in order of confidence
        sorted_labels = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print('Classification:')
        for index in sorted_labels:
            class_label = class_labels[index]
            percentage = predictions[0][index]*100
            print(('%s (%.2f' % (class_label, percentage))+'%)')