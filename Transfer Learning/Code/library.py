import tensorflow as tf
import numpy as np

import tarfile
import os
import sys
from six.moves import urllib
import hashlib
import re
import random


DIRECTORY_MODEL = os.path.join('Model')
DIRECTORY_TENSORBOARD = 'Logs'
DIRECTORY_BOTTLENECK = 'Bottlenecks'
DIRECTORY_IMAGES = os.path.join("Images")
TRAINED_GRAPH = "sets_graph.pb"
LABELS = "label.txt"
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
FINAL_TENSOR_NAME = "final_tensor"
PERCENTAGE_TESTING = 10
PERCENTAGE_VALIDATION = 10
TRAINING_STEPS = 500
VALIDATION_INTERVAL = 10
LEARNING_RATE = 0.01
TRAINING_BATCH_SIZE = 100
VALIDATION_BATCH_SIZE = 100
TESTING_BATCH_SIZE = -1

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_pretrained_model(MODEL):
    create_directory(DIRECTORY_MODEL)
    file_name = MODEL.split('/')[-1]
    file_path = os.path.join(DIRECTORY_MODEL, file_name)
    if not os.path.exists(file_path):
        def prog_bar(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (file_name, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(MODEL, file_path, prog_bar)
        print("\nDownload Completed")
        tarfile.open(file_path, 'r:gz').extractall(DIRECTORY_MODEL)

def create_graph():
    with tf.Session() as sess:
        file_name = os.path.join(DIRECTORY_MODEL, 'classify_image_graph_def.pb')
        with tf.gfile.FastGFile(file_name, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, image_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='',
                                    return_elements=[BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, image_data_tensor, resized_input_tensor

def split_data_into_sets(images_count, class_count):
    if not tf.gfile.Exists(DIRECTORY_IMAGES):
        print("Cannot find image directory named: " + DIRECTORY_IMAGES)
        return None
    sets = {}
    directories = [x[0] for x in tf.gfile.Walk(DIRECTORY_IMAGES)]
    is_root_dir = True
    for directory in directories[:class_count+1]:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        image_list = []
        class_directory = os.path.basename(directory)
        if class_directory == DIRECTORY_IMAGES:
            continue
        print("Reading images from class: " + class_directory)
        for extension in extensions:
            image_list.extend(tf.gfile.Glob(os.path.join(DIRECTORY_IMAGES, class_directory, '*.' + extension)))

        if not image_list:
            print('The directory is empty')
            continue
        
        class_label = re.sub(r'[^a-z0-9]+', ' ', class_directory.lower())
        training_list = []
        testing_list = []
        validation_list = []

        for file_name in image_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                          (images_count + 1)) *
                         (100.0 / images_count))
            if percentage_hash < PERCENTAGE_VALIDATION:
                validation_list.append(base_name)
            elif percentage_hash < (PERCENTAGE_TESTING + PERCENTAGE_VALIDATION):
                testing_list.append(base_name)
            else:
                training_list.append(base_name)
        sets[class_label] = {
        'Class': class_directory,
        'Training': training_list,
        'Testing': testing_list,
        'Validation': validation_list,
    }
    return sets

def get_path(directory, data_sets, class_label, index, category):
    class_list = data_sets[class_label]
    category_list = class_list[category]
    return os.path.join(directory, class_list['Class'], category_list[index % len(category_list)])

def create_bottleneck(bottleneck_path, data_sets, class_label, index, category, sess, image_data_tensor, bottleneck_tensor):
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_path(DIRECTORY_IMAGES, data_sets, class_label, index, category)
  image_data = tf.gfile.FastGFile(image_path, 'rb').read()
  bottleneck_values = np.squeeze(sess.run(bottleneck_tensor, {image_data_tensor: image_data}))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
      bottleneck_file.write(bottleneck_string)

def calculate_bottleneck_values(sess, data_sets, class_label, index, category, image_data_tensor, bottleneck_tensor):
    class_list = data_sets[class_label]
    class_directory = class_list['Class']
    class_directory_path = os.path.join(DIRECTORY_BOTTLENECK, class_directory)
    create_directory(class_directory_path)
    bottleneck_path = get_path(DIRECTORY_BOTTLENECK, data_sets, class_label, index, category) + '.txt'
    if not os.path.exists(bottleneck_path):
        create_bottleneck(bottleneck_path, data_sets, class_label, index, category, sess, image_data_tensor, bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def store_bottleneck_values(sess, data_sets, image_data_tensor, bottleneck_tensor):
    bottleneck_count = 0
    create_directory(DIRECTORY_BOTTLENECK)
    for class_label, class_list in data_sets.items():
        for category in ['Training', 'Testing', 'Validation']:
            category_list = class_list[category]
            for index, base_name in enumerate(category_list):
                calculate_bottleneck_values(sess, data_sets, class_label, index, category, image_data_tensor, bottleneck_tensor)
                bottleneck_count += 1
                if bottleneck_count % 100 == 0:
                    print(str(bottleneck_count) + ' bottleneck files created.')


def tensorboard_visualization(data):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(data)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(data - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(data))
        tf.summary.scalar('min', tf.reduce_min(data))
        tf.summary.histogram('histogram', data)

def create_final_layer(class_count, bottleneck_tensor):

    if tf.gfile.Exists(DIRECTORY_TENSORBOARD):
        tf.gfile.DeleteRecursively(DIRECTORY_TENSORBOARD)
    tf.gfile.MakeDirs(DIRECTORY_TENSORBOARD)

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor,
                                                       shape=[None, BOTTLENECK_TENSOR_SIZE],
                                                       name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder_with_default(tf.zeros([1, class_count], tf.float32),
                                                         [None, class_count],
                                                         name='GroundTruthInput')
    layer_name = 'final_training'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
                                        name='final_weights')
            tensorboard_visualization(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            tensorboard_visualization(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=FINAL_TENSOR_NAME)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)

def create_evaluation_step(final_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(final_tensor, 1)
      correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction

def load_bottleneck_values(sess, data_sets, batch_size, category, image_data_tensor, bottleneck_tensor, image_count, class_count):
    bottlenecks = []
    ground_truths = []
    filenames = []
    if batch_size >= 0:
        for i in range(batch_size):
            class_index = random.randrange(class_count)
            class_label = list(data_sets.keys())[class_index]
            image_index = random.randrange(image_count + 1)
            image_name = get_path(DIRECTORY_IMAGES, data_sets, class_label, image_index, category)
            bottleneck = calculate_bottleneck_values(sess, data_sets, class_label, image_index, category,
                                            image_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[class_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        for class_index, class_label in enumerate(data_sets.keys()):
            for image_index, image_name in enumerate(data_sets[class_label][category]):
                image_name = get_path(DIRECTORY_IMAGES, data_sets, class_label, image_index, category)
                bottleneck = calculate_bottleneck_values(sess, data_sets, class_label, image_index, category,
                                            image_data_tensor, bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[class_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)

    return bottlenecks, ground_truths, filenames

def run_model(images_count, class_count, graph, bottleneck_tensor, image_data_tensor, resized_image_tensor):

    #Split the images to training, testing and validation sets
    data_sets = split_data_into_sets(images_count, class_count)

    sess = tf.Session()
    store_bottleneck_values(sess, data_sets, image_data_tensor, bottleneck_tensor)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
       final_tensor) = create_final_layer(class_count, bottleneck_tensor)

    evaluation_step, prediction = create_evaluation_step(final_tensor, ground_truth_input)

    summary_merge_all = tf.summary.merge_all()
    training_file_writer = tf.summary.FileWriter(DIRECTORY_TENSORBOARD + '/Training', sess.graph)
    validation_file_writer = tf.summary.FileWriter(DIRECTORY_TENSORBOARD + '/Validation')

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(TRAINING_STEPS):
        train_bottlenecks, train_ground_truth, _ = load_bottleneck_values(
                sess, data_sets, TRAINING_BATCH_SIZE, 'Training', image_data_tensor,
                bottleneck_tensor, images_count, class_count)

        train_summary, _ = sess.run([summary_merge_all, train_step],
                                    feed_dict={bottleneck_input: train_bottlenecks,
                                               ground_truth_input: train_ground_truth})
        training_file_writer.add_summary(train_summary, i)

        # Validate how well the graph is training
        is_last_step = (i + 1 == TRAINING_STEPS)
        if (i % VALIDATION_INTERVAL) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            print('Step %d: Train accuracy = %.1f%%' % (i, train_accuracy * 100))
            print('Step %d: Cross entropy = %f' % (i, cross_entropy_value))

            validation_bottlenecks, validation_ground_truth, _ = (
                load_bottleneck_values(
                    sess, data_sets, VALIDATION_BATCH_SIZE, 'Validation',
                    image_data_tensor, bottleneck_tensor, images_count, class_count))

            validation_summary, validation_accuracy = sess.run(
                [summary_merge_all, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                         ground_truth_input: validation_ground_truth})
            validation_file_writer.add_summary(validation_summary, i)
            print('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (i, validation_accuracy * 100, len(validation_bottlenecks)))

    test_bottlenecks, test_ground_truth, test_file_names = (
        load_bottleneck_values(sess, data_sets, TESTING_BATCH_SIZE,
                               'Testing', image_data_tensor, bottleneck_tensor,
                               images_count, class_count))

    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})

    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    #Save the trained graph for later use
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with tf.gfile.FastGFile(TRAINED_GRAPH, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with tf.gfile.FastGFile(LABELS, 'w') as f:
        f.write('\n'.join(data_sets.keys()) + '\n')
    return test_accuracy * 100