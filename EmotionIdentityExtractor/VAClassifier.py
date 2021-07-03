from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys

sys.path.insert(1, "Libs\\Aff-Wild-models\\vggface")
sys.path.insert(1, "Libs\\Aff-Wild-models")

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import data_process
import numpy as np
import tf_slim as slim
import time
import os
import DataIO as dw
import cv2
import Display as dp


class VAClassifier:

    def __init__(self):
        """
        This is the constructor of the VAClassifier which will sets the flags used by the evaluate function.
        """
        self.batch_size = 1
        self.seq_length = 1
        self.size = 96
        self.network = 'vggface_2000'
        path_to_pretrained_model = os.path.join("res", "models", "affwildnet", "vggface", "best_model_4096x2000x2",
                                                "4096x2000x2", "model.ckpt-0")
        self.pretrained_model_checkpoint_path = path_to_pretrained_model

    def evaluate(self, path_to_csv_input_file, display=False):
        """
        This function takes the path to a .csv file where each of the N rows is of the form
        "path_to_face_img,valence_label,arousal_label".
        Important here is the path_to_face_img in each row which represents the path to a .jpg file containing a face.
        valence_label and arousal_label can be arbitrary values since their value won't affect the result prediction
        of this function. The trained model, here defined by pretrained_model_checkpoint_path, will run in a session
        takes the values from the input .csv file and gives the actual valence and arousal values estimated for
        each face picture.
        :param path_to_csv_input_file: string, representing a path to .csv file of the above defined format
        :param display: boolean, if true the evaluating progress will be displayed in an extra window, otherwise
            no feedback is given on the progress of evaluating all the face images
        :return: a numpy nd-array of the form [[valence1, arousal1], ..., [valenceN, arousalN]] which contains
            by the model estimated valence and arousal values for each of the N face images.
        """
        start_evaluate = time.time()
        g = tf.Graph()
        with g.as_default():
            image_list, label_list = data_process.read_labeled_image_list(path_to_csv_input_file)
            # split into sequences, note: in the cnn models case this is splitting into batches of length: seq_length ;
            #                             for the cnn-rnn models case, I do not check whether the images in a sequence are consecutive or the images are from the same video/the images are displaying the same person
            image_list, label_list = data_process.make_rnn_input_per_seq_length_size(image_list, label_list,
                                                                                     self.seq_length)

            images = tf.convert_to_tensor(image_list)
            labels = tf.convert_to_tensor(label_list)

            # Makes an input queue
            input_queue = tf.train.slice_input_producer([images, labels, images], num_epochs=None, shuffle=False,
                                                        seed=None,
                                                        capacity=1000, shared_name=None, name=None)
            images_batch, labels_batch, image_locations_batch = data_process.decodeRGB(input_queue, self.seq_length,
                                                                                       self.size)
            images_batch = tf.to_float(images_batch)
            images_batch -= 128.0
            images_batch /= 128.0  # scale all pixel values in range: [-1,1]

            images_batch = tf.reshape(images_batch, [-1, 96, 96, 3])
            labels_batch = tf.reshape(labels_batch, [-1, 2])

            if self.network == 'vggface_4096':
                from vggface import vggface_4096x4096x2 as net
                network = net.VGGFace(self.batch_size * self.seq_length)
                network.setup(images_batch)
                prediction = network.get_output()

            elif self.network == 'vggface_2000':
                from vggface import vggface_4096x2000x2 as net
                network = net.VGGFace(self.batch_size * self.seq_length)
                network.setup(images_batch)
                prediction = network.get_output()

            elif self.network == 'affwildnet_resnet':
                from tensorflow.contrib.slim.python.slim.nets import resnet_v1
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    net, _ = resnet_v1.resnet_v1_50(inputs=images_batch, is_training=False, num_classes=None)

                    with tf.variable_scope('rnn') as scope:
                        cnn = tf.reshape(net, [self.batch_size, self.sequence_length, -1])
                        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(128) for _ in range(2)])
                        outputs, _ = tf.nn.dynamic_rnn(cell, cnn, dtype=tf.float32)
                        outputs = tf.reshape(outputs, (self.batch_size * self.sequence_length, 128))

                        weights_initializer = tf.truncated_normal_initializer(
                            stddev=0.01)
                        weights = tf.get_variable('weights_output',
                                                  shape=[128, 2],
                                                  initializer=weights_initializer,
                                                  trainable=True)
                        biases = tf.get_variable('biases_output',
                                                 shape=[2],
                                                 initializer=tf.zeros_initializer, trainable=True)

                        prediction = tf.nn.xw_plus_b(outputs, weights, biases)

            elif self.network == 'affwildnet_vggface':
                from affwildnet import vggface_gru as net
                network = net.VGGFace(self.batch_size, self.seq_length)
                network.setup(images_batch)
                prediction = network.get_output()

            num_batches = int(len(image_list) / self.batch_size)
            variables_to_restore = tf.global_variables()

            with tf.Session() as sess:
                init_fn = slim.assign_from_checkpoint_fn(
                    self.pretrained_model_checkpoint_path, variables_to_restore,
                    ignore_missing_vars=False)
                init_fn(sess)
                print('Loading model {}'.format(self.pretrained_model_checkpoint_path))

                # changed following two lines from the original to prevent exceptions
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                evaluated_predictions = []

                display_obj = dp.Display()
                try:
                    img_index = 0
                    for _ in range(num_batches):

                        pr, l, imm = sess.run([prediction, labels_batch, image_locations_batch])
                        # print("time for predicting val, arousal for one frame in secs:", sess_end-sess_start)

                        frame = cv2.imread(image_list[img_index][0], cv2.IMREAD_COLOR)
                        frame = display_obj.display_va_circle(frame, pr[0][0], pr[0][1])
                        display_obj.display(frame, time_labels={"img": os.path.basename(image_list[img_index][0])},
                                            data_labels={"valence": pr[0][0], "arousal": pr[0][1]})

                        evaluated_predictions.append(pr)
                        img_index += 1
                        if coord.should_stop():
                            break
                    coord.request_stop()
                except Exception as e:
                    coord.request_stop(e)

                # also to prevent exceptions
                coord.join(threads)

                predictions = np.reshape(evaluated_predictions, (-1, 2))
                end_evaluate = time.time()
                print("time needed for executing eval in secs:", end_evaluate - start_evaluate)
            return predictions


def display_progress(path_to_face_img, valence_prediction, arousal_prediction):
    print(path_to_face_img)
    text_color = (50, 255, 0)
    face_frame = cv2.imread(path_to_face_img, cv2.IMREAD_COLOR)
    frame = cv2.resize(face_frame, (800, 800), interpolation=cv2.INTER_CUBIC)
    img_name = os.path.basename(path_to_face_img)
    img_name_label = "image:" + img_name
    valence_label = "valence:" + str(valence_prediction)
    arousal_label = "arousal:" + str(arousal_prediction)
    font_scale = 1
    line_thickness = 2
    (_, text_height), _ = cv2.getTextSize(img_name_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
    print("text_size", text_height)
    cv2.putText(frame, img_name_label, (0, int(10 + text_height)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, line_thickness, cv2.LINE_AA)
    cv2.putText(frame, valence_label, (0, int(np.shape(frame)[0] - 50 - 10 - text_height)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, line_thickness, cv2.LINE_AA)
    cv2.putText(frame, arousal_label, (0, int(np.shape(frame)[0] - 30)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, line_thickness, cv2.LINE_AA)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', frame)
    cv2.waitKey(1)


def save_valence_arousal_to(path_to_results_dir, time_stamps, display=False):
    """
    This function takes as input a string representing a directory which has a subdirectory called "faces"
    which contains all face images. Assume a list containing all face images sorted by their name. The i-th face image
    in this list has the i-th timestamp from time_stamps. The i-th estimated valence and arousal value for the i-th
    face image will be written to the valence-arousal.csv contained in the subdirectory "va".
    The rows in the valence-arousal.csv have format "frame_number,timestamp,valence,arousal", where the frame_number i
    represents image i and the i-th frame in the original video.
    :param path_to_results_dir: string representing the path to the dir containing all results for the video,
        the directory is named after the video
    :param time_stamps: list of floats, the time_stamps[i] represent the time in msec of the image i in the video
    :param display: if True, the valence and arousal evaluating progress will be display in an extra window, otherwise
        not
    :return:
    """
    starttime = time.time()

    va_dir = os.path.join(path_to_results_dir, "va")
    faces_dir = os.path.join(path_to_results_dir, "faces")
    list_of_jpgs_names = dw.get_jpg_filenames_in_dir(faces_dir)

    print("jpg names for valar:", list_of_jpgs_names)

    path_to_imgs = []
    valence_labels = []
    arousal_labels = []
    frame_numbers = []

    for img_name in list_of_jpgs_names:
        path_to_imgs.append(os.path.join(faces_dir, img_name))
        valence_labels.append(0.8)
        arousal_labels.append(0.8)
        frame_number = int(img_name.split("-")[0])
        frame_numbers.append(frame_number)

    path_to_eval_input_csv = os.path.join(va_dir, "input_to_eval_function.csv")
    eval_input_csv_data = [path_to_imgs, valence_labels, arousal_labels]
    valence_labels = []
    arousal_labels = []
    dw.write_CSV_File(path_to_eval_input_csv, ["img paths", "valence labels", "arousal labels"], eval_input_csv_data,
                      write_header=False)
    del eval_input_csv_data

    va_classifier = VAClassifier()
    predictions = va_classifier.evaluate(path_to_eval_input_csv)

    for prediction in predictions:
        valence_labels.append(prediction[0])
        arousal_labels.append(prediction[1])

    path_to_va_csv = os.path.join(va_dir, "valence_arousal.csv")
    va_csv_data = [frame_numbers, time_stamps, valence_labels, arousal_labels]
    dw.write_CSV_File(path_to_va_csv, ["frame number", "timestamp in ms", "valence", "arousal"], va_csv_data,
                      write_header=True)

    endtime = time.time()
    print("time for determining valence and arousal for all face images in secs:", endtime - starttime)
