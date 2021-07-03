import tensorflow.compat.v1 as tf_old
import cv2
import numpy as np
import sys
import os
import DataIO as dw
import time

sys.path.insert(1, "Libs\\MTCNN_USE_TF_E2E")
from detect_face_tf import Network
import Display as dp
#import EyeDetector as eyed


class FaceExtractor:

    def __init__(self):
        """
        This constructor builds the FaceExtractor by loading the pretrained model and building the tensorflow network.
        """
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # f_path = os.path.dirname(os.path.realpath(__file__))

        tfmodel = 'Libs\\MTCNN_USE_TF_E2E\\mtcnn_model\\mtcnn_ckpt\\mtcnn.ckpt'
        # set config
        tfconfig = tf_old.compat.v1.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.07
        tfconfig.gpu_options.allow_growth = True

        # init session
        sess = tf_old.compat.v1.Session(config=tfconfig)
        net = Network(minsize=50, threshold=[0.6, 0.8, 0.9])
        net.create_architecture()
        saver = tf_old.train.Saver()
        saver.restore(sess, tfmodel)

        # places the loaded session and network in class variables
        self.net = net
        self.sess = sess
        print('Loaded network {:s}'.format(tfmodel))
        self.previous_face_coords = np.zeros((1, 4), dtype='int32')
        self.previous_face_coords = np.reshape([[0, 0, 1, 1]], (1, 4))
        # print("intialization of previous face coords", self.previous_face_coords)
        # print("datatype of previous face coords", type(self.previous_face_coords))
        # print("prev face coords shape", np.shape(self.previous_face_coords))

    def has_negative_coordinates(self, coordinates):
        for box in coordinates:
            for element in box:
                if element < 0:
                    return True
        return False

    def get_faces_coords(self, img):
        """
        This function takes as input an image and returns a numpy-nd array of the form
        [[x1, y1, x2, y2], ... , [x1, y1, x2, y2]], where each [x1, y1, x2, y2] stands for the bounding rect around a
        detected face. The upper left corner of the rect is given by the coordinates (x1, y1) and the bottom
        right corner by (x2, y2).
        :param img: a numpy nd-array containing representing an image which shall be examined for faces
        :return: numpy-nd array the form [[x1, y1, x2, y2], ... , [x1, y1, x2, y2]] representing faces bounding boxes
        """
        image_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = image_convert[np.newaxis, :, :, :]
        total_boxes, points = self.net.test_image(self.sess, im)
        # print("total boxes:", total_boxes)

        if len(total_boxes) > 0:
            face_coords = np.ndarray(shape=(len(total_boxes), 4), dtype='int32')
            for i in range(0, len(total_boxes)):
                face_coords[i] = np.absolute(np.rint(total_boxes[i][0:4]))
            return face_coords, 1
        else:
            return self.previous_face_coords, 0

    def get_single_face_coords(self, all_faces_coords):
        """
        This is postprocessing function which takes all the coordinates from the rects around detected faces
        and returns the coordinates for a single rect around the main face. The main face is chosen as the one which
        has the bounding box most similar to the chosen one from the previous frame.
        :param all_faces_coords: numpy n-d array of the form [[x1, y1, x2, y2], ... , [x1, y1, x2, y2]], where
            each [x1, y1, x2, y2] denotes a bounding box of a detected face
        :return: a numpy n-d array of the form [[x1, y1, x2, y2]] where (x1,y1) are the coordinates of the upper left
            corner of a rect and (x2, y2) are the coordinates of the bottom right corner.
        """
        # print("all face bounding boxes:", all_faces_coords)
        if np.array_equal(all_faces_coords, self.previous_face_coords):
            # print("all face bounding boxes1:", all_faces_coords)
            return self.previous_face_coords
        else:
            main_face_index = 0
            min = sys.float_info.max
            for i in range(0, len(all_faces_coords)):
                diff = np.absolute(all_faces_coords[i] - self.previous_face_coords[0])
                diff_norm = np.linalg.norm(diff)
                if diff_norm < min:
                    main_face_index = i
                    min = diff_norm
            single_face_coords = np.zeros((1, 4), dtype='int32')
            single_face_coords[0] = all_faces_coords[main_face_index]
            self.previous_face_coords = single_face_coords
        # print("all face bounding boxes2:", all_faces_coords)
        return single_face_coords


def create_box_coordinates_storing_format(frameNumber, raw_box_coordinates, main_box_coordinates):
    print("create_box_coordinates_storing_format", frameNumber, raw_box_coordinates, main_box_coordinates)
    result_box_coordinates = np.zeros(shape=(len(raw_box_coordinates), 6), dtype=np.int)
    box_coordinates_entry = np.ndarray(shape=(1, 6), dtype=np.int)

    is_main_box = 0
    # print("len(raw_box_coordinates)", len(raw_box_coordinates))
    for i in range(0, len(raw_box_coordinates)):
        # print("iteration i=", i)
        # print("raw_box_coordinates[i]", raw_box_coordinates[i])
        # print("main_box_coordinates[0]", main_box_coordinates[0])
        if np.array_equal(raw_box_coordinates[i], main_box_coordinates[0]):
            is_main_box = 1

        box_coordinates_entry[0][0] = frameNumber
        box_coordinates_entry[0][1] = is_main_box
        box_coordinates_entry[0][2:6] = raw_box_coordinates[i]
        # print("box_coordinates_entry", box_coordinates_entry)
        is_main_box = 0

        result_box_coordinates[i] = box_coordinates_entry[0]
        # print("result_box_coordinates", result_box_coordinates)

    return result_box_coordinates


def save_face_imgs_to_dir(path_to_video, path_to_save_extracted_faces, img_size=240, display=False):
    """
    This method will extract a face from each frame in the video (defined by path_to_video) and save it to a
    subdirectory of path_to_save_extracted_faces called faces. For each detected face the coordinates of a rect
    surrounding the face will be determined. The face which has the most similar coordinates the coordinates of the
    previous frame will be seen as main face, extracted and saved to the faces directory. If no faces are detected
    in a frame the coordinates from the main face in the previous frame will used.
    :param path_to_video: string, describes path to source video
    :param path_to_save_extracted_faces: string, describing a path to a directory in which a faces subdirectory will be
        created which will contain the extracted faces
    :param img_size: images will be saved with the dimension (size, size)
    :param display: boolean, if true will display the boxes around the faces in an extra window, otherwise no extra
        window will appear
    :return: img_names, timestamps - img_names is a list of strings which represented the names of .jpg files for the
        extracted faces in the faces directory, timestamps is a list of floats which represent the time in msec of each
        frame. So the .jpg file named as in img_names[i] has the timestamp timestamps[i].
    """
    starttime = time.time()
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise Exception("Couldn't open the video: " + path_to_video)

    face_extractor = FaceExtractor()
    dir_to_save_imgs = os.path.join(path_to_save_extracted_faces, "faces")
    totalNumberofFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total number of frames", totalNumberofFrames)
    img_names = []
    time_stamps = []

    # eye_detector = eyed.Eye_Detector()

    display_obj = dp.Display()

    # eyes_x1_coords = []
    # eyes_y1_coords = []
    # eyes_x2_coords = []
    # eyes_y2_coords = []
    # eyes_frame_numbers = []

    face_x1_coords = []
    face_y1_coords = []
    face_x2_coords = []
    face_y2_coords = []
    face_is_mainface_coords = []
    face_frame_numbers = []
    previous_frame = np.ndarray((4, 4, 3), np.int)
    previous_frame.fill(1)
    while True:
        currentFrameNumber = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        is_retrieved, frame = cap.read()
        if not is_retrieved:
            # print("no frame is retrieved, current frame number ", currentFrameNumber)
            break

        raw_face_coords, has_face_detected = face_extractor.get_faces_coords(frame)
        single_face_coords = face_extractor.get_single_face_coords(raw_face_coords)
        x1, y1, x2, y2 = single_face_coords[0]

        # print("choosen single face coords:", single_face_coords[0])

        # print("frame type: ", type(frame))
        # print("previous frame type: ", type(previous_frame))
        # print("shape of frame", np.shape(frame))
        # print("shape of previous frame", np.shape(previous_frame))

        if has_face_detected == 1:
            # print("face detected in frame number:", currentFrameNumber)
            previous_frame = frame
            face_img = frame[y1:y2, x1:x2]
        else:
            # print("no face detected in frame number:", currentFrameNumber)
            # print("previous frame in case no frame was detected:", previous_frame)
            # frame = previous_frame

            # if currentFrameNumber == 0:
            #    frame = frame
            #    frame = np.zeros(shape=(100, 100, 3), dtype=np.int)
            #    frame.fill(0)
            # else:
            #    frame = previous_frame

            face_img = frame

        # face_img = frame[y1:y2, x1:x2]
        # print("face image in bouding box:", face_img)
        # eyes_bounding_boxes = eye_detector.get_eye_box_coords(frame, single_face_coords)

        # for i in range(0, len(eyes_bounding_boxes)):
        # x1_fc, y1_fc, x2_fc, y2_fc = eyes_bounding_boxes[i]
        # eyes_x1_coords.append(x1_fc)
        # eyes_y1_coords.append(y1_fc)
        # eyes_x2_coords.append(x2_fc)
        # eyes_y2_coords.append(y2_fc)
        # eyes_frame_numbers.append(currentFrameNumber)

        for i in range(0, len(raw_face_coords)):
            x1_fc, y1_fc, x2_fc, y2_fc = raw_face_coords[i]
            face_x1_coords.append(x1_fc)
            face_y1_coords.append(y1_fc)
            face_x2_coords.append(x2_fc)
            face_y2_coords.append(y2_fc)
            face_frame_numbers.append(currentFrameNumber)
            if x1 == x1_fc and y1 == y1_fc and x2 == x2_fc and y2 == y2_fc:
                face_is_mainface_coords.append(1)
            else:
                face_is_mainface_coords.append(0)

        time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_stamps.append(time_stamp)
        zero_filed_frame_number = str(currentFrameNumber).zfill(len(str(totalNumberofFrames)))
        img_name = zero_filed_frame_number + "-" + str(has_face_detected)
        img_names.append(img_name)
        dw.write_img(dir_to_save_imgs, face_img, img_name, img_size)

        if display:
            frame_progress = zero_filed_frame_number + "/" + str(totalNumberofFrames)
            # print("main face box", single_face_coords)
            display_obj.display(frame, box_coordinates=raw_face_coords, main_face_box=single_face_coords,
                                time_labels={"timestamp": time_stamp, "Frame": frame_progress,
                                             "name": img_name + ".jpg"},
                                data_labels={"detected face": str(has_face_detected)}
                                )
            # display_obj.display(frame, box_coordinates=eyes_bounding_boxes, main_face_box=single_face_coords,
            #                    time_labels={"timestamp": time_stamp, "Frame": frame_progress, "name": img_name+".jpg"},
            #                    data_labels={"detected face": str(has_face_detected)}
            #                    )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    dir_to_save_face_box_coords = os.path.join(path_to_save_extracted_faces, "box_coords")
    csv_path = os.path.join(dir_to_save_face_box_coords, "faces_box_coords.csv")
    csv_data = [face_frame_numbers, face_is_mainface_coords, face_x1_coords, face_y1_coords, face_x2_coords,
                face_y2_coords]
    dw.write_CSV_File(csv_path, ["frame number", "is_mainface_coords", "x1", "y1", "x2", "y2"],
                      csv_data, write_header=True)

    # dir_to_save_eye_box_coords = os.path.join(path_to_save_extracted_faces, "eye_coords")
    # csv_path = os.path.join(dir_to_save_eye_box_coords, "eyes_box_coords.csv")
    # csv_data = [eyes_frame_numbers, eyes_x1_coords, eyes_y1_coords, eyes_x2_coords, eyes_y2_coords]
    # dw.write_CSV_File(csv_path, ["frame number", "x1", "y1", "x2", "y2"],
    #                  csv_data, write_header=True)

    cap.release()
    endtime = time.time()
    print("time needed for extracting all faces in secs:", endtime - starttime)
    return img_names, time_stamps
