from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import time
import cv2
import numpy as np
import DataIO as dw
import os
import Display as dp

class EmotionDetector:
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    def __init__(self):
        print(self.emotion_dict)
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        model_path = os.path.join("res", "models", "EmotionDetector", "model.h5")
        self.model.load_weights(model_path)

    def get_emotionlabel(self, source_frame):
        start = time.time()
        gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
        prediction = self.model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        end = time.time()

        # print("model emotion prediction:", prediction)
        # print("emotion prediction time in secs:", end - start)
        # return label
        return maxindex, self.emotion_dict[maxindex]


def display_emot_labeling(face_frame, emotion_label, emot_index, img_name):
    text_color = (50, 255, 0)
    frame = cv2.resize(face_frame, (800, 800), interpolation=cv2.INTER_CUBIC)
    img_name_label = "image:" + img_name
    emotion = "emotion:" + emotion_label + " " + str(emot_index)
    font_scale = 1
    line_thickness = 2
    cv2.putText(frame, img_name_label, (0, int(np.shape(frame)[0] * 95 / 100)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, line_thickness, cv2.LINE_AA)
    cv2.putText(frame, emotion, (0, int(np.shape(frame)[0] * 5 / 100)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, line_thickness, cv2.LINE_AA)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', frame)
    cv2.waitKey(1)


def save_emotion_labels_to(path_to_results_dir, time_stamps, display=False):
    starttime = time.time()

    discrete_dir = os.path.join(path_to_results_dir, "discrete")
    faces_dir = os.path.join(path_to_results_dir, "faces")
    emotion_classifier = EmotionDetector()
    list_of_jpgs_names = dw.get_jpg_filenames_in_dir(faces_dir)
    print(list_of_jpgs_names)
    emotion_labels = []
    frame_numbers = []

    display_obj = dp.Display()

    for img_name in list_of_jpgs_names:
        face_frame = cv2.imread(os.path.join(faces_dir, img_name), cv2.IMREAD_COLOR)
        emot_index, emotion_label = emotion_classifier.get_emotionlabel(face_frame)

        emotion_labels.append(emot_index)
        frame_number = int(img_name.split("-")[0])
        frame_numbers.append(frame_number)

        if display:
            #display_emot_labeling(face_frame, emotion_label, emot_index, img_name)
            display_obj.display(face_frame, time_labels={"img_name": img_name},
                                data_labels={"emotion label": emotion_label, "emotion index": emot_index})

    csv_path = os.path.join(discrete_dir, "emotions.csv")
    csv_data = [frame_numbers, time_stamps, emotion_labels]
    dw.write_CSV_File(csv_path, ["frame number", "timestamp in ms", "emotion"], csv_data, write_header=True)

    endtime = time.time()
    print("time needed for determing emotions for all frames in secs:", endtime - starttime)
