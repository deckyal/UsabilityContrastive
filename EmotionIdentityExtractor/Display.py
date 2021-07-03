import cv2
import numpy as np
import random


class Display:

    def __init__(self, window_name="Display"):
        """This is the constructor defining some basic constants for the following functions"""
        self.display_val_arousal_circle = False
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.line_thickness = 2
        self.text_color = (0, 255, 0)
        self.box_color = (255, 0, 177)
        self.main_box_color = self.text_color

        self.window_name = window_name
        window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.show_only_face = False

    def display_boxes(self, input_frame, box_coordinates=np.array((), dtype=int), color=(0, 255, 0)):
        """
        This method will print the boxes defined by the coordinates in box_coordinates onto the frame input_frame
        with the defined color. [x1 y1 x2 y2] represent a box around a face with the coordinates (x1,y1) of the
        top left corner and the bottom right corner defined by (x2,y2)
        :param input_frame: numpy nd-array representing a frame
        :param box_coordinates: numpy nd-array of the form [[x1 y1 x2 y2], ... , [x1 y1 x2 y2]]
        :param color: (r,g,b) tuple with r,g,b integers representing the color of the box borderline
        :return:
        """
        for box in box_coordinates:
            x1, y1, x2, y2 = box
            cv2.rectangle(input_frame, (x1, y1), (x2, y2), color, self.line_thickness)

    def display_labels(self, input_frame, x_coord, y_coord, labels={}, color=(0, 255, 0), font_scale=1,
                       use_top_orientation=True):
        """
        This method will print the label informations onto the input frame. If use_top_orientation is true it will use
        (x_coord, y_coord) as the top left corner of all labels and print the labels to the lines under it.
        If use_top_orientation is false it will use (x_coord, y_coord) as the bottom left corner of all labels and
        print the labels in the lines above it. labels entries will be written onto the frame in the form "key:value"
        :param input_frame: numpy nd-array representing a frame
        :param x_coord: integer, representing x coordinate
        :param y_coord: integer, representing y coordinate
        :param labels: dict, representing keys as label categories and values as the values
        :param color: (r,g,b) tuple with r,g,b integers representing the color of the box borderline
        :param use_top_orientation: boolean, see description above
        :return:
        """
        (_, text_height), _ = cv2.getTextSize("A", self.font, font_scale, self.line_thickness)
        offset = 10
        if use_top_orientation:
            y_coord = y_coord + text_height
            for k, v in labels.items():
                cv2.putText(input_frame, str(k) + ":" + str(v), (x_coord, y_coord), self.font, font_scale,
                            color, self.line_thickness, cv2.LINE_AA)
                y_coord = y_coord + text_height + offset
        else:
            for k, v in labels.items():
                cv2.putText(input_frame, str(k) + ":" + str(v), (x_coord, y_coord), self.font, font_scale,
                            color, self.line_thickness, cv2.LINE_AA)
                y_coord = y_coord - text_height - offset

    def display_va_circle(self, input_frame, valence, arousal):
        """
        
        :param input_frame:
        :param x_coord:
        :param y_coord:
        :param width:
        :param height:
        :param valence:
        :param arousal:
        :return:
        """
        input_frame = cv2.resize(input_frame, (800, 800), interpolation=cv2.INTER_CUBIC)
        frame_width = len(input_frame[0])
        frame_height = len(input_frame)
        (width, label_height), _ = cv2.getTextSize("valence:x-Axis  ", self.font, 0.8, self.line_thickness)
        x_coord = frame_width - width - 10
        y_coord = frame_height - width - 30 - 2 * label_height

        font_scale = 0.8
        (text_width_1, text_height), _ = cv2.getTextSize("1", self.font, font_scale, self.line_thickness)
        (text_width_neg1, _), _ = cv2.getTextSize("-1", self.font, font_scale, self.line_thickness)
        offset = 6
        # circle parameters
        diameter = width - text_width_1 - text_width_neg1 - 2*offset
        radius = int(diameter / 2)
        center = (int(x_coord + text_width_neg1 + offset + radius), int(y_coord + text_height + offset + radius + text_height/2))

        cv2.circle(input_frame, center, radius, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        lp1 = (center[0] - radius, center[1])
        lp2 = (center[0] + radius, center[1])
        cv2.line(input_frame, lp1, lp2, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        lp1 = (center[0], center[1] - radius)
        lp2 = (center[0], center[1] + radius)
        cv2.line(input_frame, lp1, lp2, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        text_x_coord = center[0] + radius + offset
        text_y_coord = int(center[1] + text_height / 2)
        # circle east end
        cv2.putText(input_frame, "1", (text_x_coord, text_y_coord), self.font, font_scale, (0, 0, 0),
                    self.line_thickness, cv2.LINE_AA)
        # circle west end
        cv2.putText(input_frame, "-1", (x_coord, text_y_coord), self.font, font_scale, (0, 0, 0),
                    self.line_thickness, cv2.LINE_AA)

        text_x_coord = int(center[0] - text_width_1 / 2)
        text_y_coord = y_coord + text_height
        #circle north end
        cv2.putText(input_frame, "1", (text_x_coord, text_y_coord), self.font, font_scale, (0, 0, 0),
                    self.line_thickness, cv2.LINE_AA)

        text_x_coord = int(center[0] - text_width_neg1 / 2)
        text_y_coord = center[1] + radius + offset + text_height
        #circle south end
        cv2.putText(input_frame, "-1", (text_x_coord, text_y_coord), self.font, font_scale, (0, 0, 0),
                    self.line_thickness, cv2.LINE_AA)

        # caption for labeling the axis inside the circle
        caption_y_coord = center[1] + radius + 2*offset + text_height
        (caption_width, _), _ = cv2.getTextSize("arousal:y-Axis", self.font, font_scale, self.line_thickness)
        caption_x_coord = int(center[0] - caption_width / 2)
        self.display_labels(input_frame, caption_x_coord, caption_y_coord, color=(0, 0, 0), font_scale=font_scale,
                            labels={"valence": "x-axis", "arousal": "y-axis"})

        # blue line displaying the valence, arousal vector
        val_ar_vec = (int(valence * radius), int(arousal * radius))
        cv2.line(input_frame, center, (center[0] + val_ar_vec[0], center[1] - val_ar_vec[1]), color=(255, 0, 0),
                 thickness=2, lineType=cv2.LINE_AA)
        return input_frame

    def display(self, input_frame, box_coordinates=np.array((), dtype=int), main_face_box=np.array((), dtype=int),
                time_labels={}, data_labels={}):
        """

        :param input_frame:
        :param box_coordinates:
        :param main_face_box:
        :param time_labels:
        :param data_labels:
        :return:
        """
        if not (box_coordinates.size == 0):
            if self.show_only_face:
                x1, y1, x2, y2 = main_face_box[0]
                input_frame = input_frame[y1:y2, x1:x2]
            else:
                # prints purple boxes around all detected faces
                self.display_boxes(input_frame, box_coordinates, self.box_color)
                # prints a green box around the main face
                self.display_boxes(input_frame, main_face_box, self.text_color)

        frame = cv2.resize(input_frame, (800, 800), interpolation=cv2.INTER_CUBIC)
        #frame_width = len(frame[0])
        frame_height = len(frame)

        # prints the time and frame number information in the top left corner
        self.display_labels(frame, 0, 10, time_labels, self.text_color, use_top_orientation=True)
        # prints the detected labels in the bottom left corner
        self.display_labels(frame, 0, frame_height - 10, data_labels, self.text_color, use_top_orientation=False)

        #if "valence" in data_labels and "arousal" in data_labels:  # self.display_val_arousal_circle:
            #self.display_va_circle(frame, valence=data_labels["valence"], arousal=data_labels["arousal"])

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            self.show_only_face = not self.show_only_face
