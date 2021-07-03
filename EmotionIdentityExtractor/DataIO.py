import csv
import cv2
import os
import numpy as np


def read_CSV_File(path_to_csv):
    """

    :param path_to_csv:
    :return:
    """
    with open(path_to_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        column_names = []
        list_of_labellists = []
        for row in csv_reader:
            if line_count == 0:
                column_names = list(row.keys())
                for _ in column_names:
                    empty_label_list = []
                    list_of_labellists.append(empty_label_list)
                line_count += 1

            column_index = 0
            for key in column_names:
                list_of_labellists[column_index].append(row[key])
                column_index += 1
            line_count += 1
    return column_names, list_of_labellists


def read_csv_eye_boxes(path_to_eye_boxes_csv):
    column_names, box_csv_data = read_CSV_File(path_to_eye_boxes_csv)
    # last_framenumber = int(box_csv_data[column_names.index('frame number')][-1])

    box_coords_per_frame = np.ndarray((0, 4), dtype=np.int)
    all_box_coords = []

    previous_framenumber = 0
    for i in range(0, len(box_csv_data[0])):
        x1 = int(box_csv_data[column_names.index('x1')][i])
        y1 = int(box_csv_data[column_names.index('y1')][i])
        x2 = int(box_csv_data[column_names.index('x2')][i])
        y2 = int(box_csv_data[column_names.index('y2')][i])
        box_coordinates = np.array([[x1, y1, x2, y2]], dtype=np.int)
        current_frame_number = int(box_csv_data[column_names.index('frame number')][i])

        if previous_framenumber == current_frame_number:
            box_coords_per_frame = np.vstack((box_coords_per_frame, box_coordinates))
        else:
            all_box_coords.append(box_coords_per_frame)
            previous_framenumber = current_frame_number
            box_coords_per_frame = np.ndarray((0, 4), dtype=np.int)
            box_coords_per_frame = np.vstack((box_coords_per_frame, box_coordinates))

    all_box_coords.append(box_coords_per_frame)
    return all_box_coords


def read_csv_face_boxes(path_to_face_boxes_csv):
    """
    This method reads a .csv file holding the data regarding the bounding boxes around the faces for each frame.
    It will return a numpy 2d-array for the main boxes around the main face in each frame.
    main_face_coords[i] holds the bounding box of the main face in frame i.
    main_face_coords[i] is a numpy array of the form [x1, y1, x2, y2] where (x1,y1) is the upper left corner coordinate
    of the bounding box and (x2,y2) the bottom right corner.
    Additional it returns the list all_box_coords which is a list of 2d numpy arrays.
    all_box_coords[i] is a numpy 2d-array holding all the detected bounding boxes in frame i and is of the form:
    [[x1,y1,x2,y2], <- first detected bounding box in frame i
    ...
    [x1,y1,x2,y2]] <- last detected bounding box in frame i
    :param path_to_face_boxes_csv: string representing a path to a .csv file
    :return: main_face_coords, aall_box_coords
    """
    column_names, box_csv_data = read_CSV_File(path_to_face_boxes_csv)
    last_framenumber = int(box_csv_data[column_names.index('frame number')][-1])

    main_face_coords = np.ndarray((last_framenumber + 1, 4), dtype=np.int)
    box_coords_per_frame = np.ndarray((0, 4), dtype=np.int)
    all_box_coords = []

    previous_framenumber = 0
    for i in range(0, len(box_csv_data[0])):
        x1 = int(box_csv_data[column_names.index('x1')][i])
        y1 = int(box_csv_data[column_names.index('y1')][i])
        x2 = int(box_csv_data[column_names.index('x2')][i])
        y2 = int(box_csv_data[column_names.index('y2')][i])
        box_coordinates = np.array([[x1, y1, x2, y2]], dtype=np.int)
        current_frame_number = int(box_csv_data[column_names.index('frame number')][i])

        if int(box_csv_data[column_names.index('is_mainface_coords')][i]) == 1:
            main_face_coords[current_frame_number] = box_coordinates[0]

        if previous_framenumber == current_frame_number:
            box_coords_per_frame = np.vstack((box_coords_per_frame, box_coordinates))
        else:
            all_box_coords.append(box_coords_per_frame)
            previous_framenumber = current_frame_number
            box_coords_per_frame = np.ndarray((0, 4), dtype=np.int)
            box_coords_per_frame = np.vstack((box_coords_per_frame, box_coordinates))

    all_box_coords.append(box_coords_per_frame)
    return main_face_coords, all_box_coords


def write_CSV_File(path_to_csv, label_categories, list_of_labellists, write_header=False):
    """
    This method writes the data contained in list_of_labelists to a .csv file. First row contains info about the
    different label_categories. Second and following rows contains the actual label data from the lists inside the
    list_of_labellists.
    :param path_to_csv: string representing the path to the .csv file, which will be created
    :param label_categories: list of strings which will be written in the first row of the .csv file
    :param list_of_labellists: list of label lists containing the actual data
    :return:
    """
    if not (len(label_categories) == len(list_of_labellists)):
        raise Exception("len(label_categories) and len(list_of_labellists) must be equal")

    dir_path = os.path.split(path_to_csv)[0]
    if (not os.path.exists(dir_path)) or (not os.path.isdir(dir_path)):
        os.makedirs(dir_path)

    with open(path_to_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=label_categories)
        if write_header:
            writer.writeheader()
        row = {}
        for i in range(0, len(list_of_labellists[0])):
            for j in range(0, len(list_of_labellists)):
                row[label_categories[j]] = list_of_labellists[j][i]
            writer.writerow(row)


def sort_img_numbers(img_name):
    img_number = int(img_name.split("-")[0])
    return img_number


def get_jpg_filenames_in_dir(path_to_dir):
    img_filenames = []
    fileending = ".jpg"
    for file in os.listdir(path_to_dir):
        if file.endswith(fileending):
            img_filenames.append(file)
    img_filenames.sort(key=sort_img_numbers)
    return img_filenames


def write_img(dir_path, img, name, size=128):
    """
    Saves the numpy nd-array defined by img as .jpg file with the name specified by name, e.g. as <name>.jpg
    in the directory specified by dir_path. The image has the dimensions of width = size and height = size.
    :param dir_path:
    :param img:
    :param name:
    :param size:
    :return:
    """
    if (not os.path.exists(dir_path)) or (not os.path.isdir(dir_path)):
        os.makedirs(dir_path)
    # print("shape of img", np.shape(img))
    # print("img value;", img)
    img_name = str(name) + ".jpg"
    path_to_img = os.path.join(dir_path, img_name)

    # print("save shape of img", np.shape(img))
    # print("save size in size:", size, "x", size)
    # print("save img value:", img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(path_to_img, img):
        raise Exception("Could not write the picture ", name, "to", path_to_img)
