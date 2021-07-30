import FaceExtraction as fx
import EmotionDetector as ed
import os
import time
import cv2
# import eval_script as ev
import VAClassifier as va


def get_time_stamps(path_to_video):
    starttime = time.time()
    print("Acquiring timestamp in ms for every frame now. This may take a while ...")
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise Exception("Couldn't open the video: " + path_to_video)

    time_stamps = []
    while True:
        is_retrieved, frame = cap.read()
        if not is_retrieved:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_stamps.append(timestamp)
    cap.release()
    endtime = time.time()
    print("time needed for extracting all timestamps in secs:", endtime - starttime)
    return time_stamps


def process_video(path_to_video, path_to_results_dir):
    vidname = os.path.splitext(os.path.basename(path_to_video))[0]
    result_dir = os.path.join(path_to_results_dir, vidname)
    if (not os.path.exists(result_dir)) or (not os.path.isdir(result_dir)):
        os.makedirs(result_dir)

    img_names, time_stamps = fx.save_face_imgs_to_dir(path_to_video, result_dir, display=True)
    # time_stamps = get_time_stamps(path_to_video)
    ed.save_emotion_labels_to(result_dir, time_stamps, display=True)
    # ev.save_valence_arousal_to(result_dir, time_stamps, display=True)
    va.save_valence_arousal_to(result_dir, time_stamps, display=True)


##################################################################
# Edit here the path to your video and the path to an output directory

video_path = os.path.join("C:", "Desktop", "test.mp4")
save_results_in_dir = os.path.join("imgs")
process_video(video_path, save_results_in_dir)

###################################################################
