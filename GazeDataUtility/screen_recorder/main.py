import cv2
import numpy as np
import pyautogui
from pynput.mouse import Listener
from pynput import keyboard
import win32gui
from win32api import GetSystemMetrics
import os
from datetime import datetime

# display screen resolution, get it from your OS settings
SCREEN_SIZE = (GetSystemMetrics(0), GetSystemMetrics(1))

# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")

# webcam capture
vc = cv2.VideoCapture(0)

# colors of certain pixels of the screen to determine whether to start recording or not
downClickColors = [0, 0, 0, 0]
upClickColors = [0, 0, 0, 0]

# stops the recording when it's set to false
is_recording = False

# current proband and website number
subject_number = ""
site_number = 0

# maximum constraints
maximum_subjects = 20
maximum_site_number = 8

# stops program when set to True
stopLoop = False


# starts the screen recording of the current user/site combination
def start_recording():
    global out
    global recordingsCounter
    global is_recording
    path = "data/user_" + subject_number + "/site_" + str(site_number) + "/video/"
    videoTitle = "" + datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
    filename = path + videoTitle + ".avi"
    print(filename)
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    print("STARTED: " + filename)
    print("Proband: " + subject_number)
    print("Site: " + str(site_number))
    print("--------")
    is_recording = True


# ends the recording and releases the video capture
def end_recording():
    global out
    global is_recording
    global site_number
    is_recording = False
    print("STOPPED")
    print("Proband: " + subject_number)
    print("Site: " + str(site_number))
    print("--------")
    try:
        out.release()
    except:
        ()
    site_number += 1
    if site_number > maximum_site_number:
        stop()


# checks for escape key press to end recording
def on_press(key):
    if key == keyboard.Key.esc and is_recording:
        end_recording()


# mouse on click listeners that checks if screen recording should begin
def on_click(x, y, button, pressed):
    global is_recording
    if not is_recording:
        try:
            if pressed:
                # left side
                downClickColors[0] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                       round(SCREEN_SIZE[0] * 925 / 1920),
                                                       pyautogui.position()[1])
                # on the actual button 1
                downClickColors[1] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                       round((940 / 1920) * SCREEN_SIZE[0]),
                                                       pyautogui.position()[1] + 3)
                # on the actual button 2
                downClickColors[2] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                       round((980 / 1920) * SCREEN_SIZE[0]),
                                                       pyautogui.position()[1] - 3)
                # right side
                downClickColors[3] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                       round((SCREEN_SIZE[0] * 995) / 1920),
                                                       pyautogui.position()[1])
            else:
                # left side
                upClickColors[0] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                     round((SCREEN_SIZE[0] * 925) / 1920),
                                                     pyautogui.position()[1])
                # on the actual button 1
                upClickColors[1] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                     round((940 / 1920) * SCREEN_SIZE[0]),
                                                     pyautogui.position()[1] + 3)
                # on the actual button 2
                upClickColors[2] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                     round((980 / 1920) * SCREEN_SIZE[0]),
                                                     pyautogui.position()[1] - 3)
                # right side
                upClickColors[3] = win32gui.GetPixel(win32gui.GetDC(win32gui.GetActiveWindow()),
                                                     round((SCREEN_SIZE[0] * 995) / 1920),
                                                     pyautogui.position()[1])
        except:
            # can throw an error if the mouse is not in the specified area
            print("Error: side of screen?")

        #  if should start recording
        if not pressed:
            # checks if colors match on mouse button down
            if (downClickColors[1] == 8488448 or downClickColors[2] == 8488448) and \
                    downClickColors[0] == 16777215 and downClickColors[3] == 16777215:

                # checks if colors match on mouse button up
                if (upClickColors[1] == 7621920 or upClickColors[2] == 7621920) and \
                        upClickColors[0] == 16777215 and upClickColors[3] == 16777215:
                    # if enough matched, start recording the screen
                    start_recording()


def record_frames():
    global is_recording
    global out
    while not stopLoop:
        # initialize camera
        has_frame = False
        try:
            has_frame, frame = vc.read()
        except:
            print("no cam found! - error")

        # actually record webcam
        if is_recording:
            if has_frame:
                out.write(frame)
            else:
                print("no cam found!")

            # UNCOMMENT FOR SCREEN RECORDING INSTEAD
            # take screenshot
            # img = pyautogui.screenshot()
            # convert these pixels to a proper numpy array to work with OpenCV
            # frame = np.array(img)
            # convert colors from BGR to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # draw mouse position
            # frame = cv2.circle(frame, pyautogui.position(), 5, (0, 255, 255), -1)
            # write frame
            # out.write(frame)


def create_folder_structure():
    existingFoldersIgnored = False
    print("Create folder structure...")
    for x in range(maximum_subjects):
        for y in range(maximum_site_number):
            try:
                os.makedirs("data/user_" + str(x + 1) + "/site_" + str(y + 1) + "/video")
                os.makedirs("data/user_" + str(x + 1) + "/site_" + str(y + 1) + "/json")
                os.makedirs("data/user_" + str(x + 1) + "/site_" + str(y + 1) + "/output")
            except:
                existingFoldersIgnored = True
    if existingFoldersIgnored:
        print("No new folders added")
    else:
        print("New folders added")


def start():
    global mouse_listener
    global keyboard_listener
    global subject_number
    global site_number
    correctInput = False;
    create_folder_structure()

    # asks for current subject/website combination
    while not correctInput:
        subject_number = input("Bitte geben sie den Probanten ein (1-" + str(maximum_subjects) + "):")
        site_number = int(input("Bei welcher Website starten sie (1-" + str(maximum_site_number) + "):"))

        # checks if input is within constraints
        if int(subject_number) < 1 or int(
                subject_number) > maximum_subjects or site_number < 1 or site_number > maximum_site_number:
            print("Falsche Eingabe")
        else:
            correctInput = True

    # define and start mouse listener thread
    mouse_listener = Listener(on_click=on_click)
    mouse_listener.start()

    # define and start keyboard listener thread
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    print("Bereit")
    record_frames()


def stop():
    global mouse_listener
    global keyboard_listener
    global out
    global stopLoop
    stopLoop = True
    mouse_listener.stop()
    keyboard_listener.stop()
    cv2.destroyAllWindows()
    out.release()


start()
stop()
