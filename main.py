import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

import csv
import copy
import itertools
import numpy as np
import time
import cv2 as cv
import mediapipe as mp
from Model.classifier import Classifier

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import screen_brightness_control as sbc

import warnings
warnings.filterwarnings('ignore')

def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1366)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

    hands = (mp.solutions.hands.Hands(
                                        static_image_mode = False,
                                        max_num_hands = 1,
                                        min_detection_confidence = 0.7
                                     ))

    classifier = Classifier()

    with open('Model/Label.csv', encoding = 'utf-8-sig') as f:
        labels = [row[0] for row in csv.reader(f)]

    mode = 0

    command = np.loadtxt('Model/Command.csv', delimiter = ',', dtype = 'str')
    command = {int(i[0]): i[1] for i in command}

    commands = {
                1: 'increase volume',
                2: 'decrease volume',
                3: 'increase brightness',
                4: 'decrease brightness'
               }

    while True:
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                logging_csv(number, mode, pre_processed_landmark_list)

                hand_sign_id = classifier(pre_processed_landmark_list)

                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, handedness, labels[hand_sign_id], command[hand_sign_id])

                run_command(commands, volume, hand_sign_id)

        debug_image = draw_info(debug_image, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:
        mode = 0
    if key == 107:
        mode = 1
    return number, mode

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize(n):
        return n / max_value

    temp_landmark_list = list(map(normalize, temp_landmark_list))

    return [round(i, 3) for i in temp_landmark_list]

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        time.sleep(0.1)
    if mode == 1 and (0 <= number <= 9):
        with open('Model/Keypoint.csv', 'a', newline = "") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        from_to = [[2, 3], [3, 4],                                                        # Thumb
                   [5, 6], [6, 7], [7, 8],                                                # Index Finger
                   [9, 10], [10, 11], [11, 12],                                           # Middle Finger
                   [13, 14], [14, 15], [15, 16],                                          # Ring Finger
                   [17, 18], [18, 19], [19, 20],                                          # Pinky Finger
                   [0, 1], [1, 2], [2, 5], [5, 9], [9, 13], [13, 17], [17, 0]]            # Palm
        
        for i in from_to:
            cv.line(image, tuple(landmark_point[i[0]]), tuple(landmark_point[i[1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[i[0]]), tuple(landmark_point[i[1]]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        size = 8 if index % 4 == 0 else 5

        cv.circle(image, (landmark[0], landmark[1]), size, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), size, (0, 0, 0), 1)

    return image

def draw_info_text(image, handedness, hand_sign_text, command_text):
    info_text = handedness.classification[0].label[0:]

    if hand_sign_text != "":
        info_text = info_text + ': ' + hand_sign_text
    cv.putText(image, info_text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image, 'You order me to' + command_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_info(image, mode, number):
    if mode == 0:
        cv.putText(image, 'Mode: Detecting Mode', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    else:
        cv.putText(image, 'Mode: Logging Mode', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if 0 <= number <= 9:
            cv.putText(image, 'NUM: ' + str(number), (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def inc_volume(volume, increase):
    min_vol = volume.GetVolumeRange()[0]
    max_vol = volume.GetVolumeRange()[1]

    current_volume = volume.GetMasterVolumeLevel()
    new_volume = max(min(max_vol, current_volume + increase), min_vol)

    volume.SetMasterVolumeLevel(new_volume, None)

def dec_volume(volume, decrease):
    min_vol = volume.GetVolumeRange()[0]
    max_vol = volume.GetVolumeRange()[1]

    current_volume = volume.GetMasterVolumeLevel()
    new_volume = max(min(max_vol, current_volume - decrease), min_vol)

    volume.SetMasterVolumeLevel(new_volume, None)

def inc_brightness():
    sbc.set_brightness(max(min(100, sbc.get_brightness()[0] + 5), 0))

def dec_brightness():
    sbc.set_brightness(max(min(100, sbc.get_brightness()[0] - 5), 0))

def run_command(commands, volume, c):
    if c in commands.keys():
        if commands[c] == 'increase volume':
            inc_volume(volume, 5)
        elif commands[c] == 'decrease volume':
            dec_volume(volume, 5)
        # elif commands[c] == 'increase brightness':
        #     inc_brightness()
        # elif commands[c] == 'decrease brightness':
        #     dec_brightness()

if __name__ == '__main__':
    main()
