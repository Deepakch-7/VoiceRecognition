#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Indian Sign Language Recognition + Voice Navigation + Text-to-Speech
Hackathon Project - Fixed & Enhanced Version
Compatible with Python 3.11 + mediapipe + tensorflow
"""

import csv
import copy
import argparse
import itertools
import threading
import queue
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyttsx3
import speech_recognition as sr

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# ── ISL Gesture to Meaning Mapping ───────────────────────────────────────────
ISL_MEANINGS = {
    "Open":              "Hello",
    "Close":             "Thank You",
    "Pointer":           "Yes",
    "OK":                "Good",
    "Stop":              "Stop",
    "Clockwise":         "Please",
    "Counter Clockwise": "Sorry",
    "Move":              "Come Here",
}

# ── TTS Engine (global, thread-safe queue) ────────────────────────────────────
tts_queue = queue.Queue()
last_spoken = ""
last_spoken_time = 0
SPEAK_COOLDOWN = 3  # seconds between repeating same gesture

def tts_worker():
    """Background thread that speaks queued messages."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        tts_queue.task_done()

def speak(text):
    """Add text to TTS queue (non-blocking)."""
    tts_queue.put(text)

# ── Voice Navigation ──────────────────────────────────────────────────────────
voice_command = ""
voice_listening = False

def listen_voice_command():
    """Listen for one voice command and store it."""
    global voice_command, voice_listening
    voice_listening = True
    speak("Listening")
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5)
        voice_command = recognizer.recognize_google(audio)
        print(f"[Voice] Heard: {voice_command}")
        speak(f"You said {voice_command}")
    except sr.WaitTimeoutError:
        voice_command = ""
        speak("No voice detected")
    except sr.UnknownValueError:
        voice_command = ""
        speak("Sorry, I did not understand")
    except Exception as e:
        voice_command = ""
        speak("Microphone error")
        print(f"[Voice Error] {e}")
    finally:
        voice_listening = False

# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width",  type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return parser.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global last_spoken, last_spoken_time, voice_command

    args = get_args()

    # Start TTS background thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    speak("Welcome to Indian Sign Language Recognition. Press V for voice commands.")

    # Camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Classifiers
    keypoint_classifier       = KeyPointClassifier()
    point_history_classifier  = PointHistoryClassifier()

    # Labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length      = 16
    point_history       = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    print("=" * 50)
    print("  ISL Recognition Running!")
    print("  ESC = Quit | V = Voice Command")
    print("  K = Keypoint logging mode")
    print("  H = Point history logging mode")
    print("  N = Normal mode")
    print("=" * 50)

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)

        # Key handling
        if key == 27:   # ESC
            break
        if key == ord('v') or key == ord('V'):
            if not voice_listening:
                threading.Thread(target=listen_voice_command, daemon=True).start()

        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image      = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        current_gesture  = ""
        current_meaning  = ""
        finger_gesture_text = ""

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness):

                brect         = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Get labels
                current_gesture = keypoint_classifier_labels[hand_sign_id]
                current_meaning = ISL_MEANINGS.get(current_gesture, current_gesture)
                finger_gesture_text = point_history_classifier_labels[most_common_fg_id[0][0]]

                # Speak gesture meaning with cooldown
                now = time.time()
                if (current_gesture != last_spoken or
                        now - last_spoken_time > SPEAK_COOLDOWN):
                    if current_gesture:
                        speak(current_meaning)
                        last_spoken      = current_gesture
                        last_spoken_time = now

                # Draw
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image, brect, handedness,
                    f"{current_gesture} ({current_meaning})",
                    finger_gesture_text,
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_ui(debug_image, fps, mode, number,
                               voice_command, voice_listening)

        cv.imshow('ISL Recognition - Press V for Voice | ESC to Quit', debug_image)

    cap.release()
    cv.destroyAllWindows()
    tts_queue.put(None)  # Stop TTS thread


# ── UI Drawing ────────────────────────────────────────────────────────────────
def draw_ui(image, fps, mode, number, voice_cmd, is_listening):
    h, w = image.shape[:2]

    # FPS
    cv.putText(image, f"FPS: {fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # Voice status
    if is_listening:
        cv.putText(image, "🎤 LISTENING...", (10, h - 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
    elif voice_cmd:
        cv.putText(image, f"Voice: {voice_cmd}", (10, h - 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv.LINE_AA)

    # Instructions
    cv.putText(image, "V=Voice | ESC=Quit", (10, h - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

    # Mode
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, f"MODE: {mode_string[mode-1]}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM: {number}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv.LINE_AA)

    return image


# ── Helper Functions (unchanged from original) ────────────────────────────────
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110: mode = 0
    if key == 107: mode = 1
    if key == 104: mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    arr = np.empty((0, 2), int)
    for lm in landmarks.landmark:
        x = min(int(lm.x * iw), iw - 1)
        y = min(int(lm.y * ih), ih - 1)
        arr = np.append(arr, [[x, y]], axis=0)
    x, y, w, h = cv.boundingRect(arr)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    pts = []
    for lm in landmarks.landmark:
        pts.append([min(int(lm.x * iw), iw - 1),
                    min(int(lm.y * ih), ih - 1)])
    return pts

def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = list(itertools.chain.from_iterable(tmp))
    mx   = max(map(abs, flat))
    return [v / mx for v in flat]

def pre_process_point_history(image, point_history):
    iw, ih = image.shape[1], image.shape[0]
    tmp = copy.deepcopy(point_history)
    bx, by = 0, 0
    for i, p in enumerate(tmp):
        if i == 0:
            bx, by = p
        tmp[i] = [(p[0] - bx) / iw, (p[1] - by) / ih]
    return list(itertools.chain.from_iterable(tmp))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and 0 <= number <= 9:
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline='') as f:
            csv.writer(f).writerow([number, *landmark_list])
    if mode == 2 and 0 <= number <= 9:
        with open('model/point_history_classifier/point_history.csv', 'a', newline='') as f:
            csv.writer(f).writerow([number, *point_history_list])

def draw_landmarks(image, pts):
    connections = [
        (2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
        (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0)
    ]
    for a, b in connections:
        cv.line(image, tuple(pts[a]), tuple(pts[b]), (0,0,0), 6)
        cv.line(image, tuple(pts[a]), tuple(pts[b]), (255,255,255), 2)
    for i, p in enumerate(pts):
        r = 8 if i in [4,8,12,16,20] else 5
        cv.circle(image, tuple(p), r, (255,255,255), -1)
        cv.circle(image, tuple(p), r, (0,0,0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1]-22), (0,0,0), -1)
    label = handedness.classification[0].label
    if hand_sign_text:
        label += ': ' + hand_sign_text
    cv.putText(image, label, (brect[0]+5, brect[1]-4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    if finger_gesture_text:
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for i, p in enumerate(point_history):
        if p[0] != 0 and p[1] != 0:
            cv.circle(image, (p[0], p[1]), 1 + int(i/2), (152,251,152), 2)
    return image


if __name__ == '__main__':
    main()
