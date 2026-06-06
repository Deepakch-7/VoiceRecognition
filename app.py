#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Indian Sign Language Recognition - Streamlit Web App
Deploy on Streamlit Community Cloud (share.streamlit.io)
"""

import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from collections import Counter, deque
from PIL import Image

from model import KeyPointClassifier
from model import PointHistoryClassifier

# ── ISL Gesture to Meaning Mapping ───────────────────────────────────────────
ISL_MEANINGS = {
    "Open": "Hello",
    "Close": "Thank You",
    "Pointer": "Yes",
    "OK": "Good",
    "Stop": "Stop",
    "Clockwise": "Please",
    "Counter Clockwise": "Sorry",
    "Move": "Come Here",
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Hand Gesture Recognition",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 Indian Sign Language Recognition")
st.markdown(
    "Show a **hand gesture** to the camera. The app will detect and translate it into meaning."
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    min_detection_confidence = st.slider(
        "Min Detection Confidence", 0.1, 1.0, 0.7, 0.05
    )
    min_tracking_confidence = st.slider(
        "Min Tracking Confidence", 0.1, 1.0, 0.5, 0.05
    )
    draw_landmarks = st.checkbox("Draw Hand Landmarks", value=True)
    draw_bounding_box = st.checkbox("Draw Bounding Box", value=True)
    st.markdown("---")
    st.markdown("### 📖 Gesture Guide")
    for gesture, meaning in ISL_MEANINGS.items():
        st.markdown(f"- **{gesture}** → {meaning}")

# ── Load Models (cached so they load only once) ───────────────────────────────
@st.cache_resource
def load_models():
    kpc = KeyPointClassifier()
    phc = PointHistoryClassifier()
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        kp_labels = [row[0] for row in csv.reader(f)]
    with open(
        "model/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        ph_labels = [row[0] for row in csv.reader(f)]
    return kpc, phc, kp_labels, ph_labels


@st.cache_resource
def load_hands(det_conf, track_conf):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # True for single-frame (camera_input) mode
        max_num_hands=1,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return hands, mp_hands


keypoint_classifier, point_history_classifier, kp_labels, ph_labels = load_models()
hands_model, mp_hands = load_hands(min_detection_confidence, min_tracking_confidence)

# ── Helper Functions ──────────────────────────────────────────────────────────
def calc_landmark_list(image, landmarks):
    ih, iw = image.shape[:2]
    return [
        [min(int(lm.x * iw), iw - 1), min(int(lm.y * ih), ih - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0]
    for p in tmp:
        p[0] -= bx
        p[1] -= by
    flat = list(itertools.chain.from_iterable(tmp))
    mx = max(map(abs, flat)) or 1
    return [v / mx for v in flat]


def calc_bounding_rect(image, landmarks):
    ih, iw = image.shape[:2]
    arr = np.array(
        [
            [min(int(lm.x * iw), iw - 1), min(int(lm.y * ih), ih - 1)]
            for lm in landmarks.landmark
        ]
    )
    x, y, w, h = cv.boundingRect(arr)
    return [x, y, x + w, y + h]


def draw_info_on_image(image, brect, handedness_label, gesture, meaning, draw_bb, draw_lm, hand_landmarks):
    if draw_bb:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 200, 0), 2)
        cv.rectangle(image, (brect[0], brect[1] - 30), (brect[2], brect[1]), (0, 200, 0), -1)
        label = f"{handedness_label}: {gesture} ({meaning})"
        cv.putText(
            image, label, (brect[0] + 5, brect[1] - 8),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA,
        )

    if draw_lm:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmark_style(),
            mp.solutions.drawing_styles.get_default_hand_connection_style(),
        )
    return image


# ── Main Camera Section ───────────────────────────────────────────────────────
st.markdown("---")
col_cam, col_result = st.columns([2, 1])

with col_cam:
    st.subheader("📷 Camera")
    img_file = st.camera_input("Take a photo or allow continuous capture")

with col_result:
    st.subheader("🔍 Recognition Result")
    gesture_placeholder = st.empty()
    meaning_placeholder = st.empty()
    confidence_placeholder = st.empty()
    history_placeholder = st.empty()

# ── History (session state) ───────────────────────────────────────────────────
if "gesture_history" not in st.session_state:
    st.session_state.gesture_history = []

# ── Process Frame ─────────────────────────────────────────────────────────────
if img_file is not None:
    # Decode image
    bytes_data = img_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # Run MediaPipe
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands_model.process(rgb)

    gesture_text = "No hand detected ✋"
    meaning_text = "—"
    handedness_label = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            processed = pre_process_landmark(landmark_list)

            hand_sign_id = keypoint_classifier(processed)
            gesture_text = kp_labels[hand_sign_id]
            meaning_text = ISL_MEANINGS.get(gesture_text, gesture_text)
            handedness_label = handedness.classification[0].label

            debug_image = draw_info_on_image(
                debug_image, brect, handedness_label,
                gesture_text, meaning_text,
                draw_bounding_box, draw_landmarks, hand_landmarks,
            )

        # Add to history
        if gesture_text and gesture_text != "No hand detected ✋":
            st.session_state.gesture_history.append(gesture_text)
            if len(st.session_state.gesture_history) > 10:
                st.session_state.gesture_history.pop(0)

    # Show processed image
    with col_cam:
        st.image(
            cv.cvtColor(debug_image, cv.COLOR_BGR2RGB),
            caption="Processed Frame",
            use_column_width=True,
        )

    # Show results
    with col_result:
        if gesture_text == "No hand detected ✋":
            gesture_placeholder.warning("No hand detected ✋")
            meaning_placeholder.info("Show your hand to the camera")
        else:
            gesture_placeholder.success(f"**Gesture:** {gesture_text}")
            meaning_placeholder.info(f"**Meaning:** {meaning_text}")
            confidence_placeholder.caption(f"Hand: {handedness_label}")

        # Gesture history
        if st.session_state.gesture_history:
            st.markdown("### 📜 Recent Gestures")
            history_text = " → ".join(st.session_state.gesture_history[-5:])
            st.markdown(f"`{history_text}`")
            sentence = " ".join(
                [ISL_MEANINGS.get(g, g) for g in st.session_state.gesture_history]
            )
            st.markdown(f"**Sentence:** _{sentence}_")

            if st.button("🗑️ Clear History"):
                st.session_state.gesture_history = []
                st.rerun()

else:
    with col_result:
        gesture_placeholder.info("Waiting for camera input...")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with MediaPipe + Streamlit · Deploy free at share.streamlit.io")
