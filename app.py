import cv2
import numpy as np
import pandas as pd
import time
import streamlit as st
import tempfile
import os
import base64

# 📸 Snapshot save setup
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_snapshot(frame, timestamp):
    filename = f"{SNAPSHOT_DIR}/fire_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
    cv2.imwrite(filename, frame)


# 🔊 Autoplay & loop alarm sound
def play_audio_autoplay():
    audio_file = open("Alarm.mp3", "rb")
    audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay loop>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# 🔥 Fire detection logic
def detect_fire(frame):
    fire_detected = False
    frame = cv2.resize(frame, (700, 500))
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([22, 50, 50], dtype='uint8')
    upper = np.array([35, 255, 255], dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    number_of_total = cv2.countNonZero(mask)

    if number_of_total > 2000:
        fire_detected = True
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, fire_detected

# 🖥 Streamlit UI setup
st.set_page_config(page_title="🔥 Fire Detection App", layout="centered")
st.title("🔥 Fire Detection Web Application")
st.markdown("Choose between **webcam** or **video file** for fire detection:")

source_type = st.radio("Select Input Source", ("Upload Video File", "Use Webcam"))

# 🔁 Video File Mode
if source_type == "Upload Video File":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        audio_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_frame, fire_detected = detect_fire(frame)

            if fire_detected:
                alert_placeholder.error("🚨 FIRE DETECTED!", icon="🔥")
                with audio_placeholder:
                    play_audio_autoplay()
            else:
                alert_placeholder.empty()
                audio_placeholder.empty()

            rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")

        cap.release()
        st.success("✅ Video processing completed.")

# 📷 Webcam Mode
elif source_type == "Use Webcam":
    st.warning("Make sure your webcam is connected and allowed.")

    # Initialize session states
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "fire_count" not in st.session_state:
        st.session_state.fire_count = 0
    if "fire_log" not in st.session_state:
        st.session_state.fire_log = []
    if "alarm_active" not in st.session_state:
        st.session_state.alarm_active = False

    # Buttons
    if st.button("▶️ Start Webcam Fire Detection", key="start_webcam_btn"):
        st.session_state.webcam_active = True
        st.session_state.fire_count = 0
        st.session_state.fire_log = []
        st.session_state.alarm_active = False

    if st.button("⏹️ Stop Webcam", key="stop_webcam_btn"):
        st.session_state.webcam_active = False
        st.session_state.alarm_active = False

    # Detection loop
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        fire_count_placeholder = st.empty()
        log_placeholder = st.empty()
        alert_placeholder = st.empty()
        audio_placeholder = st.empty()

        while st.session_state.webcam_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_frame, fire_detected = detect_fire(frame)

            if fire_detected:
               if not st.session_state.alarm_active:
                st.session_state.alarm_active = True
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.fire_log.append({"Time": timestamp})
                st.session_state.fire_count += 1
                save_snapshot(output_frame, timestamp)


                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                save_snapshot(output_frame, timestamp)
                alert_placeholder.error("🚨 FIRE DETECTED!", icon="🔥")
                with audio_placeholder:
                    play_audio_autoplay()

            else:
                st.session_state.alarm_active = False
                alert_placeholder.empty()
                audio_placeholder.empty()

            rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")

            fire_count_placeholder.metric("🔥 Fire Alerts Detected", st.session_state.fire_count)
            log_df = pd.DataFrame(st.session_state.fire_log)
            log_placeholder.dataframe(log_df, use_container_width=True)

        cap.release()
        st.success("✅ Webcam session ended.")

    # 📥 Fire log download
    if st.session_state.fire_log:
        df = pd.DataFrame(st.session_state.fire_log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Fire Log", csv, "fire_log.csv", "text/csv", key="download_log_btn")
