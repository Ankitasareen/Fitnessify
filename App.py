import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import cv2
import PoseTrackingModule as pm
import time
from bokeh.models.widgets import Div


detector = pm.PoseDetector()
count = 0.5
dir = 0
pTime = 0

DEMO_VIDEO = 'curls.mp4'
# r'C:\Users\HP\OneDrive\Desktop\Fit\AI Trainer\
st.title('AI Fitness trainer - FitAI')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('FitAI')
st.sidebar.subheader(
    '  Do not want to go out for working out, but still want personalized training? Do not worry, we got you covered!')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (height, int(h*r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App', 'Fitness Assistant','Yoga Pose Detector']
                                )


if app_mode == 'About App':
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    kpj1, kpj2 = st.columns(2)

    with kpj1:
        st.markdown("**How does the smart fitness assistant FitAI work?**")
        st.markdown('''

           Fitnessify's AI Fitness assistant- FitAI will help you workout at home in a proper way.You wont have to go out, to the gyms for personalized assistance.
           
           1.Computer vision for fitness technique tracking - The app automatically tracks fitness techniques and recognizes the movements of your body parts using your camera.

           2.You can master yourself using a real-time AR image over your body on the screen, and check your posture.

           3.Automatic Tracking- It counts the no.of reps, pace and other stats that would help us,provide you your weekly report, analyzing your progress.
            ''')

    with kpj2:

        st.video(
            r"C:\Users\HP\OneDrive\Desktop\Fit\AI Trainer\gif.mp4", start_time=0)


elif app_mode == 'Fitness Assistant':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**No of curls**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    while cap.isOpened():
        i += 1
        success, img = cap.read()
        if not success:
            continue
        img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        #img = image_resize(image=img, width=640)
        #img = cv2.resize(img, (640, 640))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
    # print(lmList)
        if(len(lmList)> 0):
            angle = detector.findAngle(img, 12, 14, 16)
        per = np.interp(angle, (199, 301), (0, 100))
        bar = np.interp(angle, (199, 301), (650, 100))

    #print(angle, per)

        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

    # print(count)
    #cv2.rectangle(img, (0, 450), (150, 550), (0, 255, 0), cv2.FILLED)
    # cv2.putText(img, str(int(count)), (60, 510), cv2.FONT_HERSHEY_PLAIN, 4,
                # (255, 0, 0), 5)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        if record:
            #st.checkbox("Recording", value=True)
            out.write(img)
        # Dashboard
        kpi1_text.write(
            f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(
            f"<h1 style='text-align: center; color: red;'>{count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(
            f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        stframe.image(img, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    cap.release()
    out. release()

    # cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
    # (255, 0, 0), 5)
