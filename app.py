import cv2
from deepface import DeepFace
import streamlit as st
from PIL import Image
import numpy as np
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Emotion Detection", page_icon="🐱‍🚀")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

url = "https://assets1.lottiefiles.com/packages/lf20_e7b2phbv.json"
res_json = load_lottieurl(url)
st_lottie(res_json,height=300,width=1200)

html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h3 style="color:white;text-align:center;">Detects Emotion Age Gender Using Face as Input</h3>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Tools used - Deepface OpenCV Streamlit</h4>
                                            </div>
                                            </br>"""
st.markdown(html_temp_home1, unsafe_allow_html=True)
    

one,two = st.columns(2)


with one:
    st.write("**Input Feed**")
    url1 = "https://assets1.lottiefiles.com/packages/lf20_go0eoxdr.json"
    res_json = load_lottieurl(url1)
    st_lottie(res_json,height=50,width=50)
    input_img = st.camera_input('Please click a picture.')
    
@st.cache(suppress_st_warning=True)
with two:
    if input_img:
        img = Image.open(input_img)
        img = np.array(img)
        predictions = DeepFace.analyze(img,enforce_detection=False)
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText( img, predictions['dominant_emotion'], (0,50), font, 1, (201, 52, 235), 2, cv2.LINE_4 );
        cv2.putText( img, 'Age: '+str(predictions['age']), (200,50), font, 1, (201, 52, 235), 2, cv2.LINE_AA );
        cv2.putText( img, predictions['gender'], (0,150), font, 1, (201, 52, 235), 2, cv2.LINE_4 );
        st.image(img)
    else:
        st.write('**No image for processing**')
        url2 = "https://assets10.lottiefiles.com/private_files/lf30_jo7huq2d.json"
        res_json = load_lottieurl(url2)
        st_lottie(res_json,height=90,width=90)
