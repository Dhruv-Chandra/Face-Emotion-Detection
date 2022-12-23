import os
import re

import cv2 as cvv
import keras
from keras.utils import img_to_array
import numpy as np
# import mediapipe as mp
from cv2 import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json


default = 'model_'
emotion_label = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
base_loc = 'H:/My Drive/Study/DS/Computer Vision/Face Emotion Detection/'

def ret_latest_file(x):
    ans = os.listdir(f'{base_loc}Models/')
    # print(ans)
    finlist = []
    re_pat = re.compile("([a-zA-Z]+_)([0-9]+)")
    for i in ans:
        try:
            if i[-4:] == 'json':
                res = re_pat.match(i).groups()  # type: ignore
                if res[0] == x:
                    finlist.append(int(res[1]))
        except:
            pass
    if len(finlist) > 0:
        return max(finlist)
    else:
        return 0

def get_prediction(image):
    name = f'{default}{int(ret_latest_file(default) + 1)}'
    image = np.expand_dims(image, axis = 0)
    # print(image.shape)
    
    model = model_from_json(open(f'{base_loc}Models/{name}/{name}.json', "r").read())
    model.load_weights(f'{base_loc}Models/{name}/{name}.h5')

    pred_score = model.predict(image)  # type: ignore
    pred = pred_score.argmax()

    return emotion_label[pred]

face_cascade = cv2.CascadeClassifier(
    cvv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        image = gray[y-5:y+h+5, x-5:x+w+5]
        image = cv2.resize(image, (48, 48))
        image = img_to_array(image)
        image = image / 255
        emotion = get_prediction(image)
        cv2.putText(
            img,
            f'{emotion}',
            (x, y - 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
            2)
        print(emotion)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()