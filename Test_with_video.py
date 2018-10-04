#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import time
import cv2
import dlib

from keras.preprocessing import image as imagekeras
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

size = 150
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_face_recognization_model.h5'

def return_name(codelist):
    names = ['樊胜美', '关雎尔', '邱莹莹']
    for it in range(0, len(codelist), 1):
        if int(codelist[it]) == 1.0:
            return names[it]

def return_name_en(codelist):
    names = ['fsm', 'gje', 'qyy']
    for it in range(0, len(codelist), 1):
        if int(codelist[it]) == 1.0:
            return names[it]

# recognize faces
def face_rec():
    global  image_ouput
    model = load_model(os.path.join(save_dir, model_name))
    camera = cv2.VideoCapture("test.mp4") 

    while (True):
        read, img = camera.read()
        try:
            if not (type(img) is np.ndarray):
                continue
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn into greyscale
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break

        # using the dlib with frontal_face_detector
        detector = dlib.get_frontal_face_detector()
        dets = detector(gray_img, 1) # detect human face

        facelist = []
        for i, d in enumerate(dets): 
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            img = cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 2) # draw rectangles

            face = img[x1:y1, x2:y2]
            face = cv2.resize(face, (size, size))

            x_input = np.expand_dims(face, axis=0)
            prey = model.predict(x_input)
            print(prey, 'prey')

            facelist.append([d,  return_name(prey[0])]) 

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # covert to opencv standard
        pil_im = Image.fromarray(cv2_im)

        draw = ImageDraw.Draw(pil_im)  
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 20, encoding="utf-8") 

        try:
            for i in facelist:
                draw.text((i[0].left() + int((i[0].right() - i[0].left()) / 2 - len(i[1]) * 10), i[0].top() - 20), i[1],
                          (255, 0, 0), font=font)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue

        # PIL image to cv2 image
        cv2_char_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        # display image
        cv2.imshow("camera", cv2_char_img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
