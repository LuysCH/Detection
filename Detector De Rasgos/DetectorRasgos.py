import sys
import os
import pathlib
import argparse
import time
from deepface import DeepFace
import cv2 as cv
import numpy as np 
from mtcnn.mtcnn import MTCNN


img=cv.imread("images/image4.jpg")


#Metodo para sacar la edad , genero y etnia
def DetectionFace(img,data):
   
  face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
  
  # Convertir en escala de grises  
  gris = cv.cvtColor(img, cv. COLOR_BGR2GRAY)
    
  # Detectar rostros  
  caras = face_cascade.detectMultiScale(gris, 1.6, 4)
    
  # Dibujar rectángulo alrededor de las caras  
  for (x, y, w, h) in caras:
      cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

  obj = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'race'], enforce_detection= False)

  #edad
  age = obj[0]['age']
  data[0]=age
    # Genero
  gen = obj[0]['dominant_gender']
  data[1]=gen
    # Race
  race = obj[0]['dominant_race']
  data[2]=race
    # Emociones
  #emotion = obj[0]['dominant_emotion']

  #cv.putText(img, str(gen), (65, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
  #cv.putText(img, str(age), (75, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
  #cv.putText(img, str(race), (75, 180), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
  #cv.putText(img, str(emotion), (75, 135), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

detector = MTCNN()

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='sample/2.jpg', help="it can be image or video or webcan id")
parser.add_argument('--input_type', default='image', help= "either image or video (for video file and webcam id)")
opt = parser.parse_args()

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0] : ((166, 21, 50), (240, 100, 85)),
    class_name[1] : ((166, 2, 25), (300, 20, 75)),
    class_name[2] : ((2, 20, 20), (40, 100, 60)),
    class_name[3] : ((20, 3, 30), (65, 60, 60)),
    class_name[4] : ((0, 10, 5), (40, 40, 25)),
    class_name[5] : ((60, 21, 50), (165, 100, 85)),
    class_name[6] : ((60, 2, 25), (165, 20, 65))
}

def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
    hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False

# define eye color category rules in HSV space
def find_class(hsv):
    color_id = 7
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id

def eye_color(image):
    imgHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    
    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate
   
    cv.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)

    cv.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (255,155,255),
              2)

    cv.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(class_name), np.float64)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y,x])] +=1 

    main_color_index = np.argmax(eye_class[:len(eye_class)-1])
    total_vote = eye_class.sum()

    label = 'Dominant Eye Color: %s' % class_name[main_color_index]
    data[3]=class_name[main_color_index]  
    cv.putText(image, label, (left_eye[0]-10, left_eye[1]-40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (155,255,0))
    
data=['age','dominant_gender','dominant_race','EyeColor']

img=cv.resize(img,(1000,1000))
image=img
DetectionFace(img,data)
eye_color(image)

print(data)

cv.waitKey()

cv. destroyAllWindows()