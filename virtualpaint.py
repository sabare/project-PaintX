import cv2
import numpy as np
import HandTrackingModule as htm
from classifier_module import *


folderpath = "Header"
overlay=[]
lmlist=[]
classes = ['saw', 'crown', 'cup', 'cloud', 'pizza', 'camera', 'face']

im1 = cv2.imread("Header/head11.jpg")
im1 = im1[0:125, 0:1280]
im2 = cv2.imread("Header/head12.jpg")
im2 = im2[0:125, 0:1280]
im3 = cv2.imread("Header/head13.jpg")
im3 = im3[0:125, 0:1280]

bcolor = (0,0,255)
size = 15
channel = 1
overlay.append(im1)
overlay.append(im2)
overlay.append(im3)
header = overlay[2]
xp,yp=0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
ctime, ptime = 0, 0

def process_img(img,shape):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    c = sorted(contours, key=cv2.contourArea, reverse=True)
    x,y,w,h = cv2.boundingRect(c[1])

    image = img[y:y+h, x:x+w]
    image1 = cv2.resize(image,shape)
    image2 = image1/255.
    #cv2.imshow("sdasda", image)
    image3 = image2.reshape(28,28,1)
    image4 = np.repeat(image3, 3, -1)
    #cv2.imshow("sa", image4)
    #print(image4.shape)
    image5 = np.transpose(image4, (2, 1, 0))

    image6 = torch.from_numpy(image5)

    image7 = torch.unsqueeze(image6, 0)

    return image7


detector = htm.HandDetector()

while True:
    success,img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img,draw=True)
    lmlist = detector.findPosition(img)
    img[0:125, 0:1280] = header

    if len(lmlist)!=0:
        fingers = detector.fingersUp()

        x1,y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        if fingers[1] and fingers[2] and fingers[3]:
            cv2.rectangle(img, (x2 - 30,y2 - 30),
                          (x2 + 30, y2 + 30), (0, 0, 255), cv2.FILLED)
            header = overlay[1]
            cv2.rectangle(imgCanvas, (x2 - 30, y2 - 30),
                          (x2 + 30, y2 + 30), (0, 0, 0), cv2.FILLED)

        elif fingers[1] and fingers[2]:
            header = overlay[2]
            xp,yp = x1,y1
            if y1<125 and 670<x1<790:
                imggen = process_img(imgInv,(28,28))
                print(classes[pred(imggen)])

        elif fingers[1]:
            header = overlay[0]
            if abs(xp-x1) > 120 or abs(yp-y1) > 120:
                xp,yp = x1,y1
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),(0,0,255), 26)

            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50,255,cv2.THRESH_BINARY_INV)

    imgcInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgcInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("image", img)

    cv2.waitKey(1)
