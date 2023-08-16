import cv2
import numpy as np
import math
import random

draw_next=True
spot_x=random.random()*1920
spot_y=random.random()*1010
sigma=3

def draw_Gaussian(img,x,y,sigma):
    for x1 in range(int(x-math.sqrt(8*math.log(2))*sigma),int(x+math.sqrt(8*math.log(2))*sigma)+1):
        for y1 in range(int(y-math.sqrt(8*math.log(2))*sigma),int(y+math.sqrt(8*math.log(2))*sigma)+1):
            if y1>=0 and x1>=0 and y1<1010 and x1<1920:
                if x1==x and y1 ==y:
                    img[y1][x1]==255
                else:
                    img[y1][x1]=int(256*np.e**(-((x-x1)**2+(y-y1)**2)/2/sigma))

def find_spot(event,x,y,flags,param):
    global draw_next,spot_x,spot_y,sigma
    if event==cv2.EVENT_LBUTTONDBLCLK:
        if x in range(int(spot_x-math.sqrt(8*math.log(2))*sigma),int(spot_x+math.sqrt(8*math.log(2))*sigma)+1)\
             and y in range(int(spot_y-math.sqrt(8*math.log(2))*sigma),int(spot_y+math.sqrt(8*math.log(2))*sigma)+1):
            draw_next=True

img=np.zeros((1010,1920),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',find_spot)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if draw_next:
        cv2.imwrite('data/{},{}.png'.format(spot_x,spot_y),frame)
        img=np.zeros((1010,1920),np.uint8)
        spot_x=random.random()*1920
        spot_y=random.random()*1010
        draw_Gaussian(img,spot_x,spot_y,sigma)
        draw_next=False
    cv2.imshow('image',img)
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()

