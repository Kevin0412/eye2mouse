import cv2
import numpy as np
import os
import tqdm
import math

def draw_Gaussian(img,x,y,sigma):
    for x1 in range(int(x-math.sqrt(8*math.log(2))*sigma),int(x+math.sqrt(8*math.log(2))*sigma)+1):
        for y1 in range(int(y-math.sqrt(8*math.log(2))*sigma),int(y+math.sqrt(8*math.log(2))*sigma)+1):
            if y1>=0 and x1>=0 and y1<1010 and x1<1920:
                if x1==x and y1 ==y:
                    img[y1][x1]==255
                else:
                    img[y1][x1]=max(int(256*np.e**(-((x-x1)**2+(y-y1)**2)/2/sigma)),img[y1][x1])

img=np.zeros((1010,1920),np.uint8)

for root, dirs, files in os.walk("data"):
    for file in tqdm.tqdm(files):
        axis=file[0:-4].split(',')
        draw_Gaussian(img,float(axis[0]),float(axis[1]),3)

cv2.imwrite('collected_points.png',img)