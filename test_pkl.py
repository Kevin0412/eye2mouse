import math
import os

import cv2
import mediapipe as mp
import numpy as np
import torch

sigma=10
left_eye =[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,473,474,475,476,477]
right_eye=[133,173,157,158,159,160,161,246, 33,  7,163,144,145,153,154,155,468,469,470,471,472]

def draw_Gaussian(img,x,y,sigma):
    for x1 in range(int(x-math.sqrt(8*math.log(2))*sigma),int(x+math.sqrt(8*math.log(2))*sigma)+1):
        for y1 in range(int(y-math.sqrt(8*math.log(2))*sigma),int(y+math.sqrt(8*math.log(2))*sigma)+1):
            if y1>=0 and x1>=0 and y1<1010 and x1<1920:
                if x1==x and y1 ==y:
                    img[y1][x1]==255
                else:
                    img[y1][x1]=max(int(256*np.e**(-((x-x1)**2+(y-y1)**2)/2/sigma)),img[y1][x1])

def get_face_data(img):
    mp_faces = mp.solutions.face_mesh
    pipe = mp_faces.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    param = [{
        'detect': False,           # Boolean to indicate whether a face is detected
        'keypt' : np.zeros((478,2)), # 2D keypt in image coordinate (pixel)
        'joint' : np.zeros((478,3)), # 3D joint in relative coordinate
    }]

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract result
    result = pipe.process(img)

    # Convert mediapipe result to my own param
    img_height, img_width, _ = img.shape

    # Reset param
    for p in param:
        p['detect'] = False

    if result.multi_face_landmarks is not None:
        # Loop through different faces
        for i, res in enumerate(result.multi_face_landmarks):
            param[i]['detect'] = True
            # Loop through 468 landmark for each face
            for j, lm in enumerate(res.landmark):
                param[i]['keypt'][j,0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                param[i]['keypt'][j,1] = lm.y * img_height # Convert normalized coor to pixel [0,1] -> [0,height]

                param[i]['joint'][j,0] = lm.x
                param[i]['joint'][j,1] = lm.y
                param[i]['joint'][j,2] = lm.z

    pipe.close()

    return param

def get_eye(net,param):
    param_l=[]
    param_r=[]
    for p in param:
        for eye in left_eye:
            for n in range(3):
                param_l.append(p['joint'][eye][n])
        for eye in right_eye:
            for n in range(3):
                param_r.append(p['joint'][eye][n])
    L=np.array(param_l,np.float32)
    left=torch.from_numpy(L)
    prediction_l = net(left)
    R=np.array(param_r,np.float32)
    right=torch.from_numpy(R)
    prediction_r = net(right)
    return prediction_l.detach().numpy(),prediction_r.detach().numpy()

net=torch.load('best.pkl')
print(net)

cap = cv2.VideoCapture(0)

lefts=[]
rights=[]
while(True):
    ret, frame = cap.read()
    param=get_face_data(frame)
    L,R=get_eye(net,param)
    if len(lefts)==16:
        del lefts[0]
        del rights[0]
    lefts.append([L[0]*1920,L[1]*1010])
    rights.append([R[0]*1920,R[1]*1010])
    img=np.zeros((1010,1920,3),np.uint8)

    Lx,Ly,Rx,Ry = 0, 0, 0, 0
    for left,right in zip(lefts,rights):
        Lx+=left[0]
        Ly+=left[1]
        Rx+=right[0]
        Ry+=right[1]
    Lx=Lx/len(lefts)
    Ly=Ly/len(lefts)
    Rx=Rx/len(rights)
    Ry=Ry/len(rights)
    VarL=np.var(lefts, ddof=1)**0.5
    VarR=np.var(rights, ddof=1)**0.5
    print(VarL,VarR)
    draw_Gaussian(img[:,:,0],Lx,Ly,sigma)
    draw_Gaussian(img[:,:,2],Lx,Ly,sigma)
    draw_Gaussian(img[:,:,1],Rx,Ry,sigma)
    cv2.imshow('image',img)
    if cv2.waitKey(1)&0xFF==27:
        break