import os

import cv2
import mediapipe as mp
import numpy as np

left_eye =[362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,473,474,475,476,477]
right_eye=[133,173,157,158,159,160,161,246, 33,  7,163,144,145,153,154,155,468,469,470,471,472]

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

def draw_2d(img,param):
    img_height, img_width, _ = img.shape
    for p in param:
        for i in range(478):
            x = int(p['keypt'][i,0])
            y = int(p['keypt'][i,1])
            if i>467:
                cv2.circle(img, (x, y), 1, (0,0,255), -1) # Red
            elif i in left_eye or i in right_eye:
                cv2.circle(img, (x, y), 1, (255,0,0), -1) # Blue
    return img

def save(pngname,param):
    out=''
    if pngname[-4:-1]=='.pn':
        out+=pngname[0:-4]
    else:
        out+=pngname
    for p in param:
        for eye in left_eye:
            for n in range(3):
                out+=','
                out+=str(p['joint'][eye][n])
    out+='\n'
    if pngname[-4:-1]=='.pn':
        out+=pngname[0:-4]
    else:
        out+=pngname
    for p in param:
        for eye in right_eye:
            for n in range(3):
                out+=','
                out+=str(p['joint'][eye][n])
    out+='\n'
    return out

file=open('data.csv','w',encoding='utf-8')

file.write('x,y,eye\n')
png_path=[]

for root, dirs, files in os.walk('data'):
    for file1 in files:
        if os.path.splitext(file1)[1] == '.png':
            png_path.append(str(os.path.join(root , file1)))

for n,png in zip(range(len(png_path)),png_path):
    img=cv2.imread(png)

    img= cv2.copyMakeBorder(
        img,int((img.shape[1]-img.shape[0]+abs(img.shape[1]-img.shape[0]))/4),
        int((img.shape[1]-img.shape[0]+abs(img.shape[1]-img.shape[0]))/4),
        int((img.shape[0]-img.shape[1]+abs(img.shape[0]-img.shape[1]))/4),
        int((img.shape[0]-img.shape[1]+abs(img.shape[0]-img.shape[1]))/4),
        cv2.BORDER_CONSTANT,value=[0,0,0]
    )

    param=get_face_data(img)
    file.write(save(png.split(os.sep)[1],param))
    img=draw_2d(img,param)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    print('\r{}/{}'.format(n,len(png_path)),end='')
file.close()
