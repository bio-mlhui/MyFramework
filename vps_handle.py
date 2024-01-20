
import os
import cv2
import numpy as np


root = '/home/xuhuihui/datasets/SUN/SUN-SEG2'

def preprocess(path_src):
    print('process', path_src)
    path_dst = path_src.replace('/SUN-SEG2/', '/WeakPolyp-Processed/')
    for folder in os.listdir(path_src+'/Frame'):
        print(folder)
        for name in os.listdir(path_src+'/Frame/'+folder):
            image    = cv2.imread(path_src+'/Frame/'+folder+'/'+name)
            image    = cv2.resize(image, (352,352), interpolation=cv2.INTER_LINEAR)
            mask     = cv2.imread(path_src+'/GT/'+folder+'/'+name.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
            mask     = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            box      = np.zeros_like(mask)
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                box[y:y+h, x:x+w] = 255
            
            os.makedirs(path_dst+'/Frame/'+folder, exist_ok=True)
            cv2.imwrite(path_dst+'/Frame/'+folder+'/'+name, image)
            os.makedirs(path_dst+'/GT/'   +folder, exist_ok=True)
            cv2.imwrite(path_dst+'/GT/'   +folder+'/'+name.replace('.jpg', '.png'), mask)
            os.makedirs(path_dst+'/Box/'  +folder, exist_ok=True)
            cv2.imwrite(path_dst+'/Box/'  +folder+'/'+name.replace('.jpg', '.png'), box)

if not os.path.exists('/home/xuhuihui/datasets/SUN/WeakPolyp-Processed/'):
    preprocess('/home/xuhuihui/datasets/SUN/SUN-SEG2/TrainDataset')
    preprocess('/home/xuhuihui/datasets/SUN/SUN-SEG2/TestEasyDataset/Seen')
    preprocess('/home/xuhuihui/datasets/SUN/SUN-SEG2/TestEasyDataset/Unseen')
    preprocess('/home/xuhuihui/datasets/SUN/SUN-SEG2/TestHardDataset/Seen')
    preprocess('/home/xuhuihui/datasets/SUN/SUN-SEG2/TestHardDataset/Unseen')