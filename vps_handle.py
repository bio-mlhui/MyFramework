
import os
import cv2
import numpy as np

def preprocess(path_src, split_name):
    print('process', path_src)
    path_dst = path_src.replace(f'/{split_name}/', f'/{split_name}/uterus_myoma_WeakPolyP_{split_name}/')
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

split_name = 'temp8'
dst_dir = f'{split_name}/uterus_myoma_WeakPolyP_{split_name}'

root = '/home/xuhuihui/datasets/uterus_myoma/Dataset'

if not os.path.exists(os.path.join(root, dst_dir)):
    preprocess(os.path.join(root, f'{split_name}/test'), split_name=split_name)
    preprocess(os.path.join(root, f'{split_name}/train'), split_name=split_name)