import os
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d


import torch
from PIL import Image
root = '/home/xuhuihui/datasets/cardiacUDC/cardiacUDC_dataset'
sites = ['Site_R_126', 'Site_R_73', 'Site_R_52', 'Site_G_100', 'Site_G_29', 'Site_G_20', ]
def crop_contour(image, annotate_mask):
    # h w, 0-255, numpy
    import cv2
    import numpy as np

    # Step 2: Apply a threshold to convert the image to binary
    # You might need to adjust the threshold value to suit your image
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    # Step 3: Perform connected component analysis to find the largest component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Step 4: Identify the largest connected component (ignore label 0 which is the background)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a mask for the largest connected component
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255

    # Step 5: Perform contour detection on the mask
    contours, _ = cv2.findContours(largest_component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw the largest contour on a new mask (optional: skip if you just want the largest component)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Step 7: Apply the mask to the original image to keep only the largest connected component
    cropped_fan_shape = cv2.bitwise_and(image, image, mask=mask)

    # Step 8: (Optional) Crop the rectangular bounding box around the fan shape
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = cropped_fan_shape[y:y+h, x:x+w]
    if annotate_mask is not None:
        annotate_mask = annotate_mask[y:y+h, x:x+w]
    return cropped_image, annotate_mask

def contour_box(image):
    # h w, 0-255, numpy
    import cv2
    import numpy as np

    # Step 2: Apply a threshold to convert the image to binary
    # You might need to adjust the threshold value to suit your image
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    # Step 3: Perform connected component analysis to find the largest component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Step 4: Identify the largest connected component (ignore label 0 which is the background)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a mask for the largest connected component
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255

    # Step 5: Perform contour detection on the mask
    contours, _ = cv2.findContours(largest_component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw the largest contour on a new mask (optional: skip if you just want the largest component)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Step 8: (Optional) Crop the rectangular bounding box around the fan shape
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

import torch
# labeled/Frame_GT/video_id
# unlabeled/Frame/video_id
labeled_videos = []
unlabeled_videos = []
os.makedirs(os.path.join(root, 'labeled'), exist_ok=True)
os.makedirs(os.path.join(root, 'unlabeled'), exist_ok=True)
for site_id in sites:
    files = os.listdir(os.path.join(root, site_id))
    video_dirs = [a[:-13] for a in files if a.endswith('_image.nii.gz')]
    label_dirs = [a[:-13] for a in files if a.endswith('_label.nii.gz')]

    for vid in tqdm(video_dirs):
        saved_video_id = f'{site_id}_{vid}'
        if vid in label_dirs:
            labeled_videos.append(saved_video_id)
            unlabel = False
        else:
            unlabeled_videos.append(saved_video_id)
            unlabel = True
        save_dir = 'labeled' if not unlabel else 'unlabeled'

        img_sitk = sitk.ReadImage(join(root, site_id, f'{vid}_image.nii.gz'))
        image_data = sitk.GetArrayFromImage(img_sitk)
        assert len(image_data.shape) == 3
        assert image_data.dtype == np.uint8
        os.makedirs(os.path.join(root, save_dir, 'Frame', saved_video_id), exist_ok=True)
        
        if not unlabel:
            os.makedirs(os.path.join(root, save_dir, 'GT', saved_video_id), exist_ok=True)
            gt_sitk = sitk.ReadImage(join(root, site_id, f'{vid}_label.nii.gz'))
            gt_data_ori = sitk.GetArrayFromImage(gt_sitk)
            assert image_data.shape == gt_data_ori.shape
            assert gt_data_ori.dtype == np.int8
            assert gt_data_ori.max() <= 4

        # 第一帧确定box
        x, y, w, h = contour_box(image_data[0])
        
        for frame_idx in range(image_data.shape[0]):
            if not unlabel:
                image_mask = gt_data_ori[frame_idx][y:y+h, x:x+w]
            cropped_img = image_data[frame_idx][y:y+h, x:x+w]
            Image.fromarray(cropped_img, mode='L').save(os.path.join(root, save_dir, 'Frame', saved_video_id, f'{frame_idx:05d}.jpg'))

            if (not unlabel) and ((torch.from_numpy(image_mask) != 0).any()) :
                np.save(os.path.join(root, save_dir, 'GT', saved_video_id, f'{frame_idx:05d}.npy'), image_mask)


assert len(labeled_videos) == len(list(set(labeled_videos)))
assert len(unlabeled_videos) == len(list(set(unlabeled_videos)))
