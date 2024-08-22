import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import PIL
import SimpleITK as sitk
from PIL.Image import Resampling
from skimage.measure import find_contours
import torch

def sitk_load(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))


    return im_array, info


# 每个类别有自己的目录:  类别1/video0, video2/  frame01.png, frames01_gt.png
# 类别的名字:[camus_2ch_1, camus_2ch_2, camus_2ch_3, camus_4ch_1, camus_4ch_2, camus_4ch_3]
# transformed_meta.json
# transformed
#   /patient1
#       /2CH
#           /frames
#               /frame1.jpg
#               /frame2.jpg (0-255)
#           /gts
#               /frame1_gt1.png (0-1,bit)
#               /frame1_gt2.png
#               /frame1_gt3.png
#               /frame2.png (0,1,2,3)
#       /4CH
#   /patient2
#   /patient3


# % CAMUS: 有3个类别: 左心室的心内膜, 左心室的心外膜，左心房的心内膜, 
# % 每个病人 * 2个view (心尖四腔和心尖二腔视图) 
# % 每个view有3个部分: ed, half-sequence, es, 合并在一起成为一整个video
# class1, class2, class3
from tqdm import tqdm
from PIL import Image
import os
import shutil
database_nifti_root = Path("../uni_med_video/CAMUS/database_nifti/")
patients = sorted(os.listdir(database_nifti_root))

views = ['2CH', '4CH'] # 每个view有3个类别，这6个类别不相同，一个是2CH的左心房心内膜，一个是4CH的左心房心内膜
gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"
image_pattern = "{patient_name}_{view}_{instant}.nii.gz"

root = Path("../uni_med_video/CAMUS/transformed")
if os.path.exists(root):
    shutil.rmtree(root)
os.makedirs(root)

for pati in tqdm(patients):
    
    for view in views:
        cnt_dir = os.path.join(root, pati, view)
        os.makedirs(cnt_dir)

        # images, 0-255, float32
        ed_image, _ = sitk_load(database_nifti_root / pati / image_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='ED'))
        hf_images, _ = sitk_load(database_nifti_root / pati / image_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='half_sequence')) 
        es_image, _ = sitk_load(database_nifti_root / pati / image_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='ES')) 
        # gts, 0, 1, 2, 3, float32
        ed_gt, _ = sitk_load(database_nifti_root / pati / gt_mask_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='ED'))
        hf_gts, _ = sitk_load(database_nifti_root / pati / gt_mask_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='half_sequence')) 
        es_gt, _ = sitk_load(database_nifti_root / pati / gt_mask_pattern.format(patient_name=pati, 
                                                                                  view=view, 
                                                                                  instant='ES'))  
           
        assert ed_image.shape== ed_gt.shape
        assert hf_images.shape== hf_gts.shape
        assert es_image.shape== es_gt.shape      

        # t h w, 0-255, float32
        whole_video = torch.cat([
            torch.from_numpy(ed_image).unsqueeze(0), torch.from_numpy(hf_images), torch.from_numpy(es_image).unsqueeze(0)
        ], dim=0)
        
        # t h w, 0,1,2,3, float32
        whole_gts = torch.cat([
            torch.from_numpy(ed_gt).unsqueeze(0), torch.from_numpy(hf_gts), torch.from_numpy(es_gt).unsqueeze(0)
        ], dim=0) 

        assert len(whole_gts.unique()) == 4
        assert whole_gts.max() == 3
        assert whole_video.shape[1] == whole_gts.shape[1]

        os.makedirs(os.path.join(cnt_dir, 'frames'))
        for frame_idx, t in enumerate(whole_video.numpy()):
            img = Image.fromarray(t.astype(np.uint8), mode='L')
            img.save(os.path.join(cnt_dir, 'frames', f'{frame_idx:04d}.jpg'))
        
        os.makedirs(os.path.join(cnt_dir, 'gts'))
        for frame_idx, t in enumerate(whole_gts.numpy()):
            for cls in [1,2,3]:
                mask = Image.fromarray((t==cls).astype(np.uint8) * 255, mode='L')
                mask.save(os.path.join(cnt_dir, 'gts', f'{frame_idx:04d}_gt{cls}.png'))