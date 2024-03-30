import os
import numpy as np
import torch
from PIL import Image
from miccai_generate_Dir.visualize import save_model_output

method_names = ['GT', 'LGRNet', 'FLANet']
method_pred_dirs = [
    '/home/xuhuihui/datasets/uterus_myoma/Dataset/temp8/test/GT',
    '/home/xuhuihui/workspace/rvos_encoder/output/VIS/fibroid/pvt_localGlobal_tune_tmp7/epc[40_84]_iter[7626]_sap[30504]/eval_fibroid_validate_temp8/web',
    '/home/xuhuihui/workspace/rvos_encoder/output/VIS/fibroid/flanet/output/eval_fibroid_validate_temp8/web'
]
video_dir = '/home/xuhuihui/datasets/uterus_myoma/Dataset/temp8/test/Frame'

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f,)).convert('RGB') for f in frames]

# t' h w, int, obj_ids ;  has_ann t
def get_frames_mask(mask_path, video_id, frames):
    masks = [Image.open(os.path.join(mask_path, video_id, f)).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()

def generate_videos(video_dir,
                    method_names,
                    method_pred_dirs,
                    method_colors=None,
                    target_videos=None):

    video_ids = os.listdir(video_dir)


    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('youtube_rvos_gt').set(thing_classes = ['r', 'nr'],
                                            thing_colors = [(0, 128, 255), (0, 128, 255)])
    MetadataCatalog.get('youtube_rvos_pred').set(thing_classes = ['r', 'nr'],
                                            thing_colors = [(0., 204., 102.), (255., 204., 102.)])
    MetadataCatalog.get('youtube_rvos_flanet').set(thing_classes = ['r', 'nr'],
                                            thing_colors = [(255., 51., 51.), (255., 51., 51.)])
    # MetadataCatalog.get('youtube_rvos_flanet').set(thing_classes = ['r', 'nr'],
    #                                         thing_colors = [(255., 51., 51.), (255., 51., 51.)])
    import torchvision.transforms.functional as F
    save_video_dir = '/home/xuhuihui/workspace/rvos_encoder/miccai_generate_Dir'
    for vid in video_ids:
        if vid == 'IM_0004':
            vid_frames = os.listdir(os.path.join(video_dir, vid))
            frames = get_frames(video_dir, vid, frames=vid_frames) # list[PIL]
            
            tensor_video = torch.stack([F.to_tensor(frame) for frame in frames], dim=0) # t 3 h w, float, 0-1
            assert len(vid_frames) ==50
            
            gt_masks, _ = get_frames_mask(gt_dir, vid, frames=vid_frames) # t h w
            pred_masks, _ = get_frames_mask(pred_dir, vid, frames=vid_frames) # t h w
            flanet_masks, _ = get_frames_mask(fla_net_dir, vid, frames=vid_frames) # t h w)
            fps = 10
            save_model_output(videos=tensor_video, directory=os.path.join(save_video_dir, vid), file_name=f'{vid}.mp4', pred_masks = None,fps=fps)
            save_model_output(videos=tensor_video, directory=os.path.join(save_video_dir, vid), file_name=f'{vid}_gt.mp4', pred_masks = gt_masks.unsqueeze(0), 
                            scores=torch.tensor([[1]]), color='youtube_rvos_gt', fps=fps)
            save_model_output(videos=tensor_video, directory=os.path.join(save_video_dir, vid), file_name=f'{vid}_pred.mp4', pred_masks = pred_masks.unsqueeze(0), 
                            scores=torch.tensor([[1]]), color='youtube_rvos_pred', fps=fps)
            save_model_output(videos=tensor_video, directory=os.path.join(save_video_dir, vid), file_name=f'{vid}_flanet.mp4', pred_masks = flanet_masks.unsqueeze(0), 
                            scores=torch.tensor([[1]]), color='youtube_rvos_flanet', fps=fps)

    from moviepy.editor import VideoFileClip, clips_array, vfx, CompositeVideoClip, TextClip

    for vid in video_ids:
        if vid == 'IM_0004':
            clip1 = VideoFileClip(os.path.join(save_video_dir, vid, f'{vid}.mp4'))
            clip2 = VideoFileClip(os.path.join(save_video_dir, vid, f'{vid}_gt.mp4'))
            clip3 = VideoFileClip(os.path.join(save_video_dir, vid, f'{vid}_pred.mp4'))
            clip4 = VideoFileClip(os.path.join(save_video_dir, vid, f'{vid}_flanet.mp4'))
            margin = 10 
            final_clip = CompositeVideoClip([
                clip1.set_position((margin, 0)),
                clip2.set_position((clip1.w + margin*2, 0)),
                clip3.set_position((clip2.w * 2 + margin*3, 0)),
                clip4.set_position((clip3.w * 3 + margin*4, 0))
            ])
            title1 = TextClip("Video", fontsize=30, color='white').set_position((margin, clip1.h))
            title2 = TextClip("Ground Truth", fontsize=30, color='white').set_position((clip1.w + margin*2, clip1.h))
            title3 = TextClip("Our Method", fontsize=30, color='white').set_position((clip1.w * 2 + margin*3, clip1.h))
            title4 = TextClip("FLA-Net MICCAI'23", fontsize=30, color='white').set_position((clip1.w * 3 + margin*4, clip1.h))

            # common_text = TextClip("Common Text", fontsize=40, color='white').set_position(('center', clip1.h + title1.h + 50))

            final_clip = CompositeVideoClip([
                final_clip,
                title1,
                title2,
                title3,
                title4,
                # common_text
            ], size=(final_clip.w, final_clip.h + title1.h + 50))

            final_clip.write_videofile(os.path.join(save_video_dir, vid, f'{vid}_combine.mp4'))

        
    

