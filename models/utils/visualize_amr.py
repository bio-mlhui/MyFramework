from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.visualizer import ColorMode
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx

def generate_instance_canvas(vid_frames, metadata, H, W, pred_mask):
    """pred_mask: h w, score:float"""
    istce_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([1]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()

def save_model_output(videos, text_query, amr, amr_tree_string,  directory, pred_masks, scores):
    # t 3 h w
    # nq t h w
    # vi nq 
    #
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tgt_dir = '/'.join(directory.split('/')[:-1])
    os.makedirs(tgt_dir, exist_ok=True)
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])
    metadata = MetadataCatalog.get('youtube_rvos')

    final_image = []
    # draw video frames
    vid_frames = videos.detach().cpu()
    vid_frames = invTrans(vid_frames)
    vid_frames = torch.clamp(vid_frames, min=0, max=1).permute(2, 0, 3, 1).flatten(1,2)  # t 3 h w -> h t w 3 -> h (t w) 3
    H, W = vid_frames.shape[:2]
    final_image.append(vid_frames*255)
    # draw refer preditions
    pred_masks = pred_masks.permute(0, 2, 1, 3).flatten(2,3).detach().cpu() # t nq h w -> nq h t w -> nq h (t w) 
    pred_masks = (F.interpolate(pred_masks.float().unsqueeze(0), size=[H, W], mode='bilinear', align_corners=False) > 0)[0]
    scores = scores.detach().cpu()# vi nq   
    _, map_nqs = scores.max(-1)
    num_instances = len(pred_masks)
    from joblib import Parallel, delayed
    import multiprocessing
    params_by_instance = [(vid_frames, metadata, H, W, pred_mask) for pred_mask in pred_masks]
    n_jobs = min(multiprocessing.cpu_count(), num_instances)
    instances_canvas = Parallel(n_jobs)(delayed(generate_instance_canvas)(*p) for p in params_by_instance)
    final_image.extend(instances_canvas) # h (t w) 3

    title = [text_query, amr_tree_string]
    amr_tree_lines = len(amr_tree_string.split('\n'))
    max_sentence_length = max([len(tit) for tit in title])
    num_sentences = 2 + amr_tree_lines

    assert amr.num_nodes == len(map_nqs)
    max_nq_string = ' '.join([f'{str(key)} / ' + str(max_nq_idx) + ';' for key, max_nq_idx in zip(list(range(amr.num_nodes)), map_nqs.tolist())])
    title.append(max_nq_string)
    title = '\n'.join(title)
    font_size = 20
    linespacing = 2
    whole_image = np.vstack(final_image) / 255.0 # (# h) (t w) 3

    fig_with = max(whole_image.shape[1], (font_size*max_sentence_length))
    fig_height = whole_image.shape[0] + (num_sentences+linespacing*(num_sentences-1)) * font_size

    sep = whole_image.shape[0] / float(fig_height)
    fig, axs = plt.subplots(1, 2, figsize=(fig_with/100.0, fig_height/100.0))
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[0].imshow(whole_image)
    axs[0].set_position([(0.5 - whole_image.shape[1]/(float(fig_with)*2)),
                        0, 
                        whole_image.shape[1]/float(fig_with), whole_image.shape[0]/float(fig_height)])

    axs[1].xaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    axs[1].set_position([0.3, sep, 1 - sep, 1 - sep])
    G = to_networkx(amr)
    options = {
        "font_size": 20,
        "node_color": "red",
        "edgecolors": "blue",
        "linewidths": 0,
        "width": 5,
        "ax": axs[1],
        "labels": {key: str(key) for key in range(amr.num_nodes)},
    }
    nx.draw(G,**options)

    fig.text(0, sep, title, fontsize=font_size, linespacing=linespacing,)
    fig.savefig(directory)
    plt.close()     

