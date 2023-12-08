import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import copy

_model_entrypoints = {}
def register_model(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn

    return fn
def model_entrypoint(model_name):
    try:
        return _model_entrypoints[model_name]
    except KeyError as e:
        print(f'RVOS moel {model_name} not found')

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

def pad_1d_feats(feat_list):
    # list[ni c] -> b nmax c
    feat_len = [len(feat) for feat in feat_list]
    n_max = max(feat_len) 
    batch_size = len(feat_list)
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
    for i in range(batch_size):
        feat_list[i] = F.pad(feat_list[i].clone(), pad=[0, 0, 0, n_max-feat_len[i]])
        pad_mask[i, :feat_len[i]] = False
    feat_list = torch.stack(feat_list, dim=0) # b nmax c
    return feat_list, pad_mask

def find_scale_from_multiscales(multiscale_des, scale_des):
    """
    multiscale_des: list(['1','4'], ['1','8']), ['1','8']
    """
    retrieved_idx = []
    for idx, scale in enumerate(multiscale_des):
        if (scale_des[0] == scale[0]) and (scale_des[1] == scale[1]):
            retrieved_idx.append(idx)
    assert len(retrieved_idx) == 1
    return retrieved_idx[0]

def find_scales_from_multiscales(multiscale_des, scale_deses):
    """
    multiscale_des: list(['1','4'], ['1','8']), ['1','8']
    """
    output = []
    for scale_des in scale_deses:
        output.append(find_scale_from_multiscales(multiscale_des, scale_des))
    return output

def get_optimizer(param_dicts, configs):
    name = configs['name']
    lr = configs['lr']
    wd = configs['wd']
    if name == 'AdamW':
        return torch.optim.AdamW(param_dicts,
                                 lr=lr,
                                 weight_decay=wd
                                 )
    elif name == 'Lion':
        from lion_pytorch import Lion
        return Lion(param_dicts, 
                    lr=lr, 
                    weight_decay=wd,
                    use_triton=False)
    
    else:
        raise NotImplementedError
    
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def polynomial_decay_lambda(step, initial_learning_rate : float=8e-5, end_learning_rate: float=1.5e-5, decay_steps=25, power=1.0):
    step = min(step, decay_steps)
    return (((initial_learning_rate - end_learning_rate) *
            ((1 - step / decay_steps) ** (power))
            ) + end_learning_rate) / initial_learning_rate

def inverse_sqrt_warmup_lambda(step, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if step < num_warmup_steps:
        return float(step) / float(max(1, num_warmup_steps))
    return max(0.0, (num_warmup_steps / step)**0.5)

def generate_instance_canvas(vid_frames, metadata, H, W, pred_mask, score,):
    """pred_mask: h w, score:float"""
    istce_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([score]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()


def visualization_for_AMR_V0(videos, text_query, directory,
                                targets=None,
                                masked_text_tokenized=None, masked_text_gt=None, mlm_pred=None,
                                masked_video=None,mvm_pred=None, 
                                refer_pred=None,
                                losses=None,
                                draw_all_instances=False,):
    """ Training / Validation, 
    Evalution:
        对于youtube-rvos来说, 没有targets: 只visualize模型的预测, 可以做到不提交, 只观察模型是否对某几个sample产生足够好的效果
        对于其他数据集来说, targets在上一级上: 调试的时候, 可以从上一级上把targets拿过来
    Trainnig:
        targets非空

    videos: t b 3 h w
    text_query: list[str], batch
    masked_sentence: list[str],
    masked_video: t b 3 h w
    targets:
    mlm_pred: list[list[list[], 预测的最可能的10个token和概率], 被masked掉的token的数量], 一个batch的大小
    mvm_pred: same shape as videos
    refer_pred: {pred_masks, pred_logits}
    losses: {'loss_mask', 'loss_dice', '}
    """
    import torchvision.transforms as transforms
    from detectron2.utils.visualizer import Visualizer
    from detectron2.structures import Instances
    from detectron2.data import MetadataCatalog
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    os.makedirs(directory, exist_ok=True)
    nf, batch_size, *_= videos.shape
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])
    metadata = MetadataCatalog.get('youtube_rvos')
    for batch_idx in range(batch_size):
        final_image = [] # all images are in [0, 255]
        # draw video frames
        vid_frames = videos[:, batch_idx].detach().cpu()
        vid_frames = invTrans(vid_frames)
        vid_frames = torch.clamp(vid_frames, min=0, max=1).permute(2, 0, 3, 1).flatten(1,2)  # t 3 h w -> h t w 3 -> h (t w) 3
        H, W = vid_frames.shape[:2]
        final_image.append(vid_frames*255)
        # draw refer preditions
        if refer_pred is not None:
            refer_pred_mask = refer_pred['pred_masks'][batch_idx].permute(1, 2, 0, 3).flatten(2,3).detach().cpu() # t nq h w -> nq h t w -> nq h (t w) 
            refer_pred_mask = (F.interpolate(refer_pred_mask.float().unsqueeze(0), size=[H, W], mode='bilinear', align_corners=False) > 0)[0]
            refer_pred_class = refer_pred['pred_logits'][batch_idx].detach().cpu().softmax(-1)[:, 0]   # nq

            # scores: list[float]
            # pred_classes: list[int]
            # cls_score, cls_idx= refer_pred_class.softmax(-1).max(dim=-1)

            # cls_score = refer_pred_class.softmax(-1)[:, 0]
            # cls_idx = torch.zeros([len(refer_pred_class)]).long() # nq
            if not draw_all_instances: #只画score最高的
                refer_pred_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
                max_score, max_score_idx = refer_pred_class.max(-1)
                instances = Instances([H, W], 
                    pred_masks=refer_pred_mask[[max_score_idx], :], # 1 H W
                    scores=torch.tensor([max_score]), # 1,
                    pred_classes=torch.tensor([0]) # 1,
                )
                refer_pred_canvas.draw_instance_predictions(instances)
                refer_pred_canvas = refer_pred_canvas.get_output()
                final_image.append(refer_pred_canvas.get_image()) # h (t w) 3
            else:
                num_instances = len(refer_pred_mask)
                from joblib import Parallel, delayed
                import multiprocessing
                params_by_instance = [(vid_frames, metadata, H, W, pred_mask, score.item()) for pred_mask, score in zip(refer_pred_mask,
                                                                                                                  refer_pred_class)]
                n_jobs = min(multiprocessing.cpu_count(), num_instances)
                instances_canvas = Parallel(n_jobs)(delayed(generate_instance_canvas)(*p) for p in params_by_instance)
                final_image.extend(instances_canvas) # h (t w) 3

                # for istce_idx in range(num_instances): # nq (t h) w
                #     istce_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata)
                #     istce = Instances([H, W], 
                #         pred_masks=refer_pred_mask[[istce_idx], :], # 1 H W
                #         scores=torch.tensor([refer_pred_class.softmax(-1)[istce_idx, 0]]), # 1,
                #         pred_classes=torch.tensor([0]) # 1,
                #     )
                #     istce_canvas.draw_instance_predictions(istce)
                #     istce_canvas = istce_canvas.get_output()
                #     final_image.append(istce_canvas.get_image()) # (t h) w 3

        # draw refer ground truth
        if targets is not None:
            gt_referent_mask = targets[batch_idx]['masks'].permute(1, 0, 2).flatten(1, 2).detach().cpu() # t h w -> h t w -> h (t w)
            refer_gt_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata)
            refer_gt_canvas.overlay_instances(masks=[gt_referent_mask.numpy()], alpha=0.5,
                                              assigned_colors=[(1., 0.549, 0)]) # assigned_colors
            refer_gt_canvas = refer_gt_canvas.get_output() # h (t w) 3
            final_image.append(refer_gt_canvas.get_image())

        # draw masked video gt & predictions
        if mvm_pred is not None:
            assert masked_video is not None
            mv = masked_video[:, batch_idx].permute(2, 0, 3, 1).flatten(1,2).detach().cpu()  # t 3 h w -> h t w 3 -> h (t w) 3
            mv_pred = mvm_pred[:, batch_idx].permute(2, 0, 3, 1).flatten(1,2).detach().cpu() # 
            final_image.append(mv*255) 
            final_image.append(mv_pred*255)         

        title = [text_query[batch_idx]]
        # set mlm as additional title 
        if mlm_pred is not None:
            assert masked_text_gt is not None
            assert masked_text_tokenized is not None
            (tokenized_sentence, _, masked_sentence) = masked_text_tokenized
            tknd_sentence = tokenized_sentence[batch_idx]  # list[tokens]
            title.append(' '.join(tknd_sentence))
            title.append(masked_sentence[batch_idx])
            token_masked_bool = masked_text_gt['token_masked_bool']
            mlm_bool_idx = token_masked_bool[batch_idx].nonzero(as_tuple=True)[0].cpu()

            for idx, masked_token_pred_list in zip(mlm_bool_idx, mlm_pred[batch_idx]):
                num_token_shown = 5
                # 10个最可能的token 和对应的probability
                tit = f'{tknd_sentence[idx]}: '
                for token, prob in masked_token_pred_list[:num_token_shown]:
                    tit = f'{tit}{token}({prob:.2f}), '
                title.append(tit)
        
        if losses is not None:
            tit = '(batch loss) '
            for k,v in losses.items():
                tit = f'{tit}{k}: {v.item()}, '
            title.append(tit)
        
        max_sentence_length = max([len(tit) for tit in title])
        num_sentences = len(title)
        title = '\n'.join(title)

        font_size = 20
        linespacing = 2
        whole_image = np.vstack(final_image) / 255.0 # (# h) (t w) 3

        fig_with = max(whole_image.shape[1], (font_size*max_sentence_length))
        fig_height = whole_image.shape[0] + (num_sentences+linespacing*(num_sentences-1)) * font_size

        sep = whole_image.shape[0] / float(fig_height)
        fig, axs = plt.subplots(figsize=(fig_with/100.0, fig_height/100.0))
        axs.xaxis.set_visible(False)
        axs.yaxis.set_visible(False)
        axs.imshow(whole_image)
        axs.set_position([(0.5 - whole_image.shape[1]/(float(fig_with)*2)),
                          0, 
                          whole_image.shape[1]/float(fig_with), whole_image.shape[0]/float(fig_height)])
        fig.text(0, sep, title, fontsize=font_size, linespacing=linespacing,)
        fig.savefig(os.path.join(directory,f'sample{batch_idx}.png'))
        plt.close()  
