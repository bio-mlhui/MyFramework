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
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from .transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones
from einops import rearrange, reduce, repeat
_model_entrypoints = {}
def register_model(fn):
    model_name = fn.__name__
    if model_name in _model_entrypoints:
        raise ValueError(f'model name {model_name} has been registered')
    _model_entrypoints[model_name] = fn

    return fn
def model_entrypoint(model_name):
    try:
        return _model_entrypoints[model_name]
    except KeyError as e:
        print(f'Model Name {model_name} not found')

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



def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def ce_mask_loss(inputs, targets, num_boxes):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: 
            b=n_sigma thw
        targets: b=n_sigma thw
            (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # n_sigma=b thw
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # mean(b mean(thw)), 对于a2d来说，num_boxes=
    return ce_loss.mean(1).sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1) # [N, M, THW]
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1) # [N, M, THW]

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2) # [N, M]

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

class Fpn2D(nn.Module):
    def __init__(self, dim, cascaded_scales) -> None:
        """
        cascaded_scales: ['1','4'],  ['1','16'], ['1','32']
        """
        super().__init__()
        # from small to big
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert len(cascaded_scales) > 1
        cascaded_scales = cascaded_scales[::-1] # ['1','32'], ['1','16'], ['1','4'],
        for (temporal_stride, spatial_stride), (next_temporal_stride, next_spatial_stride) \
            in zip(cascaded_scales[:-1], cascaded_scales[1:]):
            assert temporal_stride == next_temporal_stride, 'the temporal stride must be the same for the FPN 2D'
            self.adapters.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.upsamples.append(nn.Upsample(scale_factor=spatial_stride//next_spatial_stride, mode='bilinear'))
            self.convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(32, dim))
        
        self.cascaded_scales = cascaded_scales
    
    def forward(self, multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales):
        """ bt c h w"""
        idxs = find_scales_from_multiscales(video_feat_scales, self.cascaded_scales) 
        fused_feats = [multiscales[idx] for idx in idxs]  # 从小到大

        for idx, (small_feat, large_feat) in enumerate(zip(fused_feats[:-1], fused_feats[1:])): # from small map to large map 
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feat) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            fused_feats[idx+1] = large_feat
        
        for idx, scale_idx in enumerate(idxs):
            multiscales[scale_idx] = fused_feats[idx]

        return multiscales

class Fpn2D_multiple(nn.Module):
    def __init__(self, dim, cascaded_scales) -> None:
        """
        cascaded_scales: ['1','4'],  ['1','16'], ['1','32']
        """
        super().__init__()
        # from small to big
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert len(cascaded_scales) > 1
        cascaded_scales = cascaded_scales[::-1] # ['1','32'], ['1','16'], ['1','4'],
        for (temporal_stride, spatial_stride), (next_temporal_stride, next_spatial_stride) \
            in zip(cascaded_scales[:-1], cascaded_scales[1:]):
            assert temporal_stride == next_temporal_stride, 'the temporal stride must be the same for the FPN 2D'
            self.adapters.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.upsamples.append(nn.Upsample(scale_factor=spatial_stride//next_spatial_stride, mode='bilinear'))
            self.convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(32, dim))
        
        self.cascaded_scales = cascaded_scales
    
    def forward(self, multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales):
        """ bt c h w"""
        idxs = find_scales_from_multiscales(video_feat_scales, self.cascaded_scales) 
        fused_feats = [multiscales[idx] for idx in idxs]  # 从小到大
        new_fused_feats = []
        new_fused_feats.append(fused_feats[0])
        for idx, large_feat in enumerate(fused_feats[1:]): # from small map to large map 
            small_feats = new_fused_feats[-1]
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feats) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            new_fused_feats.append(large_feat)
        
        for idx, scale_idx in enumerate(idxs):
            multiscales[scale_idx] = new_fused_feats[idx]

        return multiscales

class DeformVideo2D_with_FPN(nn.Module):
    def __init__(self, 
                 d_model,
                d_ffn=2048,
                dropout=0.,
                activation='relu',
                nheads=8,
                # important
                fused_scales=None, 
                fpn_strides=None,

                npoints=4, 
                nlayers=6,
                 ) -> None:
        super().__init__()
        n_levels = len(fused_scales)
        self.fused_scales = fused_scales
        encoder = DeformableTransformerEncoder(
                DeformableTransformerEncoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    n_levels=n_levels,
                    n_heads=nheads,
                    n_points=npoints,
                ),
                nlayers
        )
        self.deform_encoder = encoder
        self.level_embed = nn.Embedding(n_levels, d_model)
        self.num_feature_levels = n_levels

        if fpn_strides is not None:
            self.fpn = Fpn2D(dim=d_model, cascaded_scales=fpn_strides)
        else:
            self.fpn = None
        
    def get_valid_ratio(self, mask):
        """
        Input:
            - mask:
                bt h w
        Output:
            - int
        """
        _, H, W = mask.shape
        # T(bt, )
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # T(bt, )
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # T(bt, 2)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, multiscales, multiscales_pad_masks, multiscales_poses, video_feat_scales):
        """ bt c h w"""
        fused_scale_idxs = find_scales_from_multiscales(video_feat_scales, self.fused_scales)
        srcs = [multiscales[idx] for idx in fused_scale_idxs]
        masks = [multiscales_pad_masks[idx] for idx in fused_scale_idxs]
        pos_embeds = [multiscales_poses[idx] for idx in fused_scale_idxs]

        src_flatten = []
        mask_flattn = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bt, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = rearrange(src, 'bt c h w -> bt (h w) c')
            mask = rearrange(mask, 'bt h w -> bt (h w)')
            pos_embed = rearrange(pos_embed, 'bt c h w -> bt (h w) c')
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl][None, None, :]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
            src_flatten.append(src)
            mask_flattn.append(mask)
            
        # bt \sigma(hi wi) c
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flattn, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        
        # #levels, 2
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # (0, h0*wo, h1*w1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # bt num_levels 2
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # bt (h_sigma, w_sigma) c  # bt hw_sigma heads num_scales npoints 2
        memory, sampling_locations_by_layer, attention_weights_by_layer = \
            self.deform_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                lvl_pos_embed_flatten, mask_flatten)
        
        memory_features = []
        spatial_index = 0
        for lvl in range(self.num_feature_levels):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index: (spatial_index + h*w), :].contiguous()
            memory_lvl = rearrange(memory_lvl, 'bt (h w) c -> bt c h w',h=h, w=w)
            memory_features.append(memory_lvl)
            spatial_index += h*w
        
        for idx, scale_idx in enumerate(fused_scale_idxs):
            multiscales[scale_idx] = memory_features[idx]

        multiscales = self.fpn(multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales)
        return multiscales, sampling_locations_by_layer, attention_weights_by_layer

class Scale32CatText_Encoder_FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

def get_parsing_encoder(name, configs):
    if name == 'deform_video_2d_fpn':
        return DeformVideo2D_with_FPN(**configs)
    
    elif name == 'split_obj_ref_deform_video_2d_fpn':
        obj_seg_nlayers = configs.pop('obj_seg_nlayers')
        ref_seg_nlayers = configs.pop('ref_seg_nlayers')
        assert obj_seg_nlayers > 0
        obj_parsing_encoder = DeformVideo2D_with_FPN(**configs, nlayers=obj_seg_nlayers)
        if ref_seg_nlayers == 0:
            ref_parsing_encoder = None
        else:
            ref_parsing_encoder = DeformVideo2D_with_FPN(**configs, nlayers=ref_seg_nlayers)
        return obj_parsing_encoder, ref_parsing_encoder
    elif name == 'fpn2d':
        return Fpn2D_multiple(dim=configs['d_model'],
                     cascaded_scales=configs['cascaded_scales'])
    else:
        raise ValueError()

def get_fusion(name, configs):
    if name == 'VisionLanguageFusionModule':
        return VisionLanguageFusionModule(**configs)
    elif name == 'self_encoder':
        encoder_nlayers = configs.pop('nlayers')
        return TransformerEncoder(
            TransformerEncoderLayer(
                **configs
            ),encoder_nlayers
        )
    elif name == 'none':
        return None

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None):
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask) # b tgt src, float, 0,1
        tgt = tgt * tgt2
        return tgt, attn_weights
