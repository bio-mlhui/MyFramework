import random
import logging
import sys
import os
import math
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
import numpy as np
from typing import Any, Optional
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms as transforms
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.visualizer import ColorMode

from einops import repeat, reduce, rearrange
from util.misc import get_total_grad_norm
from .position_encoding import build_position_encoding
from util.misc import NestedTensor
import matplotlib.pyplot as plt
import copy
from .encoder_fusion import fusion_entrypoints
from .encoder_video import video_encoder_entrypoints
from .encoder_text import text_encoder_entrypoints


from .encoder_video_text import video_text_encoder_entrypoints, build_video_clip
from torch_geometric.data import Data
from .decoder_refer import refer_decoder_entrypoints
# 根据使用的backbone的不同, 将rvos model分成了
# 1. 使用了单模态encoder
# 2. 使用了多模态encoder
# 3. 使用了多模态+query encoder


# helpers
# 1. 把optimization也看作 model的一部分
# 2. train/validate的时候, 把模型的输出可视化进行debug
def targets_refer_handler(targets, mask_size):
    H, W = mask_size
    targets = list(zip(*targets)) # list[list[dict, n h w], batch], nf -> list[list[dict, n h w], nf], b
    outputs = []
    for batch in targets:
        batch_out = {}
        masks = torch.stack([time_batch['masks'][time_batch['referred_instance_idx']] for time_batch in batch],dim=0) # nf h w
        
        valid = torch.stack([time_batch['valid'][time_batch['referred_instance_idx']] for time_batch in batch],dim=0).long() # nf, 1/0
        masks = F.pad(masks,  pad=(0, W-masks.shape[2], 0, H-masks.shape[1]), value=0)
        batch_out['masks'] = masks

        batch_out['valid'] = valid # nf
            
        outputs.append(batch_out)
    return outputs 
    
def targets_all_instance_handler(targets, mask_size, class_token_id_map):
    H, W = mask_size
    targets = list(zip(*targets)) # list[list[dict], batch], nf -> list[list[dict], nf], b
    outputs = []
    for batch in targets:
        batch_out = {}
        labels = batch[0]['labels'] # n, [5,2,6,1,0]
        batch_out['labels'] = labels # n
        if class_token_id_map is not None:
            batch_out['token'] = torch.tensor([class_token_id_map[lab.item()] for lab in labels], device=labels.device).long()
        
        masks = torch.stack([time_batch['masks'] for time_batch in batch], dim=1) # list[n h w] t -> n t H W
        masks = F.pad(masks,  pad=(0, W-masks.shape[3], 0, H-masks.shape[2]), value=0)
        batch_out['masks'] = masks # n t h w
        
        valid = torch.stack([time_batch['valid'] for time_batch in batch], dim=1)# list[n] t -> n t
        batch_out['valid'] = valid  # n t
        
        refer_idx = batch[0]['referred_instance_idx'] # int
        refer_labels = torch.ones(len(labels), device=labels.device).long() # n
        if valid[refer_idx].any():
            refer_labels[refer_idx] = 0  # n
        batch_out['refer_labels'] = refer_labels # n
        
        outputs.append(batch_out)
    return outputs

def get_optimizer(param_dicts, configs):
    if configs.name == 'AdamW':
        return torch.optim.AdamW(param_dicts,
                                 lr=configs.lr,
                                 weight_decay=configs.wd
                                 )
    elif configs.name == 'Lion':
        from lion_pytorch import Lion
        return Lion(param_dicts, 
                    lr=configs.lr, 
                    weight_decay=configs.wd,
                    use_triton=False)
    
    else:
        raise NotImplementedError

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

def pad_1ds_with_grad(data):
    """list[(n_i c), float] -> (b n_max c, b n_max) 1是Padding的位置 
    """
    batch_size, device = len(data), data[0].device
    n_max = max([len(d) for d in data])
    
    out = []
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=device)
    for i in range(batch_size):
        out.append(F.pad(data[i], [0, 0, 0, n_max-len(data[i])]))
        pad_mask[i, :(len(data[i]))] = False
    out = torch.stack(out, dim=0) # b n_max c
    return out, pad_mask

# 存储一个batch的debug
def save_model_output(videos, text_query, directory,
                        targets=None,
                        masked_text_tokenized=None, masked_text_gt=None, mlm_pred=None,
                        masked_video=None,mvm_pred=None, 
                        refer_pred=None,
                        losses=None,
                        draw_all_instances=False,
                        ):
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
  

# 如果考虑还要做mlm/mvm, 所有rvos model的基类
# 如果不考虑mlm/mvm, 那么可以重新写一个类
# 1. 三种任务是否做, do_refer, do_mvm, do_mlm
# 2. loss dict
# 3. mode形式: finetune/pretrain
# 4. mlm/mvm抽取mask的方式
# 5. clip_configs: 如果使用clip
class Video_Model(nn.Module):

    def __init__(self,
                 
                 do_refer,
                 do_mvm,
                 do_mlm,

                 weight_dict,
   
                 mode_name,
                 
                 mask_sampling,
                 clip_configs,
                 d_model,

                 ) -> None:
        super().__init__()

        weight_dict = vars(weight_dict)
        self.weight_dict = weight_dict

        Video_Model.check_mode(weight_dict, do_mlm=do_mlm, do_mvm=do_mvm, do_refer=do_refer,
                   mode_name=mode_name)
        
        self.do_refer = do_refer
        self.do_mvm = do_mvm
        self.do_mlm = do_mlm

        self.mode_name = mode_name
        self.mask_sampling = mask_sampling
        
        if mask_sampling == 'gradient':
            # gradient必须是现有主任务的计算
            assert self.mode_name == 'joint'

        if mask_sampling == 'clip':
            assert clip_configs is not {}
            create_clip = video_text_encoder_entrypoints(clip_configs.name)
            self.clip = create_clip(clip_configs, d_model=d_model)
        
        self.d_model = d_model      

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, visualize=False, targets=None, saved_path=None):
        pass
        # return {'pred_masks':, 'pred_is_referred':,}

    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        pass
        # return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

    @classmethod
    def check_mode(cls, weight_dict : dict, do_refer, do_mlm, do_mvm, mode_name):
        wd = copy.deepcopy(weight_dict)

        loss_refer = wd.pop('loss_refer', 0)
        loss_classes = wd.pop('loss_classes', 0)
        loss_dice = wd.pop('loss_dice', 0)
        loss_mask = wd.pop('loss_mask', 0)

        loss_contrastive = wd.pop('loss_contrastive', 0)
        loss_matching = wd.pop('loss_matching', 0)

        loss_mvm = wd.pop('loss_mvm', 0)
        loss_mlm = wd.pop('loss_mlm', 0)

        if do_refer:
            assert (loss_refer !=0) or (loss_classes!=0) or (loss_dice !=0)\
                or (loss_mask != 0) or (loss_contrastive != 0) or (loss_matching !=0)
            
        if do_mlm:
            assert (loss_mlm !=0)

        if do_mvm:
            assert (loss_mvm !=0) 


        if mode_name == 'joint':
            assert (do_refer) and (do_mvm or do_mlm)
        elif mode_name == 'pretrain':
            assert (not do_refer) and (do_mvm or do_mlm)
        elif mode_name == 'finetune':
            assert (do_refer) and (not do_mvm) and (not do_mlm)
        else:
            raise ValueError()

# 两个backbone输出的feature经过fusion_encoder进行融合
# 对于mlm, 融合后的text feature输入到text encoder的forward_mlm中
# 对于mvm, 融合后的video feature输入到video encoder的foreard_mvm中
# 对于refer, 融合后的multimodal feature输入到fusion encoder的forward_refer中
class UnimodalEncoder_Model(Video_Model):
    def __init__(self, do_refer, do_mvm, do_mlm, weight_dict, mode_name, mask_sampling, clip_configs, d_model,
                 
                video_encoder_configs,
                text_encoder_configs,
                fusion_encoder_configs,
                decoder_configs) -> None:
        
        super().__init__(do_refer, do_mvm, do_mlm, weight_dict, mode_name, mask_sampling, clip_configs, d_model)

        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        video_encoder_with_mvm_head = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        text_encoder_with_mlm_head = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        fusion_encoder_with_refer_decoder = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)

        # 定义decoder
        create_decoder = refer_decoder_entrypoints(decoder_configs.name)
        self.decoder = create_decoder(fusion_encoder_configs.decoder, d_model=d_model)
        
        self.video_encoder = video_encoder_with_mvm_head
        self.text_encoder = text_encoder_with_mlm_head

        self.text_encoder.mask_sampling = self.mask_sampling
        self.video_encoder.mask_sampling = self.mask_sampling

        self.fusion_encoder = fusion_encoder_with_refer_decoder
        
        # 由于两个encoder是分开的, 所以有些任务可以使用其他任务forward的特征
        self.need_maskvideo = False
        self.need_masktext= False
        self.need_video=False
        self.need_text=False

        if self.do_mvm:
            self.need_maskvideo = True
            self.need_text = True
        if self.do_mlm:
            self.need_masktext = True
            self.need_video = True
        if self.do_refer:
            self.need_video = True
            self.need_text = True
            
        assert self.need_maskvideo or self.need_video or self.need_text or self.need_masktext 

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        text_encoder_output = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=None,
                                    device=device,
                                    mask_sentence=False, masked_indices=None)  
                
        video_encoder_output = self.video_encoder(videos=samples.tensors,
                                    video_pad_mask=samples.mask,
                                    valid_indices=valid_indices,
                                    mask_video=False, masked_indices=None) 
        
    
        refer_text_args =  {
            'token_feats': text_encoder_output['token_feats'],
            'token_pad_masks': text_encoder_output['token_pad_masks'],
            'token_sentence_feats': text_encoder_output['token_sentence_feats']
        } 
        refer_video_args = {
            'multiscales': video_encoder_output['multiscales'],
            'multiscale_pad_masks': video_encoder_output['multiscale_pad_masks'],
            'multiscale_poses': video_encoder_output['multiscale_poses'],
            'multiscale_des': video_encoder_output['multiscale_des']
        }

        fused_encoder_output = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
                                
        refer_video_args['multiscales'] =  fused_encoder_output['fused_video_feats']
        refer_text_args['token_feats'] = fused_encoder_output['fused_text_feats']
        
        
        out, _ = self.decoder(refer_video_args,refer_text_args,
                              return_loss=False, targets=None)
        
        out, _ = self.decoder(refer_video_args, refer_text_args,
                                                                return_loss=False, targets=None)
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)

        return output
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):

        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        if self.mask_sampling == 'gradient':
            return self.forward_gradient_mask_sampling(samples, valid_indices, text_queries, targets, text_auxiliary,
                                                       visualize=visualize, saved_path=saved_path)
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        if self.mask_sampling == 'clip':
            clip_video_feats, clip_text_feats, clip_video_attn, clip_text_attn = self.clip(video_inputs.clone(), copy.deepcopy(text_queries))
            video_mask_indices = self.sample_video_indice(attn=clip_video_attn)  # bool mask, True for masking
            text_mask_indices = self.sample_text_indice(attn=clip_text_attn)
        elif self.mask_sampling == 'random':
            video_mask_indices = None
            text_mask_indices = None
        else:
            raise ValueError()
        
        if self.need_masktext:
            mlm_text_encoder_output = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                     text_auxiliary=text_auxiliary,
                                     device=device,
                                     mask_sentence=True, masked_indices=text_mask_indices) # random sampling
                 
        if self.need_maskvideo:
            mvm_video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                     video_pad_mask=video_pad_masks.clone(),
                                     valid_indices=valid_indices.clone(),
                                     mask_video=True, masked_indices=video_mask_indices)  
                
        if self.need_text:
            text_encoder_output = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                     text_auxiliary=text_auxiliary,
                                     device=device,
                                     mask_sentence=False, masked_indices=None)
                          
        if self.need_video: 
            video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                     video_pad_mask=video_pad_masks.clone(),
                                     valid_indices=valid_indices.clone(),
                                     mask_video=False, masked_indices=None)     
        
        losses = {}
        if self.do_mlm:
            mlm_video_args = {
                'multiscales': [f.clone() for f in video_encoder_output['multiscales']],
                'multiscale_pad_masks': [m.clone() for m in video_encoder_output['multiscale_pad_masks']],
                'multiscale_poses': [p.clone() for p in video_encoder_output['multiscale_poses']],
                'multiscale_des': video_encoder_output['multiscale_des']
            }
            referent_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
            mlm_text_args = {
                'token_feats': mlm_text_encoder_output['token_feats'],
                'token_pad_masks': mlm_text_encoder_output['token_pad_masks'],
                'token_sentence_feats': mlm_text_encoder_output['token_sentence_feats']
            }
            mlm_text_gt = mlm_text_encoder_output['mlm_text_gt']
            mlm_fused_encoder_output = self.fusion_encoder(mlm_video_args, mlm_text_args,
                                                       referent_mask_condition=referent_masks, mlm=True)
            
            mlm_pred, mlm_loss_dict = self.text_encoder.forward_mlm(mlm_fused_encoder_output['fused_text_feats'], mlm_text_gt)
            losses.update(mlm_loss_dict)
            
        if self.do_mvm:
            mvm_text_args = {
                'token_feats': text_encoder_output['token_feats'].clone(),
                'token_pad_masks': text_encoder_output['token_pad_masks'].clone(),
                'token_sentence_feats': text_encoder_output['token_sentence_feats'].clone()
            }
            mvm_video_gt = mvm_video_encoder_output.pop('mvm_video_gt')
            mvm_video_args = mvm_video_encoder_output
            mvm_fused_encoder_output = self.fusion_encoder(mvm_video_args, mvm_text_args,
                                                        referent_mask_condition=referent_masks, mvm=True)
            
            mvm_pred, mvm_loss_dict = self.video_encoder.forward_mvm(mvm_fused_encoder_output['fused_video_feats'],
                                                                     mvm_video_gt)
            losses.update(mvm_loss_dict)
            
        if self.do_refer:
            refer_text_args = {
                'token_feats': text_encoder_output['token_feats'].clone(),
                'token_pad_masks': text_encoder_output['token_pad_masks'].clone(),
                'token_sentence_feats': text_encoder_output['token_sentence_feats'].clone()
            }
            refer_video_args = {
                'multiscales': [f.clone() for f in video_encoder_output['multiscales']],
                'multiscale_pad_masks': [m.clone() for m in video_encoder_output['multiscale_pad_masks']],
                'multiscale_poses': [p.clone() for p in video_encoder_output['multiscale_poses']],
                'multiscale_des': video_encoder_output['multiscale_des']
            }
    
            fused_encoder_output = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
               
            refer_video_args['multiscales'] =  fused_encoder_output['fused_video_feats']
            refer_text_args['token_feats'] = fused_encoder_output['fused_text_feats']
            refer_pred, refer_loss_dict = self.decoder(refer_video_args, refer_text_args,
                                                                   return_loss=True, targets=targets)
            losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices),
                              text_query=text_queries,
                              directory=saved_path, 
                              targets=targets, 
                              masked_text_tokenized=mlm_text_encoder_output['tokenized_result'],
                              masked_text_gt=mlm_text_encoder_output['mlm_text_gt'],
                              mlm_pred=mlm_pred, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

    def forward_gradient_mask_sampling(self, samples : NestedTensor, valid_indices : Tensor,
                                        text_queries, targets, text_auxiliary, visualize, saved_path):    
        video_inputs : Tensor = samples.tensors # t b c h w
        if self.do_mvm:
            video_inputs.requires_grad_(True)
        video_pad_masks = samples.mask
        device = video_inputs.device 
        losses = {}

        (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=text_auxiliary,
                                    device=device,
                                    mask_sentence=False, masked_indices=None)  
                
        (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
            = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)  

        refer_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())
        refer_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                            [p.clone() for p in video_poses], multiscale_dec)

        refer_fused_video_feats, query_feats, refer_fused_text_feats \
                                = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
        refer_video_args = (refer_fused_video_feats, [m.clone() for m in video_pad_masks], [p.clone() for p in video_poses], multiscale_dec)
        refer_text_args = (refer_fused_text_feats, token_pad_mask.clone(), text_sentence_features.clone())
        refer_pred, refer_loss_dict = self.fusion_encoder.forward_refer(refer_video_args, query_feats, refer_text_args,
                                                                return_loss=True, targets=targets)
        losses.update(refer_loss_dict)

        # just three {loss_cre, loss_mask, loss_dice,}, no {loss_ce0, loss_mask0}
        refer_loss: torch.Tensor = sum((refer_loss_dict[k] * self.weight_dict[k] for k in refer_loss_dict.keys()))
        if not math.isfinite(refer_loss.item()):
            print("Loss is {}, stopping training".format(refer_loss.item()))
            print(refer_loss_dict)
            sys.exit(1)
        refer_loss.backward(retain_graph=True)
        mlm_loss, mvm_loss = 0, 0
        if self.do_mlm:
            masked_text_tokenized, masked_text_args, masked_text_gt = \
                  self.text_encoder(texts=copy.deepcopy(text_queries), 
                                     text_auxiliary=text_auxiliary,
                                     device=device,
                                     mask_sentence=True, masked_indices=None,
                                     get_masking_info={'grad': tokenized_feats.grad.detach()}) # random sampling
            mlm_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                              [p.clone() for p in video_poses], multiscale_dec)
            # conditions:, 如果
            referent_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
            mlm_fused_text_feats = self.fusion_encoder(mlm_video_args, masked_text_args,
                                                       referent_mask_condition=referent_masks, mlm=True)
            mlm_pred, mlm_loss_dict = self.text_encoder.forward_mlm(mlm_fused_text_feats, masked_text_gt)
            mlm_loss = sum((mlm_loss_dict[k] * self.weight_dict[k] for k in mlm_loss_dict.keys()))
            losses.update(mlm_loss_dict)

        if self.do_mvm:
            masked_video_indices = self.sample_video_indice(grad=video_inputs.grad)
            masked_video_args, masked_video_gt \
                = self.video_encoder(videos=video_inputs.clone(),
                                     video_pad_mask=video_pad_masks.clone(),
                                     valid_indices=valid_indices.clone(),
                                     mask_video=True, masked_indices=masked_video_indices) 
            mvm_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())
    
            mvm_fused_video_feats = self.fusion_encoder(masked_video_args, mvm_text_args,
                                                        referent_mask_condition=referent_masks, mvm=True)
            
            mvm_pred, mvm_loss_dict = self.video_encoder.forward_mvm(mvm_fused_video_feats, masked_video_gt)
            mvm_loss = sum((mvm_loss_dict[k] * self.weight_dict[k] for k in mvm_loss_dict.keys()))
            losses.update(mvm_loss_dict)


        masking_loss: torch.Tensor = mlm_loss + mvm_loss
        masking_loss.backward()

        assert set(losses.keys()).issubset(self.weight_dict.keys())
        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries, directory=saved_path, 
                                targets=targets, 
                                masked_text_tokenized=masked_text_tokenized, masked_text_gt=masked_text_gt,
                                mlm_pred=mlm_pred, 
                                refer_pred=refer_pred,
                                losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def unimodal_encoder(device, model_configs):
    
    configs = vars(model_configs)
        
    model = UnimodalEncoder_Model(
        do_refer=configs['do_refer'],
        do_mvm=configs['do_mvm'],
        do_mlm=configs['do_mlm'],
        weight_dict=configs['weight_dict'],

        mode_name=configs['mode_name'],
        mask_sampling=configs['mask_sampling'],
        clip_configs=model_configs.clip_configs,
        d_model=configs['d_model'],

        video_encoder_configs=model_configs.video_encoder,
        text_encoder_configs=model_configs.text_encoder,
        fusion_encoder_configs=model_configs.fusion_encoder,
        decoder_configs=model_configs.decoder
    )
    model.to(device)
    
    optmization_configs = model_configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer  

      
# text已经被parse成了一个AMR graph
# text encoder是一堆graph-self-attention, 经过text encoder后, 舍弃graph的结构, 只要每个node的feature作为text feature
# 不做mlm/mvm
class Unimodal_Video_GraphEncoder(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 video_encoder_configs,
                 text_encoder_configs,
                 fusion_encoder_configs,
                 decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.d_model = d_model
        
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        # a gnn
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        self.fusion_encoder = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)
        
        # 定义decoder
        create_decoder = refer_decoder_entrypoints(decoder_configs.name)
        # 跑之前的模型的时候把configs中的nqueries, query_feat放到decoder中
        self.decoder = create_decoder(decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device) 
        if 'graphs' in graph_encoder_output:
            graphs = graph_encoder_output['graphs'] # list[Graph]
            nodes_feats = [] #list[ni c]
            for graph in graphs:
                nodes_feats.append(graph.x)
            nodes_feats, nodes_pad = pad_1ds_with_grad(nodes_feats) # b node_max c, b node_max
            
            refer_text_args = {
                'token_feats': nodes_feats,
                'token_pad_masks': nodes_pad,
                'token_sentence_feats': None
            }  
        elif 'linearized_graphs' in graph_encoder_output:
            refer_text_args = graph_encoder_output['linearized_graphs']
        else:
            raise ValueError()
        
        video_encoder_output = self.video_encoder(videos=samples.tensors,
                                                    video_pad_mask=samples.mask,
                                                    valid_indices=valid_indices,
                                                    mask_video=False, masked_indices=None)       
        
       
        refer_video_args = {
            'multiscales': video_encoder_output['multiscales'],
            'multiscale_pad_masks': video_encoder_output['multiscale_pad_masks'],
            'multiscale_poses': video_encoder_output['multiscale_poses'],
            'multiscale_des': video_encoder_output['multiscale_des']
        }        
        
        fused_encoder_output = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
                                
        refer_video_args['multiscales'] =  fused_encoder_output['fused_video_feats']
        refer_text_args['token_feats'] = fused_encoder_output['fused_text_feats']
        
        
        out, _ = self.decoder(refer_video_args,refer_text_args,
                              return_loss=False, targets=None)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
        
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        losses = {}
        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])

        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        # transform penman graph to geometric graph
       
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device) 
        if 'graphs' in graph_encoder_output:
            graphs = graph_encoder_output['graphs'] # list[Graph]
            nodes_feats = [] #list[ni c]
            for graph in graphs:
                nodes_feats.append(graph.x)
            nodes_feats, nodes_pad = pad_1ds_with_grad(nodes_feats) # b node_max c, b node_max
            
            refer_text_args = {
                'token_feats': nodes_feats,
                'token_pad_masks': nodes_pad,
                'token_sentence_feats': None
            }  
        elif 'linearized_graphs' in graph_encoder_output:
            refer_text_args = graph_encoder_output['linearized_graphs']
        else:
            raise ValueError()
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)       
       
        refer_video_args = {
            'multiscales': video_encoder_output['multiscales'],
            'multiscale_pad_masks': video_encoder_output['multiscale_pad_masks'],
            'multiscale_poses': video_encoder_output['multiscale_poses'],
            'multiscale_des': video_encoder_output['multiscale_des']
        }
        
        
        fused_encoder_output = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
                                
        refer_video_args['multiscales'] =  fused_encoder_output['fused_video_feats']
        refer_text_args['token_feats'] = fused_encoder_output['fused_text_feats']
        
        refer_pred, refer_loss_dict = self.decoder(refer_video_args, refer_text_args,
                                                                return_loss=True, targets=targets)
        # just three {loss_cre, loss_mask, loss_dice,}, no {loss_ce0, loss_mask0}
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def unimodal_video_graphencoder(device, model_configs):
    configs = model_configs
    model =  Unimodal_Video_GraphEncoder(
        d_model=configs.d_model,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        decoder_configs=model_configs.decoder
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


# text 只有在encode中是linearized
# 在decoder, fusion encoder中都是graph的结构

class Unimodal_Video_GraphDecoder(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 video_encoder_configs,
                 text_encoder_configs,
                 fusion_encoder_configs,
                 decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.d_model = d_model
        
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        # a gnn
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        self.fusion_encoder = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)
        
        # 定义decoder
        create_decoder = refer_decoder_entrypoints(decoder_configs.name)
        # 跑之前的模型的时候把configs中的nqueries, query_feat放到decoder中
        self.decoder = create_decoder(decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device) 

        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)               
        
        fused_encoder_output = self.fusion_encoder(video_encoder_output, graph_encoder_output, refer=True)
                                
        if fused_encoder_output is None:
            # no fusion
            pass
        elif (len(list(fused_encoder_output.keys())) == 1) and (list(fused_encoder_output.keys())[0] == 'fused_video_feats'):
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
        else:
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
            graph_encoder_output['graphs'] = fused_encoder_output['fused_graph_feats']
        
        
        out, _ = self.decoder(video_encoder_output, graph_encoder_output,
                              return_loss=False, targets=None)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
        
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        losses = {}
        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])

        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        # transform penman graph to geometric graph
       
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device) 

        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)              
        
        fused_encoder_output = self.fusion_encoder(video_encoder_output, graph_encoder_output, refer=True)
        if fused_encoder_output is None:
            # no fusion
            pass
        elif (len(list(fused_encoder_output.keys())) == 1) and (list(fused_encoder_output.keys())[0] == 'fused_video_feats'):
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
        else:
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
            graph_encoder_output['graphs'] = fused_encoder_output['fused_graph_feats']
        
        refer_pred, refer_loss_dict = self.decoder(video_encoder_output, graph_encoder_output,
                                                   return_loss=True, targets=targets)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def unimodal_video_graphDecoder(device, model_configs):
    configs = model_configs
    model =  Unimodal_Video_GraphDecoder(
        d_model=configs.d_model,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        decoder_configs=model_configs.decoder
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 



class TwoDecoder(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 video_encoder_configs,
                 object_decoder_configs,
                 
                 text_encoder_configs,
                 fusion_encoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_mask: 5.
        # loss_dice: 5.
        # loss_ce: 2,
        # loss_refer_mask: 5.
        # loss_refer_dice: 5.
        # loss_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # 抽取text的特征
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        # feature map和text进行融合
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        self.fusion_encoder = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)

        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
    
    
       
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        
        object_decoder_output, _ = self.object_decoder(video_encoder_output, 
                                                                      return_loss=False,
                                                                      targets=None)
        
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)             
        fused_encoder_output = self.fusion_encoder(video_encoder_output, graph_encoder_output, refer=True)
                                
        if fused_encoder_output is None:
            # no fusion
            pass
        elif (len(list(fused_encoder_output.keys())) == 1) and (list(fused_encoder_output.keys())[0] == 'fused_video_feats'):
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
        else:
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
            graph_encoder_output['graphs'] = fused_encoder_output['fused_graph_feats']
        
        
        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
        
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        losses = {}
        

        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)  
        # 找出video中的所有物体
        # 不能改变video encoder output
        # object_loss_dict中可能与text相关的就是token classification
        # 一个类别可以对应多个text word, 比如'person' -> 'man', 'woman', 'human', 就是多类分类
        # 用text那边的知识: 'woman'和'man'都是人 来指导视觉的训练
        # object_decoder_output有
        # 1. object embedding
        # 2. matching结果

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], class_token_id_map=None)
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device) 
        # video_encoder_output和text进行融合
        fused_encoder_output = self.fusion_encoder(video_encoder_output, graph_encoder_output, refer=True)
        if fused_encoder_output is None:
            # no fusion
            pass
        elif (len(list(fused_encoder_output.keys())) == 1) and (list(fused_encoder_output.keys())[0] == 'fused_video_feats'):
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
        else:
            video_encoder_output['multiscales'] =  fused_encoder_output['fused_video_feats']
            graph_encoder_output['graphs'] = fused_encoder_output['fused_graph_feats']
            
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder(
        d_model=configs.d_model,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


class TwoDecoder_v2(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
           
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)

        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        scale_encoder_configs = configs.scale_encoder
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)    
          
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        # 10 c
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        # b V+E_max
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v2(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v2(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


class TwoDecoder_v2_forsequence(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs,
                 text_is_graph=False) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        self.text_is_graph = text_is_graph
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)
        if self.text_is_graph:
            graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                        text=text_queries,
                                                        device=device)
        else:
            graph_encoder_output = self.text_graph_encoder(texts=text_queries, 
                                                           text_auxiliary=None,
                                                           device=device)
        
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        crossed_text_feats, crossed_text_pad_mask = graph_encoder_output['token_feats'].clone(),\
            graph_encoder_output['token_pad_masks'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
           
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)

        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        scale_encoder_configs = configs.scale_encoder
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)    
          
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        if self.text_is_graph:
            graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                        text=text_queries,
                                                        device=device)
        else:
            graph_encoder_output = self.text_graph_encoder(texts=text_queries, 
                                                           text_auxiliary=None,
                                                           device=device)
        # 10 c
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        # b max c
        crossed_text_feats, crossed_text_pad_mask = graph_encoder_output['token_feats'].clone(),\
            graph_encoder_output['token_pad_masks'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v2_forsequence(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v2_forsequence(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
        text_is_graph=configs.text_is_graph if hasattr(configs, 'text_is_graph') else True
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


# 更改下object embeds
class TwoDecoder_v3(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        assert len(object_classes) == 8, '现在使用了a2ds, 你要加上background这个词语'
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
        from .layers_unimodal_attention import FeatureResizer
        self.concate_word_linear = FeatureResizer(input_feat_size=2*d_model,
                                                  output_feat_size=d_model,
                                                  dropout=0.1,
                                                  do_ln=True)
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
           
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)

        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        scale_encoder_configs = configs.scale_encoder
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)    
    
    def extend_objects_embeds(self, outputs):
        # v4: 什么都不加
        # v4_ObjectAddToken: 加上words embeds
        # v4_ObjectAddTokenBox: 再加上box diff
        # v4_object: ..
        object_embeds = outputs['object_embeds'].clone() # b nq c
        batch_size, num_queries, _, device = *object_embeds.shape, object_embeds.device
        pred_logits = outputs['pred_logits'].clone() # b nq k+1
        pred_classes = torch.argmax(pred_logits,dim=-1).flatten().tolist() # b*nq [0, K]
        # b*nq
        pred_classes_in_words = torch.tensor([self.class_token_id_map[pred_cls] for pred_cls in pred_classes],
                                             device=object_embeds.device).long()
        pred_classes_word_embeds = self.text_graph_encoder.get_words_embeds(pred_classes_in_words, device=device)
        pred_classes_word_embeds = rearrange(pred_classes_word_embeds, '(b nq) c -> b nq c',b=batch_size, nq=num_queries)
        
        object_embeds = torch.cat([object_embeds, pred_classes_word_embeds], dim=-1) # b nq 2c
        object_embeds = self.concate_word_linear(object_embeds) # b nq c
        return object_embeds
        
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        # 10 c
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        # b V+E_max
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        new_object_embes = self.extend_objects_embeds(object_decoder_output)
        object_decoder_output['object_embeds'] = new_object_embes
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v3(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v3(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 

# text那边加上几层layer, 在外面
class TwoDecoder_v4(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 graph_layer_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        from .layers_graph import graph_layer_entrypoints
        from .transformer import _get_clones
        create_graph_layer  = graph_layer_entrypoints(graph_layer_configs.name)
        graph_layer = create_graph_layer(graph_layer_configs, d_model)
        self.text_graph_layers = _get_clones(graph_layer, graph_layer_configs.nlayers)
        
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        transformed_all_feats = self.text_self_graph_encoder(graph_encoder_output['all_feats'].clone(), 
                                                             graph_encoder_output['all_seg_ids'].clone(), 
                                                             copy.deepcopy(graph_encoder_output['graph']))
        graph_encoder_output['all_feats'] = transformed_all_feats
        
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
           
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)

        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        scale_encoder_configs = configs.scale_encoder
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)    
    
    def text_self_graph_encoder(self, all_feats, all_seg_ids, graphs):
        device = all_feats.device
        edge_index = graphs.edge_index.to(device)
        num_nodes = [(seg_ids>0).int().sum().item() for seg_ids in all_seg_ids]
        num_edges = [(seg_ids<0).int().sum().item() for seg_ids in all_seg_ids]
        nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(all_feats, all_seg_ids)], dim=0)
        edge_feats = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(all_feats, all_seg_ids)], dim=0)
        
        for graph_layer in self.text_graph_layers:
            nodes_feats, edge_feats = graph_layer(nodes_feats, edge_index, edge_feats)
            
        assert sum(num_nodes) == len(nodes_feats)
        assert sum(num_edges) == len(edge_feats)
        batch_node_feats = torch.split(nodes_feats, num_nodes)
        batch_edge_feats = torch.split(edge_feats, num_edges)
        for batch_idx, seg_ids in enumerate(all_seg_ids):
            all_feats[batch_idx, seg_ids > 0] = batch_node_feats[batch_idx]
            all_feats[batch_idx, seg_ids < 0] = batch_edge_feats[batch_idx] 
               
        return all_feats    
        
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        
        transformed_all_feats = self.text_self_graph_encoder(graph_encoder_output['all_feats'].clone(), 
                                                             graph_encoder_output['all_seg_ids'].clone(), 
                                                             copy.deepcopy(graph_encoder_output['graph']))
        graph_encoder_output['all_feats'] = transformed_all_feats
        
        # 10 c
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        # b V+E_max
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v4(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v4(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        graph_layer_configs=configs.graph_layer,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


# 用linearized AMR, 就成了一个sequence序列了
class TwoDecoder_v5(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
           
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)

        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        scale_encoder_configs = configs.scale_encoder
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)    
          
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
        # 10 c
        object_class_embeds = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
        
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        
        # b V+E_max
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
        crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
        crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0], b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales

        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)
        
        # 进行referent 推断
        # referent应该和objects的matching结果一致, 
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                                   return_loss=True, 
                                                   targets=referent_targets, 
                                                   matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v5(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v5(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 


# objects encoder, relation encoder
class TwoDecoder_v6(nn.Module):
    def __init__(self, 
                 d_model,
                 weight_dict,
                 object_classes,
                 video_encoder_configs,
                 text_encoder_configs,
                 
                 fusion_encoder_configs,
                 object_decoder_configs,
                 referent_decoder_configs) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        # loss_object_mask: 5.
        # loss_object_dice: 5.
        # loss_object_ce: 2,
        # loss_object_token_classification: 目标检测的每个object作为一个text token计算它的真实文字
        # loss_mask: 5.
        # loss_dice: 5.
        self.d_model = d_model
        
        # 抽特征 + multiscale_fusion
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)

        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_graph_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        self.object_classes = object_classes
        self.class_token_id_map = {idx:tok_id for idx, tok_id in enumerate(self.object_classes)}
        self.build_fusion_encoder(fusion_encoder_configs, d_model)
                       
        # 把feature map中的object信息decode出来
        create_object_decoder = refer_decoder_entrypoints(object_decoder_configs.name)
        self.object_decoder = create_object_decoder(object_decoder_configs, d_model=d_model) 
              
        # cross attention负责把[objects, feature map]中的action, roleset信息decode出来
        # self attention负责利用text 的结构把referent推断出来
        create_referent_decoder = refer_decoder_entrypoints(referent_decoder_configs.name)
        self.referent_decoder = create_referent_decoder(referent_decoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        batch_size = len(text_queries)
        nf, bs, *_ = samples.tensors.shape
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)
                
        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        batch_size, nf, *_ = multiscales[0].shape
        
        # 和object classes进行融合
        crossed_feats = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        crossed_feats = repeat(crossed_feats, 'k c -> k b c', b=batch_size)
        crossed_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_feats.shape[0], b=batch_size)
        srcs = self.fusion(multiscales, multiscales_pad_masks, multiscales_poses,
                           crossed_feats=crossed_feats, crossed_pad_mask=None, crossed_pos=crossed_pos,)
        # 把所有objects parse出来
        multiscales = self.object_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
        # 把objects decode出来
        object_decoder_output, _ = self.object_decoder(video_encoder_output, return_loss=False, targets=None)
        object_embeds = object_decoder_output['object_embeds'].clone() # b n c
        object_embeds = repeat(object_embeds, 'b (h w) c -> b t c h w', h=5, w=20, t=1)
        
        multiscales.append(object_embeds)
        multiscales_pad_masks.append(torch.zeros([batch_size, nf, 5, 20], device=object_embeds.device).bool())
        multiscales_poses.append(repeat(self.object_pos.weight, '1 c -> b t c h w', b=batch_size, t=nf, h=5, w=20))
        descs.append([1, 666])
        
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_feats, crossed_pad_mask = pad_1d_feats(crossed_feats)
        crossed_feats = crossed_feats.permute(1, 0, 2) # max b c
        crossed_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_feats.shape[0], b=batch_size)
        srcs = self.fusion(multiscales, multiscales_pad_masks, multiscales_poses,
                           crossed_feats=crossed_feats,
                           crossed_pad_mask=crossed_pad_mask,
                           crossed_pos=crossed_pos,)
        
        multiscales = self.relation_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales[:-1]
        
        out, _ = self.referent_decoder(video_encoder_output, object_decoder_output, graph_encoder_output,
                                        return_loss=False)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        if len(out['pred_masks'].shape) == 5:
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
            output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        else:
            assert len(out['pred_masks'].shape) == 4
            output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
            nf, batch_size, *_ = output['pred_masks'].shape
            pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
            pred_is_referred[..., 1] = -100
            output['pred_is_referred'] = pred_is_referred

        return output
    
    def build_fusion_encoder(self, configs, d_model):
        from .encoder_fusion import VisionLanguageFusionModule
        from .encoder_multiscale import multiscale_encoder_entrypoints
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        object_encoder_configs = configs.object_encoder
        create_object_encoder = multiscale_encoder_entrypoints(object_encoder_configs.name)
        self.object_encoder = create_object_encoder(object_encoder_configs, d_model=d_model)
          
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)  
        
        relation_encoder_configs = configs.relation_encoder
        create_relation_encoder = multiscale_encoder_entrypoints(relation_encoder_configs.name)
        self.relation_encoder = create_relation_encoder(relation_encoder_configs, d_model=d_model)
        
        self.object_pos = nn.Embedding(1, d_model)
          
    
    def fusion(self, 
               multiscales, multiscales_pad_masks, multiscales_poses,
               crossed_feats, crossed_pad_mask, crossed_pos):
        batch_size = multiscales[0].shape[0]
        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            thw = feat.shape[0]
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_feats,
                                    memory_key_padding_mask=crossed_pad_mask,
                                    pos=crossed_pos,
                                    query_pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size))
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat) 
        return srcs   
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):
        """
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        losses = {}
        batch_size = len(text_queries)
        # 对 all_feats进行
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        video_encoder_output = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
        # 可能有text embedding ; graph embedding
        graph_encoder_output = self.text_graph_encoder(graph=text_auxiliary,
                                                       text=text_queries,
                                                       device=device)

        all_feats, all_seg_ids = graph_encoder_output['all_feats'].clone(), graph_encoder_output['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_encoder_output['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_encoder_output['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_encoder_output['multiscale_poses']]
        descs = copy.deepcopy(video_encoder_output['multiscale_des'])
        batch_size, nf, *_ = multiscales[0].shape
        
        # 和object classes进行融合
        crossed_feats = self.text_graph_encoder.get_words_embeds(self.object_classes, device=device)  
        crossed_feats = repeat(crossed_feats, 'k c -> k b c', b=batch_size)
        crossed_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_feats.shape[0], b=batch_size)
        srcs = self.fusion(multiscales, multiscales_pad_masks, multiscales_poses,
                           crossed_feats=crossed_feats, crossed_pad_mask=None, crossed_pos=crossed_pos,)
        # 把所有objects parse出来
        multiscales = self.object_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales 
        # 把objects decode出来
        all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
                                                            class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_classes)})
        object_decoder_output, object_loss_dict = self.object_decoder(video_encoder_output, 
                                                                      return_loss=True,
                                                                      targets=all_instance_targets)
        object_embeds = object_decoder_output['object_embeds'].clone() # b n c
        # TODO: change me
        object_embeds = repeat(object_embeds, 'b (h w) c -> b t c h w', h=5, w=20, t=1)
        matching_results = object_loss_dict.pop('matching_results')
        losses.update(object_loss_dict)               
        
        # 把objects当成parsing memory
        multiscales.append(object_embeds)
        multiscales_pad_masks.append(torch.zeros([batch_size, nf, 5, 20], device=object_embeds.device).bool())
        multiscales_poses.append(repeat(self.object_pos.weight, '1 c -> b t c h w', b=batch_size, t=nf, h=5, w=20))
        descs.append([1, 666])
        
        # 把multiscale和text进行fusion
        who_does_cross_attention_mask = torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        crossed_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        def pad_1d_feats(feat_list):
            # list[ni c] -> b nmax c
            feat_len = [len(feat) for feat in feat_list]
            n_max = max(feat_len) 
            batch_size = len(feat_list)
            pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
            for i in range(batch_size):
                feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
                pad_mask[i, :feat_len[i]] = False
            feat_list = torch.stack(feat_list, dim=0) # b nmax c
            return feat_list, pad_mask
        # b max c
        crossed_feats, crossed_pad_mask = pad_1d_feats(crossed_feats)
        crossed_feats = crossed_feats.permute(1, 0, 2) # max b c
        crossed_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_feats.shape[0], b=batch_size)
        srcs = self.fusion(multiscales, multiscales_pad_masks, multiscales_poses,
                           crossed_feats=crossed_feats,
                           crossed_pad_mask=crossed_pad_mask,
                           crossed_pos=crossed_pos,)
        # 把所有relations parse出来
        multiscales = self.relation_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        video_encoder_output['multiscales'] = multiscales[:-1]
        referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        refer_pred, refer_loss_dict = self.referent_decoder(video_encoder_output, 
                                                            object_decoder_output, graph_encoder_output,
                                                            return_loss=True, 
                                                            targets=referent_targets, 
                                                            matching_results=matching_results)
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def two_decoders_v6(device, model_configs):
    
    configs = model_configs
    model =  TwoDecoder_v6(
        d_model=configs.d_model,
        object_classes=configs.object_classes,
        weight_dict=vars(configs.weight_dict),
        video_encoder_configs=configs.video_encoder,
        object_decoder_configs=configs.object_decoder,
        text_encoder_configs=configs.text_encoder,
        fusion_encoder_configs=configs.fusion_encoder,
        referent_decoder_configs=configs.referent_decoder,
    )
    model.to(device)

    optmization_configs = configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 












































"""
1. linearized graph经过tokenize后, 只获得vocabulary embedding
2. video encoder之后直接接上scale encoder
3. 没有fusion encoder, 没有early fusion
4. decoder的self attention 是dim=1024, amr encoder中的self attention layers
   decoder的cross attention是dim=256的重新开始学习的cross attention layers
   decoder只使用32x 一个scale
   decoder没有matching
"""
class Scratch1(nn.Module):
 
    def __init__(self, 
                 visual_dim, # 256
                 text_dim, # 1024
                 
                 weight_dict,
                 video_encoder_configs,
                 text_encoder_configs,
                 decoder_configs) -> None:
        super().__init__()
        
        self.weight_dict = weight_dict
        self.visual_dim = visual_dim
        
        # video encoder之后加上multiscale encoder
        # 4x, 8x, 16x, 32x
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=visual_dim)
        
        # tokenizer
        from transformers import BartForConditionalGeneration
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.amr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
        
        # word embedding
        amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
        amrencoder = amr2text_model.get_encoder()
        self.text_dim = amr2text_model.config.d_model
        amrencoder.layers = nn.ModuleList([])
        amrencoder.layernorm_embedding = None
        self.amr_vocabulary_encoder = amrencoder 
        if text_encoder_configs.freeze_vocabulary:
            for p in self.amr_vocabulary_encoder.parameters():
                p.requires_grad_(False) 

        # decoder
        # cross attention的时候要区分谁是video, 谁是text
        self.video_pos = nn.Embedding(1, self.visual_dim)
        self.text_pos = nn.Embedding(1, self.visual_dim)
        
        self.decoder_used_scale = decoder_configs.used_scale # [1, 32]
        self.decoder_self_attention_layers = nn.ModuleList()  # self里有linear
        self.decoder_cross_attention_layers = nn.ModuleList()
        from .layers_unimodal_attention import CrossAttentionLayer
        amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2').get_encoder()
        self.decoder_num_layers = len(amr2text_model.layers)
        for i in range(self.decoder_num_layers):
            self.decoder_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=self.visual_dim,
                    nhead=decoder_configs.nheads,
                    dropout=0.0,
                    normalize_before=decoder_configs.pre_norm,
                )
            )
            self.decoder_self_attention_layers.append(
                copy.deepcopy(amr2text_model.layers[i])
            ) 
        # projection
        from .layers_unimodal_attention import FeatureResizer, MLP
        from .criterion_video import matching_entrypoints
        proj_configs = text_encoder_configs.proj
        self.text_proj_to_vision = FeatureResizer(
                input_feat_size=self.text_dim,
                output_feat_size=self.visual_dim,
                dropout=proj_configs.dropout,
                do_ln=proj_configs.do_ln
        ) # 1024 -> 256
        self.vision_proj_to_text = FeatureResizer(
                input_feat_size=self.visual_dim,
                output_feat_size=self.text_dim,
                dropout=proj_configs.dropout,
                do_ln=proj_configs.do_ln
        ) # 256 -> 1024
        self.decoder_norm = nn.LayerNorm(self.visual_dim)
        self.mask_embed = MLP(self.visual_dim,
                              self.visual_dim,
                              self.visual_dim, 3)
        
        matching_configs = decoder_configs.matching_configs
        create_criterion = matching_entrypoints(decoder_configs.name)
        self.criterion = create_criterion(matching_configs)
        # 只有node/ 只有node和edge /整个linearized graph
        self.graph_which_to_cross = decoder_configs.graph_which_to_cross
               

def scratch1(configs, d_model):
    pass
 
# 做mvm的时候, 先提取text特征, 然后video encoder中间插入cross attention
# 做mlm的时候, 先提取video特征, 然后text encoder中间插入这些video,
class UnimodalEncoder_EncoderInterleaveCrossAttention(Video_Model):
    def __init__(self, do_refer, do_mvm, do_mlm, weight_dict, mode_name, mask_sampling, clip_configs, d_model,
                 
                video_encoder_configs,
                text_encoder_configs,
                fusion_encoder_configs,) -> None:
        
        super().__init__(do_refer, do_mvm, do_mlm, weight_dict, mode_name, mask_sampling, clip_configs, d_model)

        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        video_encoder_with_mvm_head = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        text_encoder_with_mlm_head = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        fusion_encoder_with_refer_decoder = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)

        self.video_encoder = video_encoder_with_mvm_head
        self.text_encoder = text_encoder_with_mlm_head

        self.text_encoder.mask_sampling = self.mask_sampling
        self.video_encoder.mask_sampling = self.mask_sampling

        self.fusion_encoder = fusion_encoder_with_refer_decoder
        
        # 由于两个encoder是分开的, 所以有些任务可以使用其他任务forward的特征
        self.need_maskvideo = False
        self.need_masktext= False
        self.need_video=False
        self.need_text=False

        if self.do_mvm:
            self.need_maskvideo = True
            self.need_text = True
        if self.do_mlm:
            self.need_masktext = True
            self.need_video = True
        if self.do_refer:
            self.need_video = True
            self.need_text = True
            
        assert self.need_maskvideo or self.need_video or self.need_text or self.need_masktext 

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=None,
                                    device=device,
                                    mask_sentence=False, masked_indices=None)  
                
        (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
            = self.video_encoder(videos=samples.tensors,
                                    video_pad_mask=samples.mask,
                                    valid_indices=valid_indices,
                                    mask_video=False, masked_indices=None) 
        
    
        refer_text_args = (token_feats, token_pad_mask, text_sentence_features)
        refer_video_args = (video_feats, video_pad_masks, video_poses , multiscale_dec)

        refer_fused_video_feats, query_feats, refer_fused_text_feats \
                                = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
        
        refer_fused_video_feats = self.video_encoder.proj_multiscales(refer_fused_video_feats) # to the same dimension
        refer_fused_video_feats = self.video_encoder.fuse_multiscales((refer_fused_video_feats, 
                                                                       video_pad_masks, video_poses , multiscale_dec))
        refer_video_args = (refer_fused_video_feats, video_pad_masks, video_poses , multiscale_dec)
        refer_text_args = (refer_fused_text_feats, token_pad_mask, text_sentence_features)
        out, _ = self.fusion_encoder.forward_refer(refer_video_args, query_feats, refer_text_args,
                                                                return_loss=False, targets=None)
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)

        return output
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):

        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        if self.mask_sampling == 'gradient':
            return self.forward_gradient_mask_sampling(samples, valid_indices, text_queries, targets, text_auxiliary,
                                                       visualize=visualize, saved_path=saved_path)
        
        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
        if self.mask_sampling == 'clip':
            clip_video_feats, clip_text_feats, clip_video_attn, clip_text_attn = self.clip(video_inputs.clone(), copy.deepcopy(text_queries))
            video_mask_indices = self.sample_video_indice(attn=clip_video_attn)  # bool mask, True for masking
            text_mask_indices = self.sample_text_indice(attn=clip_text_attn)
        elif self.mask_sampling == 'random':
            video_mask_indices = None
            text_mask_indices = None
        else:
            raise ValueError()
        
        losses = {}
        if self.do_refer:
            (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                 = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                     text_auxiliary=text_auxiliary,
                                     device=device,
                                     mask_sentence=False, masked_indices=None)  
            (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
                = self.video_encoder(videos=video_inputs.clone(),
                                     video_pad_mask=video_pad_masks.clone(),
                                     valid_indices=valid_indices.clone(),
                                     mask_video=False, masked_indices=None) 
             
            refer_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())
            refer_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                                [p.clone() for p in video_poses], multiscale_dec)
    
            refer_fused_video_feats, query_feats, refer_fused_text_feats \
                                    = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
            
            refer_fused_video_feats = self.video_encoder.proj_multiscales(refer_fused_video_feats) # to the same dimension
            refer_video_args = (refer_fused_video_feats, [m.clone() for m in video_pad_masks], [p.clone() for p in video_poses], multiscale_dec)
            
            refer_fused_video_feats = self.video_encoder.fuse_multiscales(refer_video_args)
            
            refer_video_args = (refer_fused_video_feats, [m.clone() for m in video_pad_masks], [p.clone() for p in video_poses], multiscale_dec)
            refer_text_args = (refer_fused_text_feats, token_pad_mask.clone(), text_sentence_features.clone())
            refer_pred, refer_loss_dict = self.fusion_encoder.forward_refer(refer_video_args, query_feats, refer_text_args,
                                                                   return_loss=True, targets=targets)
            # just three {loss_cre, loss_mask, loss_dice,}, no {loss_ce0, loss_mask0}
            losses.update(refer_loss_dict) 
        if self.do_mlm:
            if self.do_refer:
                pass
            else:
                (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
                    = self.video_encoder(videos=video_inputs.clone(),
                                        video_pad_mask=video_pad_masks.clone(),
                                        valid_indices=valid_indices.clone(),
                                        mask_video=False, masked_indices=None)  
            mlm_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                            [p.clone() for p in video_poses], multiscale_dec)
            # conditions:, 如果
            referent_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
            (masked_text_tokenized, masked_text_gt), mlm_pred, mlm_loss_dict = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=text_auxiliary,
                                    device=device,
                                    mlm=True, masked_indices=text_mask_indices,
                                    video_args=mlm_video_args, referent_mask_condition=referent_masks)  
            losses.update(mlm_loss_dict)                                
        if self.do_mvm:
            if self.do_refer:
                pass
            else:
                (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                    = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                        text_auxiliary=text_auxiliary,
                                        device=device,
                                        mask_sentence=False, masked_indices=None) 
            mvm_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())

            (masked_video,), mvm_pred, mvm_loss_dict\
                = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mvm=True, masked_indices=video_mask_indices, mvm_text_args=mvm_text_args) 
            
            losses.update(mvm_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())
        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, masked_text_tokenized=masked_text_tokenized, masked_text_gt=masked_text_gt,
                            mlm_pred=mlm_pred, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

    def forward_gradient_mask_sampling(self, samples : NestedTensor, valid_indices : Tensor,
                                        text_queries, targets, text_auxiliary, visualize, saved_path):    
        video_inputs : Tensor = samples.tensors # t b c h w
        if self.do_mlm:
            video_inputs.requires_grad_(True)
        video_pad_masks = samples.mask
        device = video_inputs.device 

        losses = {}
        # refer
        (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=text_auxiliary,
                                    device=device, tokenized_feats_requires_grad=True, 
                                    # 和视频不同, text encoder的tokenized是在内部实现的, 不像samples直接是tensor
                                    mask_sentence=False, masked_indices=None)  
        (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
            = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None) 
            
        refer_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())
        refer_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                            [p.clone() for p in video_poses], multiscale_dec)

        refer_fused_video_feats, query_feats, refer_fused_text_feats \
                                = self.fusion_encoder(refer_video_args, refer_text_args, refer=True)
        refer_video_args = (refer_fused_video_feats, [m.clone() for m in video_pad_masks], [p.clone() for p in video_poses], multiscale_dec)
        refer_text_args = (refer_fused_text_feats, token_pad_mask.clone(), text_sentence_features.clone())
        refer_pred, refer_loss_dict = self.fusion_encoder.forward_refer(refer_video_args, query_feats, refer_text_args,
                                                                return_loss=True, targets=targets)
        
        refer_loss: torch.Tensor = sum((refer_loss_dict[k] * self.weight_dict[k] for k in refer_loss_dict.keys()))
        if not math.isfinite(refer_loss.item()):
            print("Loss is {}, stopping training".format(refer_loss.item()))
            print(refer_loss_dict)
            sys.exit(1)
        refer_loss.backward(retain_graph=True)

        if self.do_mlm:
            mlm_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                            [p.clone() for p in video_poses], multiscale_dec)
            # conditions:, 如果
            referent_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
            (masked_text_tokenized, masked_text_gt), mlm_pred, mlm_loss_dict = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=text_auxiliary,
                                    device=device,
                                    mlm=True, masked_indices=None,
                                    get_masking_info={'grad': tokenized_feats.grad.detach()},
                                    video_args=mlm_video_args, referent_mask_condition=referent_masks) 
            mlm_loss = sum((mlm_loss_dict[k] * self.weight_dict[k] for k in mlm_loss_dict.keys())) 
            losses.update(mlm_loss_dict)  

        if self.do_mvm:
            mvm_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())

            (masked_video,), mvm_pred, mvm_loss_dict\
                = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mvm=True, masked_indices=None,
                                    get_masking_info={'grad':video_inputs.grad.detach()},
                                      mvm_text_args=mvm_text_args) 
            mvm_loss = sum((mvm_loss_dict[k] * self.weight_dict[k] for k in mvm_loss_dict.keys()))
            losses.update(mvm_loss_dict)
        
        masking_loss: torch.Tensor = mlm_loss + mvm_loss
        masking_loss.backward()

        assert set(losses.keys()).issubset(self.weight_dict.keys())
        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries, directory=saved_path, 
                                targets=targets, 
                                masked_text_tokenized=masked_text_tokenized, masked_text_gt=masked_text_gt,
                                mlm_pred=mlm_pred, 
                                refer_pred=refer_pred,
                                losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def unimodal_encoder_encoderinterleavecross(device, model_configs):
    configs = vars(model_configs)
        
    model = UnimodalEncoder_EncoderInterleaveCrossAttention(
        do_refer=configs['do_refer'],
        do_mvm=configs['do_mvm'],
        do_mlm=configs['do_mlm'],
        weight_dict=configs['weight_dict'],

        mode_name=configs['mode_name'],
        mask_sampling=configs['mask_sampling'],
        clip_configs=model_configs.clip_configs,
        d_model=configs['d_model'],

        video_encoder_configs=model_configs.video_encoder,
        text_encoder_configs=model_configs.text_encoder,
        fusion_encoder_configs=model_configs.fusion_encoder
    )
    model.to(device)
    
    optmization_configs = model_configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer  


class UnimodalEncoder_NMNs(nn.Module):
    def __init__(self, weight_dict, d_model,
                 video_encoder_configs,
                 text_encoder_configs,
                 fusion_encoder_configs,
                 
                 ) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.d_model = d_model
        
        create_vid_encoder = video_encoder_entrypoints(video_encoder_configs.name)
        self.video_encoder = create_vid_encoder(video_encoder_configs,  d_model=d_model)
        
        create_text_encoder = text_encoder_entrypoints(text_encoder_configs.name)
        self.text_encoder = create_text_encoder(text_encoder_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_configs.name)
        self.feats_dancer = create_fusion_encoder(fusion_encoder_configs, d_model=d_model)
        
    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, visualize=False, targets=None, saved_path=None):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        
        (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=None,
                                    device=device,
                                    mask_sentence=False, masked_indices=None)  
                
        (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
            = self.video_encoder(videos=samples.tensors,
                                    video_pad_mask=samples.mask,
                                    valid_indices=valid_indices,
                                    mask_video=False, masked_indices=None) 
        
    
        refer_text_args = (token_feats, token_pad_mask, text_sentence_features)
        refer_video_args = (video_feats, video_pad_masks, video_poses , multiscale_dec)

        out, _ = self.feats_dancer(refer_video_args, refer_text_args, refer=True,
                                                    return_loss=False, targets=None)
                                                                                                
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 

        if visualize:
            assert saved_path is not None and targets is not None
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), 
                              text_query=text_queries,
                               directory=saved_path, 
                               targets=targets, refer_pred=copy.deepcopy(out), draw_all_instances=False)
        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)

        return output
    
    # get the loss, and the model has gradients;
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
                visualize=False, saved_path=None):

        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])

        video_inputs : Tensor = samples.tensors # t b c h w
        video_pad_masks = samples.mask
        device = video_inputs.device 
        
       
        (tokenized_result, tokenized_feats), (token_feats, token_pad_mask, text_sentence_features), _\
                = self.text_encoder(texts=copy.deepcopy(text_queries), 
                                    text_auxiliary=text_auxiliary,
                                    device=device,
                                    mask_sentence=False, masked_indices=None)  
                 
        (video_feats, video_pad_masks, video_poses, multiscale_dec), _\
            = self.video_encoder(videos=video_inputs.clone(),
                                    video_pad_mask=video_pad_masks.clone(),
                                    valid_indices=valid_indices.clone(),
                                    mask_video=False, masked_indices=None)       
        
        losses = {}

        refer_text_args = (token_feats.clone(), token_pad_mask.clone(), text_sentence_features.clone())
        refer_video_args = ([f.clone() for f in video_feats],  [m.clone() for m in video_pad_masks],\
                            [p.clone() for p in video_poses], multiscale_dec)

        refer_pred, refer_loss_dict = self.feats_dancer(refer_video_args, refer_text_args,
                                                        refer=True, return_loss=True, targets=targets
                                                        )
        # just three {loss_cre, loss_mask, loss_dice,}, no {loss_ce0, loss_mask0}
        losses.update(refer_loss_dict)
        
        assert set(losses.keys()).issubset(self.weight_dict.keys())

        loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        if visualize:
            assert saved_path is not None
            save_model_output(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                            targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
            
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

@register_rvos_model
def unimodal_encoder_modular_nmns(device, model_configs):
    pass

class MultimodalEncoder_Model(Video_Model):
    def __init__(self, loss_refer, loss_mlm, loss_mvm, mode_name, mask_sampling, clip_configs, d_model,
                multimodal_encoder_with_mlm_mvm_configs,
                fusion_encoder_with_refer_decoder,) -> None:
        super().__init__(loss_refer, loss_mlm, loss_mvm, mode_name, mask_sampling, clip_configs, d_model)
        # anything between encoders and decoders are fusion_encoder, 
        #fusion encoder负责实现forward refer
        create_multimodal_encoder = video_text_encoder_entrypoints(multimodal_encoder_with_mlm_mvm_configs.name)
        multimodal_encoder_with_mlm_mvm_heads = create_multimodal_encoder(multimodal_encoder_with_mlm_mvm_configs, d_model=d_model)
        
        create_fusion_encoder = fusion_entrypoints(fusion_encoder_with_refer_decoder.name)
        fusion_encoder_with_refer_decoder = create_fusion_encoder(fusion_encoder_with_refer_decoder, d_model=d_model)

        self.encoder = multimodal_encoder_with_mlm_mvm_heads
        self.fusion_encoder = fusion_encoder_with_refer_decoder

    @torch.no_grad()
    def sammple(self, samples, valid_indices, text_queries):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        video_args, _, _, text_args, _ \
                = self.encoder(# video
                            video_inputs=samples.tensors,
                            vids_mask=samples.mask,
                            valid_indices=valid_indices,
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=text_queries, 
                            text_auxiliary=None,
                            device=device,
                            mask_sentence=False, text_masked_indices=None)
        
        # (fused_video, fused_text), (refer_loss, refer_out)  
        _, (_, out) = self.fusion_encoder(video_args, text_args, return_refer=True, targets=None) 
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 
        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        return output
  
    
    def forward(self, samples, valid_indices, text_queries, targets, text_auxiliary):
        if self.mask_sampling == 'gradient':
            return self.forward_gradient_mask_sampling(samples, valid_indices, text_queries, targets, text_auxiliary)
        video_inputs : Tensor = samples.tensors
        video_pad_masks = samples.mask
        device = samples.tensors.device 
        
        if self.mask_sampling == 'clip':
            clip_video_feats, clip_text_feats, clip_video_attn, clip_text_attn = self.clip(video_inputs, text_queries)
            video_mask_indices = self.sample_video_indice(attn=clip_video_attn)
            text_mask_indices = self.sample_text_indice(attn=clip_text_attn)
        elif self.mask_sampling == 'random':
            video_mask_indices = None
            text_mask_indices = None
        else:
            raise ValueError()
        
        mlm_loss, mvm_loss, refer_loss = 0, 0, 0
    
        if self.loss_refer != 0:
            (video_args, _), (_, text_args, _) \
                = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None)
                
            _, _, (refer_loss, _) = self.fusion_encoder(video_args, text_args, return_refer=True)
                 
        if self.loss_mlm != 0:
            mlm_video_args, _, _, masked_text_args, masked_text_gt \
                = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=True, text_masked_indices=text_mask_indices)
                
            _, mlm_text_feats, _ = self.fusion_encoder(mlm_video_args, masked_text_args, return_refer=False)
            
            mlm_loss = self.encoder.forward_mlm(mlm_text_feats, masked_text_gt)
            
                 
        if self.loss_mvm != 0:
            masked_video_args, masked_video_gt, _, mvm_text_args, _ \
                = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=True, video_masked_indices=video_mask_indices,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None)
                
            mvm_video_feats,_, _ = self.fusion_encoder(masked_video_args, mvm_text_args, return_refer=False)
            
            mvm_loss = self.encoder.forward_mlm(mvm_video_feats, masked_video_gt)
    
        return refer_loss * self.loss_refer + mlm_loss * self.loss_mlm + mvm_loss * self.loss_mvm
             
    def forward_gradient_mask_sampling(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary):
        video_inputs : Tensor = samples.tensors
        video_pad_masks = samples.mask
        device = samples.tensors.device 
        
        video_inputs.requires_grad_(True)
        video_args, _, (tokenized_feats, ), text_args, _ \
            = self.encoder(# video
                           video_inputs=video_inputs.clone(),
                           vids_mask=video_pad_masks.clone(),
                           valid_indices=valid_indices.clone(),
                           mask_video=False, video_masked_indices=None,
                           
                           # text
                           texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None) 
        _, _, (refer_loss, _) = self.fusion_encoder(video_args, text_args, return_refer=True)
        refer_loss = self.loss_refer * refer_loss
        refer_loss.backward(retain_graph=True)
        
        mvm_loss, mlm_loss = 0
        if self.loss_mvm != 0:
            masked_video_indices = self.sample_video_indice(grad=video_inputs.grad.detach())
            
            masked_video_args, masked_video_gt, _, text_args, _ \
                = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=True, video_masked_indices=masked_video_indices,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None) 
            
            mvm_fused_video_feats, _, _ = self.fusion_encoder(masked_video_args, text_args, return_refer=False)
            
            mvm_loss = self.encoder.forward_mvm(mvm_fused_video_feats, masked_video_gt)
        
        if self.loss_mlm != 0:
            
            masked_text_indices = self.sample_text_indice(grad=tokenized_feats.grad)
            
            video_args, _, _, masked_text_args, masked_text_gt \
                = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=True, text_masked_indices=masked_text_indices) 

            _, mlm_fused_text_feats, _ = self.fusion_encoder(video_args, masked_text_args, return_refer=False)
            
            mlm_loss = self.encoder.forward_mlm(mlm_fused_text_feats, masked_text_gt)
        
        return self.loss_mlm * mlm_loss + self.loss_mvm * mvm_loss

@register_rvos_model
def multimodal_encoder(device, model_configs, fig_directory, dataset_name):

    configs = vars(model_configs)
        
    model = MultimodalEncoder_Model(
        loss_refer=configs['lambda_refer'],
        loss_mlm=configs['lambda_mlm'],
        loss_mvm=configs['lambda_mvm'],
        mode_name=configs['mode_name'],
        mask_sampling=configs['mask_sampling'],
        clip_configs=model_configs.clip,
        d_model=configs['d_model'],
        multimodal_encoder_with_mlm_mvm_configs=model_configs.encoder,
        fusion_encoder_with_refer_decoder=model_configs.fusion_encoder
    )
    model = model.to(device)
    
    if dataset_name == 'a2ds_rvos' or dataset_name == 'jhmdb_rvos':
        postprocessor = A2DSentencesPostProcess()
    elif dataset_name == 'youtube_rvos':
        postprocessor = ReferYoutubeVOSPostProcess()
    else:
        raise ValueError
    return model, postprocessor  


class QueryMultimodalEncoder_Model(Video_Model):
    def __init__(self, loss_refer, loss_mlm, loss_mvm, mode_name, mask_sampling, clip_configs, d_model,
                 multimodal_encoder_with_mlm_mvm_refer_configs) -> None:
        super().__init__(loss_refer, loss_mlm, loss_mvm, mode_name, mask_sampling, clip_configs, d_model)
        create_multimodal_encoder = video_text_encoder_entrypoints(multimodal_encoder_with_mlm_mvm_refer_configs.name)
        multimodal_encoder_with_mlm_mvm_heads = create_multimodal_encoder(multimodal_encoder_with_mlm_mvm_refer_configs, d_model=d_model)

        self.encoder = multimodal_encoder_with_mlm_mvm_heads
        

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries):
        device = samples.tensors.device
        nf, bs, *_ = samples.tensors.shape
        _, _, (_, out) = self.encoder(# video
                            video_inputs=samples.tensors,
                            vids_mask=samples.mask,
                            valid_indices=valid_indices,
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=text_queries, 
                            text_auxiliary=None,
                            device=device,
                            mask_sentence=False, text_masked_indices=None,
                            
                            # refer
                            return_refer=True
                            )
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        # final_token_feats: 
        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
        return output
  
    
    def forward(self, samples, valid_indices, text_queries, targets, text_auxiliary):
        video_inputs : Tensor = samples.tensors
        video_pad_masks = samples.mask
        device = samples.tensors.device 
        
        if self.mask_sampling == 'clip':
            clip_video_feats, clip_text_feats, clip_video_attn, clip_text_attn = self.clip(video_inputs, text_queries)
            video_mask_indices = self.sample_video_indice(attn=clip_video_attn)
            text_mask_indices = self.sample_text_indice(attn=clip_text_attn)
        elif self.mask_sampling == 'random':
            video_mask_indices = None
            text_mask_indices = None
        else:
            raise ValueError()
        
        mlm_loss, mvm_loss, refer_loss = 0, 0, 0
    
        if self.loss_refer != 0:
            _, _, (refer_loss, _) = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None,
                            return_refer=True, targets=targets)
                 
        if self.loss_mlm != 0:
            _, (mlm_loss, _), _ = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=False, video_masked_indices=None,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=True, text_masked_indices=text_mask_indices,
                            return_mlm=True, targets=targets)            
                 
        if self.loss_mvm != 0:
            (mvm_loss, _), _, _ = self.encoder(# video
                            video_inputs=video_inputs.clone(),
                            vids_mask=video_pad_masks.clone(),
                            valid_indices=valid_indices.clone(),
                            mask_video=True, video_masked_indices=video_mask_indices,
                            
                            # text
                            texts=copy.deepcopy(text_queries), 
                            text_auxiliary=text_auxiliary,
                            device=device,
                            mask_sentence=False, text_masked_indices=None,
                            return_mvm=True, targets=targets)
    
        return refer_loss * self.loss_refer + mlm_loss * self.loss_mlm + mvm_loss * self.loss_mvm
 
@register_rvos_model
def query_multimodal_encoder(device, model_configs, fig_directory, dataset_name):
    configs = vars(model_configs)
        
    model = QueryMultimodalEncoder_Model(
        loss_refer=configs['lambda_refer'],
        loss_mlm=configs['lambda_mlm'],
        loss_mvm=configs['lambda_mvm'],
        mode_name=configs['mode_name'],
        mask_sampling=configs['mask_sampling'],
        clip_configs=model_configs.clip,
        d_model=configs['d_model'],
        multimodal_encoder_with_mlm_mvm_refer_configs=model_configs.encoder,
    )
    model = model.to(device)
    
    if dataset_name == 'a2ds_rvos' or dataset_name == 'jhmdb_rvos':
        postprocessor = A2DSentencesPostProcess()
    elif dataset_name == 'youtube_rvos':
        postprocessor = ReferYoutubeVOSPostProcess()
    else:
        raise ValueError
    return model, postprocessor 

        
class QueryVideoEncoder_TextEncoder_Model(Video_Model):
    pass

@register_rvos_model
def query_multimodal_encoder(device, model_configs, fig_directory, dataset_name):
    pass

   
