import torch

import matplotlib.pyplot as plt
from einops import rearrange
from torch.nn.functional import pad
from torch.nn.functional import interpolate
import os

def visualize_test_batch(path, outputs, targets, samples, batch_size=3, window_size=7, num_valid=1, num_queries=5):
    """
    outputs: 'pred_masks': t_target*b N hidden_h hidden_w, (-1-1)
    targets: list[dict], t_target*b
    samples: t b 3 H W
    """
    # path = f'/home/xhh/workspace/rvos/models/referdiff_debug/after_epoch{}/test_batch{}'
    os.makedirs(path, exist_ok=True)
    for i in range(batch_size):
        visualize_tensor(samples[:, i,:,:,:].cpu(), 
                         path, 
                         f'video{i}',
                         f'the {i}th video of this batch after normalize augmentation',
                         num_samples=window_size, 
                         num_channels=3,
                         channel_split=False,)
    # t_target*b 1 H W
    gt_masks = torch.stack([target['masks'][target['referred_instance_idx']] for target in targets],dim=0).unsqueeze(1)
    
    # thresholding:
    pred_masks = (outputs['pred_masks'] > 0).float()
    # t_target*b 1 H W
    pred_masks = interpolate(pred_masks, size=gt_masks.shape[-2:], mode="nearest")
    
    # t_target*b 2 H W [0-1]
    masks = torch.cat((gt_masks, pred_masks), dim=1).cpu()
    visualize_tensor(masks, 
                     path, 
                     'gt_vs_pred_masks',
                     f'(t-target*b, N) left: ground truth, right: thresholded predicted masks,',
                     num_samples=batch_size*num_valid,
                     num_channels=1+num_queries,
                     channel_split=True)
    torch.save({'outputs':outputs, 'targets':targets, 'samples':samples}, f'{path}/data.pth')

def visualize_backbone_out(path,  last_layer_backbone_out, batch_size=3, window_size=7, valid_indice=3):
    """
    backbone_out: list[t b dim H/s W/s], #scales
    """
    pass
    
def visualize_spatial_decoder_out():
    pass

def visualize_transformer(path, vid_memory):
    pass

def visualize_diffusion_loss(path, model_out, masks, noised_masks, sampled_timesteps, batch_size=2, num_valid=1):
    '''
    model_output: t_target*b, 1, H/4, W/4
    noised_masks: t_target*b, 1, H/4, W/4, 
    masks: t_target*b, 1, H/4, W/4, [-scale, scale]
    '''
    # visualize_diffusion_loss(path, model_out.data, masks.data, noised_masks.data, sampled_timesteps.data):
    # path = f'/home/xhh/workspace/rvos/models/referdiff_debug/in_epoch4/train_batch1111'
    os.makedirs(path, exist_ok=True)
    tensor = torch.cat((masks, noised_masks, model_out),dim=1).cpu()
    visualize_tensor(tensor,
                     path,
                     'gt_vs_diffusion_out_masks',
                     str=f'gt i2b, noised [{sampled_timesteps.cpu()}], unet output',
                     num_samples=batch_size*num_valid,
                     num_channels=3,
                     channel_split=True,
                    )
    torch.save({'model_out':model_out, 
                'masks':masks, 
                'noised_mask': noised_masks, 
                'sampled_timesteps':sampled_timesteps},
               f'{path}/data.pth')
    

def visualize_tensor(tensors, path, tensors_name, str, num_samples, num_channels, channel_split=True, padding=10):
    """
    visualize tensors to images, save them to path
    input:
        tensors: [b, c, h, w], 
    output:
        save: a big image
            one row one sample
    """
    os.makedirs(path, exist_ok=True)
    num_b, num_c, h, w = tensors.shape
    assert num_b >= num_samples
    assert num_c >= num_channels
    if not channel_split:
        assert num_channels == 3
        
    tensors = pad(tensors, (padding,padding,padding,padding), value=1.)
    tensors = tensors[:num_samples, :num_channels, :, :]
    if not channel_split:
        tensors = rearrange(tensors, 'b c h w -> (b h) w c')
    else:
        tensors = rearrange(tensors, 'b c h w -> (b h) (c w)')

    fig, ax = plt.subplots(figsize=((tensors.shape[1]+200)/100,(tensors.shape[0]+200)/100))
    if not channel_split:
        im = ax.imshow(tensors)
    else:
        im = ax.imshow(tensors, cmap='gray')
    ax.axis('off')
    fig.colorbar(im)
    fig.suptitle(str)
    fig.savefig(f'{path}/{tensors_name}.png')
    plt.close()
    

def visualize_one_sample(sample, path):
    """input:
        sample: dict{'name':tensor}
    """
    pass

if __name__ == '__main__':
    tensors = torch.randn([10,2,80, 90])
    path = './'
    tensors_name = 'pred_masks'
    visualize_tensor(tensors, path, 
                     tensors_name, 
                     f'(t-target*b, 2) left: ground truth, right: thresholded predicted masks,',
                     num_samples=10, 
                     num_channels=2, 
                     channel_split=True, 
                     padding=4)
    