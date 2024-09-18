# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torchvision.models.resnet import resnet50
from detectron2.modeling import BACKBONE_REGISTRY
import models.backbone.mae_funduns.models_vit as models_vit

from .pos_embed import interpolate_pos_embed
import torch.nn as nn
import os

from functools import partial
from einops import rearrange

@BACKBONE_REGISTRY.register()
class MAE_FUNDUNS(nn.Module):

    def __init__(self, configs,):
        super().__init__()


        # call the model
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=2,
            drop_path_rate=0.2,
        )

        # load RETFound weights
        checkpoint = torch.load(os.path.join(os.environ.get('PT_PATH'),'mae_funduns/RETFound_cfp_weights.pth'), map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        
        self.ssl = model
        for p in self.ssl.parameters():
            p.requires_grad_(False) 

        self.embed_dim = self.ssl.embed_dim
        self.patch_size = self.ssl.patch_embed.patch_size[0]


    def forward(self, x, n=1):
        assert n == 1
        return {
            'features': [self.ssl.forward_features(x)], # b cls+hw c
        }



