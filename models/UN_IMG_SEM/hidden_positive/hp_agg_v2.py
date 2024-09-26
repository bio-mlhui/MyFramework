"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import os
from models.UN_IMG_SEM.AggSample.code import aggo_whole_batch
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from functools import partial
import logging
from models.registry import register_model
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from models.UN_IMG_SEM.stego.utils.common_utils import freeze_bn, zero_grad_bn
from .LambdaLayer import LambdaLayer
from .loss import StegoLoss
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)



class OptimizeModel(nn.Module):
    """
    optimize_setup:
        optimizer, scheduler都是标准类
        log_lr_idx随着训练不改变
        
    optimize:
        backward, optimzier_step, optimizer_zero_grad, scheduler_step
        
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.log_lr_group_idx: Dict = None

    def optimize_setup(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

def build_optimizer(main_params, linear_params, cluster_params, opt: dict, model_type: str):
    # opt = opt["optimizer"]
    model_type = model_type.lower()

    if "stego" in model_type:
        net_optimizer_type = opt["net"]["name"].lower()
        if net_optimizer_type == "adam":
            net_optimizer = Adam(main_params, lr=opt["net"]["lr"])
        elif net_optimizer_type == "adamw":
            net_optimizer = AdamW(main_params, lr=opt["net"]["lr"], weight_decay=opt["net"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {net_optimizer_type}.")

        linear_probe_optimizer_type = opt["linear"]["name"].lower()
        if linear_probe_optimizer_type == "adam":
            linear_probe_optimizer = Adam(linear_params, lr=opt["linear"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {linear_probe_optimizer_type}.")

        cluster_probe_optimizer_type = opt["cluster"]["name"].lower()
        if cluster_probe_optimizer_type == "adam":
            cluster_probe_optimizer = Adam(cluster_params, lr=opt["cluster"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {cluster_probe_optimizer_type}.")


        return net_optimizer, linear_probe_optimizer, cluster_probe_optimizer

    else:
        raise ValueError("No model: {} found".format(model_type))

class AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                # list[3 3 h w] -> b 3 3 h w
                'img': torch.stack([item['image'] for item in batch_dict], dim=0),
                'label': torch.stack([item['mask'] for item in batch_dict], dim=0),
                'img_aug': torch.stack([item['img_aug'] for item in batch_dict], dim=0),
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict],
                'image_ids':[item['image_id'] for item in batch_dict],
            }
        elif mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                },
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()



class ClusterLookup_v2(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1) # K d
            normed_features = F.normalize(x, dim=-1) # N d
            inner_products = torch.einsum("nd,kd->nk", normed_features, normed_clusters) # N k

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=-1), self.clusters.shape[0]).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs

class DinoFeaturizer_v2(nn.Module):

    def __init__(self, dim, cfg):  # cfg["pretrained"]
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.feat_type = self.cfg["pretrained"]["dino_feat_type"]
        dino_configs = self.cfg["pretrained"]
        self.sigma = 28
        self.dropout = torch.nn.Dropout2d(p=.1)

        self.model = BACKBONE_REGISTRY.get(dino_configs['name'])(dino_configs)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()

        self.n_feats = self.model.embed_dim
        self.patch_size = self.model.patch_size

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg["pretrained"]["projection_type"]
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        self.nq_cluster = self.make_clusterer(self.n_feats)

        self.ema_model1 = self.make_clusterer(self.n_feats)
        self.ema_model2 = self.make_nonlinear_clusterer(self.n_feats)

        for param_q, param_k in zip(self.cluster1.parameters(), self.ema_model1.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model1.cuda()
        self.ema_model1.eval()

        for param_q, param_k in zip(self.cluster2.parameters(), self.ema_model2.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model2.cuda()
        self.ema_model2.eval()

        sz = cfg["spatial_size"]

        self.index_mask = torch.zeros((sz*sz, sz*sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz*sz), dtype=torch.long)
        for _im in range(sz*sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im+1, _im+sz, _im+(sz+1)])
            elif _im==(sz-1):
                index_set = torch.tensor([_im-1, _im, _im+(sz-1), _im+sz])
            elif _im==(sz*sz-sz):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1])
            elif _im==(sz*sz-1):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im])

            elif ((1 <= _im) and (_im <= (sz-2))):
                index_set = torch.tensor([_im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            elif (((sz*sz-sz+1) <= _im) and (_im <= (sz*sz-2))):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1, _im+sz, _im+(sz+1)])
            elif ((_im+1) % sz == 0):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im, _im+(sz-1), _im+sz])
            else:
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num[_im] = index_set.size(0)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()

        batch_size = cfg['batch_size']
        train_size = cfg['train_size']
        eval_size = cfg['eval_size']
        self.init_graph_utils(batch_size, train_size, eval_size)

    def init_graph_utils(self, train_batch_size, train_size, eval_size):
        # test=1
        H = W = (train_size // self.patch_size) 
        num_train_nodes = H*W
        train_edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    train_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    train_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        train_edge_index = torch.tensor(train_edge_index).permute(1, 0).contiguous().long() # 2 N

        H = W = (eval_size // self.patch_size)
        num_test_nodes = H*W
        eval_edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    eval_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    eval_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        eval_edge_index = torch.tensor(eval_edge_index).permute(1, 0).contiguous().long()

        from torch_geometric.data import Batch, Data
        train_graph = Data(edge_index=train_edge_index,)
        train_graph.num_nodes = num_train_nodes
        eval_graph = Data(edge_index=eval_edge_index)
        eval_graph.num_nodes = num_test_nodes

        whole_train_graph = [train_graph.clone() for _ in range(train_batch_size)]
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in whole_train_graph]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in whole_train_graph]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        whole_train_graph = Batch.from_data_list(whole_train_graph)

        self.register_buffer('train_node_batch_ids', torch.tensor(nodes_batch_ids).int())
        self.register_buffer('train_edge_batch_ids', torch.tensor(edges_batch_ids).int())
        self.register_buffer('train_edge_index', whole_train_graph.edge_index.long())
        self.register_buffer('train_num_patches', torch.ones(len(nodes_batch_ids)).int())

        self.register_buffer('eval_node_batch_ids', torch.zeros(eval_graph.num_nodes).int())
        self.register_buffer('eval_edge_batch_ids', torch.zeros(eval_graph.num_edges).int())
        self.register_buffer('eval_edge_index', eval_graph.edge_index.long())
        self.register_buffer('eval_num_patches', torch.ones(eval_graph.num_nodes).int())


    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            nn.Linear(in_channels, self.dim),)

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            nn.Linear(in_channels, in_channels,),
            torch.nn.ReLU(),
            nn.Linear(in_channels, self.dim))

    @torch.no_grad()
    def ema_model_update(self, model, ema_model, ema_m):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
            param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def forward(self, img, n=1, return_class_feat=False, train=False, only_pool=False):
        self.model.eval()
        batch_size = img.shape[0]

        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            rets = self.model(img, n=n)
            # b cls+hw c, 
            feat, attn, qkv = rets['features'], rets['attentions'], rets['qkvs']
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            if train==True:
                attn = attn[:, :, 1:, 1:]
                attn = torch.mean(attn, dim=1)
                attn = attn.type(torch.float32)
                attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
                attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
                attn = torch.max(torch.min(attn, attn_max), attn_min)

                attn = attn.softmax(dim=-1)
                attn = attn*self.sigma
                attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1)
            elif self.feat_type == "KK":
                raise ValueError() # 6?
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))


            if not only_pool:
                graph_node_features = image_feat.flatten(0, 2) # bhw c
                # list[list[ni c], threshold] b
                nquery_feats = aggo_whole_batch(nodes_feature=graph_node_features,
                                                edge_index= self.train_edge_index if self.training else self.eval_edge_index,
                                                node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids,
                                                edge_batch_tensor=self.train_edge_batch_ids if self.training else self.eval_edge_batch_ids,
                                                node_num_patches=self.train_num_patches if self.training else self.eval_num_patches,) 
                nquery_feats = [torch.cat(foo, dim=0) for foo in nquery_feats] # list[ni c], batch
                nquery_splits = [len(foo) for foo in nquery_feats] # list[int], batch

                orig_nquery_attns = [] # list[h w ni] batch
                for batch_idx in range(batch_size):
                    nq_attn = torch.einsum('hwc,nc->hwn', 
                                            F.normalize(image_feat[batch_idx], dim=-1, eps=1e-10), 
                                            F.normalize(nquery_feats[batch_idx], dim=-1, eps=1e-10)) # [-1, 1])
                    orig_nquery_attns.append(nq_attn)
                nquery_feats = torch.cat(nquery_feats, dim=0) # b_ni c 

            image_feat = image_feat.permute(0, 3, 1, 2) # b c h w
                    
        B, _, H, W = image_feat.shape

        code = self.cluster1(self.dropout(image_feat).permute(0, 2,3,1).flatten(0, 2))
        code += self.cluster2(self.dropout(image_feat).permute(0, 2,3,1).flatten(0, 2))
        code = rearrange(code, '(b h w) c -> b c h w',b=B,h=H,w=W)

        code_ema = self.ema_model1(self.dropout(image_feat).permute(0, 2,3,1).flatten(0, 2))
        code_ema += self.ema_model2(self.dropout(image_feat).permute(0, 2,3,1).flatten(0, 2))
        code_ema = rearrange(code_ema, '(b h w) c -> b c h w',b=B,h=H,w=W)

        if not only_pool:
            code_query = self.nq_cluster(nquery_feats) # b_ni c'

            after_nquery_attns = [] # list[h w ni] batch
            code_query_splits = code_query.split(nquery_splits) # list[ni c], batch
            for batch_idx in range(batch_size):
                nq_attn = torch.einsum('chw,nc->hwn',
                                    F.normalize(code[batch_idx], dim=0), 
                                    F.normalize(code_query_splits[batch_idx], dim=-1)) # [-1, 1])
                after_nquery_attns.append(nq_attn)
        if only_pool:
            nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns = None, None, None, None, None

        if train==True:
            attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            code_clone = code.clone()
            code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
            code_clone = code_clone.permute(0,2,1)

            code_3x3_all = []
            for bs in range(batch_size):
                code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
                code_3x3 = torch.sum(code_3x3, dim=1)
                code_3x3 = code_3x3 / self.divide_num
                code_3x3_all.append(code_3x3)
            code_3x3_all = torch.stack(code_3x3_all)
            code_3x3_all = code_3x3_all.permute(0,2,1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        if train==True:
            with torch.no_grad():
                self.ema_model_update(self.cluster1, self.ema_model1, self.cfg["ema_m"])
                self.ema_model_update(self.cluster2, self.ema_model2, self.cfg["ema_m"])

        if train==True:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema), self.dropout(code_3x3_all),\
                        nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns
            else:
                return image_feat, code, code_ema, code_3x3_all,\
                        nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns
        else:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema),\
                        nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns
            else:
                return image_feat, code, code_ema,\
                        nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):  # cfg["pretrained"]
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.feat_type = self.cfg["pretrained"]["dino_feat_type"]
        dino_configs = self.cfg["pretrained"]
        self.sigma = 28
        self.dropout = torch.nn.Dropout2d(p=.1)

        self.model = BACKBONE_REGISTRY.get(dino_configs['name'])(dino_configs)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()

        self.n_feats = self.model.embed_dim
        self.patch_size = self.model.patch_size

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg["pretrained"]["projection_type"]
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        self.ema_model1 = self.make_clusterer(self.n_feats)
        self.ema_model2 = self.make_nonlinear_clusterer(self.n_feats)

        for param_q, param_k in zip(self.cluster1.parameters(), self.ema_model1.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model1.cuda()
        self.ema_model1.eval()

        for param_q, param_k in zip(self.cluster2.parameters(), self.ema_model2.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model2.cuda()
        self.ema_model2.eval()

        sz = cfg["spatial_size"]

        self.index_mask = torch.zeros((sz*sz, sz*sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz*sz), dtype=torch.long)
        for _im in range(sz*sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im+1, _im+sz, _im+(sz+1)])
            elif _im==(sz-1):
                index_set = torch.tensor([_im-1, _im, _im+(sz-1), _im+sz])
            elif _im==(sz*sz-sz):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1])
            elif _im==(sz*sz-1):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im])

            elif ((1 <= _im) and (_im <= (sz-2))):
                index_set = torch.tensor([_im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            elif (((sz*sz-sz+1) <= _im) and (_im <= (sz*sz-2))):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1, _im+sz, _im+(sz+1)])
            elif ((_im+1) % sz == 0):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im, _im+(sz-1), _im+sz])
            else:
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num[_im] = index_set.size(0)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()

        batch_size = cfg['batch_size']
        train_size = cfg['train_size']
        eval_size = cfg['eval_size']
        self.init_graph_utils(batch_size, train_size, eval_size)

    def init_graph_utils(self, train_batch_size, train_size, eval_size):
        # test=1
        H = W = (train_size // self.patch_size) 
        num_train_nodes = H*W
        train_edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    train_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    train_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        train_edge_index = torch.tensor(train_edge_index).permute(1, 0).contiguous().long() # 2 N

        H = W = (eval_size // self.patch_size)
        num_test_nodes = H*W
        eval_edge_index = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    eval_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    eval_edge_index.append([cluster_idx-W, cluster_idx])
                cluster_idx += 1
        eval_edge_index = torch.tensor(eval_edge_index).permute(1, 0).contiguous().long()

        from torch_geometric.data import Batch, Data
        train_graph = Data(edge_index=train_edge_index,)
        train_graph.num_nodes = num_train_nodes
        eval_graph = Data(edge_index=eval_edge_index)
        eval_graph.num_nodes = num_test_nodes

        whole_train_graph = [train_graph.clone() for _ in range(train_batch_size)]
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in whole_train_graph]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in whole_train_graph]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        whole_train_graph = Batch.from_data_list(whole_train_graph)

        self.register_buffer('train_node_batch_ids', torch.tensor(nodes_batch_ids).int())
        self.register_buffer('train_edge_batch_ids', torch.tensor(edges_batch_ids).int())
        self.register_buffer('train_edge_index', whole_train_graph.edge_index.long())
        self.register_buffer('train_num_patches', torch.ones(len(nodes_batch_ids)).int())

        self.register_buffer('eval_node_batch_ids', torch.zeros(eval_graph.num_nodes).int())
        self.register_buffer('eval_edge_batch_ids', torch.zeros(eval_graph.num_edges).int())
        self.register_buffer('eval_edge_index', eval_graph.edge_index.long())
        self.register_buffer('eval_num_patches', torch.ones(eval_graph.num_nodes).int())


    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer_layer3(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    @torch.no_grad()
    def ema_model_update(self, model, ema_model, ema_m):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
            param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def forward(self, img, n=1, return_class_feat=False, train=False, only_pool=True):
        self.model.eval()
        batch_size = img.shape[0]

        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            rets = self.model(img, n=n)
            # b cls+hw c, 
            feat, attn, qkv = rets['features'], rets['attentions'], rets['qkvs']
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            if train==True:
                attn = attn[:, :, 1:, 1:]
                attn = torch.mean(attn, dim=1)
                attn = attn.type(torch.float32)
                attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
                attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
                attn = torch.max(torch.min(attn, attn_max), attn_min)

                attn = attn.softmax(dim=-1)
                attn = attn*self.sigma
                attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1)
            elif self.feat_type == "KK":
                raise ValueError() # 6?
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))
            if not only_pool:
                graph_node_features = image_feat.flatten(0, 2) # bhw c
                # list[list[ni c], threshold] b
                nquery_feats = aggo_whole_batch(nodes_feature=graph_node_features,
                                                edge_index= self.train_edge_index if self.training else self.eval_edge_index,
                                                node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids,
                                                edge_batch_tensor=self.train_edge_batch_ids if self.training else self.eval_edge_batch_ids,
                                                node_num_patches=self.train_num_patches if self.training else self.eval_num_patches,) 
                nquery_feats = [torch.cat(foo, dim=0) for foo in nquery_feats] # list[ni c], batch
                nquery_splits = [len(foo) for foo in nquery_feats] # list[int], batch

                orig_nquery_attns = [] # list[h w ni] batch
                for batch_idx in range(batch_size):
                    nq_attn = torch.einsum('hwc,nc->hwn', 
                                            F.normalize(image_feat[batch_idx], dim=-1, eps=1e-10), 
                                            F.normalize(nquery_feats[batch_idx], dim=-1, eps=1e-10)) # [-1, 1])
                    orig_nquery_attns.append(nq_attn)
                nquery_feats = torch.cat(nquery_feats, dim=0) # b_ni c 
            
            image_feat = image_feat.permute(0, 3, 1, 2) # b c h w

        # 做一个attention看看有没有用
        code_input_feats
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            code_ema = self.ema_model1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
                code_ema += self.ema_model2(self.dropout(image_feat))
        else:
            code = image_feat

        if train==True:
            attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            code_clone = code.clone()
            code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
            code_clone = code_clone.permute(0,2,1)

            code_3x3_all = []
            for bs in range(batch_size):
                code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
                code_3x3 = torch.sum(code_3x3, dim=1)
                code_3x3 = code_3x3 / self.divide_num
                code_3x3_all.append(code_3x3)
            code_3x3_all = torch.stack(code_3x3_all)
            code_3x3_all = code_3x3_all.permute(0,2,1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        if train==True:
            with torch.no_grad():
                self.ema_model_update(self.cluster1, self.ema_model1, self.cfg["ema_m"])
                self.ema_model_update(self.cluster2, self.ema_model2, self.cfg["ema_m"])

        if train==True:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema), self.dropout(code_3x3_all)
            else:
                return image_feat, code, code_ema, code_3x3_all
        else:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema)
            else:
                return image_feat, code, code_ema


class STEGOmodel(nn.Module):
    # opt["model"]
    def __init__(self,
                 opt: dict,
                 n_classes:int
                 ):
        super().__init__()
        self.opt = opt
        self.n_classes= n_classes


        if not opt["continuous"]:
            dim = n_classes
        else:
            dim = opt["dim"]

        if opt["arch"] == "dino":
            self.net = DinoFeaturizer(dim, opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Linear(dim, n_classes)
        
    def forward(self, x: torch.Tensor):
        return self.net(x)[1]

    @classmethod
    def build(cls, opt, n_classes):
        # opt = opt["model"]
        m = cls(
            opt = opt,
            n_classes= n_classes
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count

def build_model(opt: dict, n_classes: int = 27, is_direct: bool = False):
    model_type = opt["model_type"].lower()

    if "stego" in model_type:
        model = STEGOmodel.build(
            opt=opt,
            n_classes=n_classes
        )
        net_model = model.net
        linear_model = model.linear_probe
        cluster_model = model.cluster_probe
        
    elif model_type == "dino":
        model = nn.Sequential(
            DinoFeaturizer(20, opt),
            LambdaLayer(lambda p: p[0])
        )

    else:
        raise ValueError("No model: {} found".format(model_type))

    bn_momentum = opt.get("bn_momentum", None)
    if bn_momentum is not None:
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.momentum = bn_momentum

    bn_eps = opt.get("bn_eps", None)
    if bn_eps is not None:
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eps = bn_eps

    if "stego" in model_type:
        return net_model, linear_model, cluster_model
    elif model_type == "dino":
        return model

def build_criterion(n_classes: int, opt: dict):
    # opt = opt["loss"]
    loss_name = opt["name"].lower()
    if "stego" in loss_name:
        loss = StegoLoss(n_classes=n_classes, cfg=opt, corr_weight=opt["correspondence_weight"])
    else:
        raise ValueError(f"Unsupported loss type {loss_name}")

    return loss
from .make_reference_pool import renew_reference_pool

def nquery_loss(orig_attn, after_attn, orig_temperature, after_temperature):
    # list[h w ni], batch #-1,1
    # list[h w ni], batch #dot_product
    loss = []
    batch_size = len(orig_attn)
    for batch_idx in range(batch_size):
        orig_prob = F.softmax(orig_attn[batch_idx].flatten(0,1) / orig_temperature, dim=-1) # hw ni
        loss.append(F.cross_entropy(after_attn[batch_idx].flatten(0,1), target=orig_prob, reduction='none'))
    return torch.cat(loss).mean()


class HP_AGG(OptimizeModel):
    def __init__(
        self,
        configs,
        num_classes,
        train_loader_memory,
        device,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],
        batch_size=None,
        train_size=None,
        eval_size=None):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        model_configs = configs['model']
        model_configs['batch_size'] = batch_size
        model_configs['train_size'] = train_size
        model_configs['eval_size'] = eval_size
        self.nq_temperature = model_configs['nq_temperature']
        # self.nquery_loss = Nquery_Loss(model_configs)
        self.net_model, self.linear_model, self.cluster_model = build_model(opt=model_configs,
                                                                            n_classes=num_classes,
                                                                            is_direct=configs["eval"]["is_direct"]) 
        self.patch_size = self.net_model.patch_size
        self.criterion = build_criterion(n_classes=num_classes, opt=configs["loss"]) 
        self.net_model.to(device)
        self.linear_model.to(device)
        self.cluster_model.to(device)
               
        self.project_head = nn.Linear(model_configs['dim'], model_configs['dim'])
        self.project_head.to(device)
        self.head_optimizer = Adam(self.project_head.parameters(), lr=configs['optimizer']["net"]["lr"])

        from .loss import SupConLoss
        self.supcon_criterion = SupConLoss(temperature=configs["tau"])
        self.pd = nn.PairwiseDistance()
        self.feat_dim =  self.net_model.n_feats 
        self.configs = configs
        self.maxiter = len(train_loader_memory) * configs['optim']['epochs']   # 45795
        self.train_loader_memory = train_loader_memory
        self.freeze_encoder_bn = configs["train"]["freeze_encoder_bn"]
        self.freeze_all_bn = configs["train"]["freeze_all_bn"]
        self.grad_norm = configs["train"]["grad_norm"]
        self.num_queries = num_classes
        self.kmeans_strategy = 'adaptive'

    @property
    def device(self):
        return self.pixel_mean.device
        
    def optimize_setup(self, configs):
        self.net_optimizer, self.linear_probe_optimizer, self.cluster_probe_optimizer = build_optimizer(
            main_params=self.net_model.parameters(),
            linear_params=self.linear_model.parameters(),
            cluster_params=self.cluster_model.parameters(),
            opt=configs['optimizer'],
            model_type=configs["model"]["model_type"])

    def forward_backward_v2(self, batch_dict):
        trainingiter = batch_dict['num_iterations']
        current_epoch = batch_dict['num_epoch']
        device = self.device
        assert self.training
        
        if trainingiter <= self.configs["model"]["warmup"]:
            lmbd = 0
        else:
            lmbd = (trainingiter - self.configs["model"]["warmup"]) / (self.maxiter - self.configs["model"]["warmup"])

        if trainingiter % self.configs["renew_interval"] == 0 and trainingiter!= 0:
            self.Pool_sp = renew_reference_pool(self.net_model, self.train_loader_memory, self.configs, self.device,
                                                pixel_mean=self.pixel_mean, pixel_std=self.pixel_std)
            
        img: torch.Tensor = batch_dict['img'].to(device, non_blocking=True)
        label: torch.Tensor = batch_dict['label'].to(device, non_blocking=True)

        img_aug = batch_dict['img_aug'].to(device, non_blocking=True)

        with torch.no_grad():
            img = (img - self.pixel_mean) / self.pixel_std
            img_aug = (img_aug - self.pixel_mean) / self.pixel_std

        if self.freeze_encoder_bn:
            freeze_bn(self.net_model.model)
        if 0 < self.freeze_all_bn <= current_epoch:
            freeze_bn(self.net_model)

        batch_size = img.shape[0]
        self.net_optimizer.zero_grad(set_to_none=True)
        self.linear_probe_optimizer.zero_grad(set_to_none=True)
        self.cluster_probe_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)

        model_input = (img, label)

        with torch.cuda.amp.autocast(enabled=True):
            model_output = self.net_model(img, train=True)
            model_output_aug = self.net_model(img_aug)

        modeloutput_f = model_output[0].clone().detach().permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        modeloutput_f = F.normalize(modeloutput_f, dim=1)

        modeloutput_s = model_output[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        modeloutput_s_aug = model_output_aug[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z = self.project_head(modeloutput_s)
            modeloutput_z_aug = self.project_head(modeloutput_s_aug)
        modeloutput_z = F.normalize(modeloutput_z, dim=1)
        modeloutput_z_aug = F.normalize(modeloutput_z_aug, dim=1)

        loss_consistency = torch.mean(self.pd(modeloutput_z, modeloutput_z_aug))

        modeloutput_s_mix = model_output[3].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z_mix = self.project_head(modeloutput_s_mix)
        modeloutput_z_mix = F.normalize(modeloutput_z_mix, dim=1)

        modeloutput_s_pr = model_output[2].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        modeloutput_s_pr = F.normalize(modeloutput_s_pr, dim=1)

        loss_supcon = self.supcon_criterion(modeloutput_z, modeloutput_s_pr=modeloutput_s_pr, modeloutput_f=modeloutput_f,
                                Pool_ag=self.Pool_ag, Pool_sp=self.Pool_sp,
                                opt=self.configs, lmbd=lmbd, modeloutput_z_mix=modeloutput_z_mix)


        # b_ni c, 
        # b_ni c', 
        # list[int],batch, 
        # lis[h w ni], batch
        # lis[h w ni], batch
        nquery_feats, code_query, nquery_splits, orig_nquery_attns, after_nquery_attns\
              = model_output[4], model_output[5].clone(), model_output[6], model_output[7], model_output[8]
        # with torch.cuda.amp.autocast(enabled=True):
        #     code_query = self.project_head(code_query)
        # -1, 1; to dot_product
        # h w nq(-1,1), h w nq.sigmoid(0, 1)
        loss_nq = nquery_loss(orig_attn=orig_nquery_attns, 
                              after_attn=after_nquery_attns, orig_temperature=0.07, after_temperature=0.07)

        detached_code_nq = torch.clone(model_output[5].detach()) # b_ni c'
        after_nquery_attns = [foo.detach()for foo in after_nquery_attns]
        with torch.cuda.amp.autocast(enabled=True):
            linear_output = self.linear_model(detached_code_nq)
            # cluster_output = self.cluster_model(detached_code_nq, None, is_direct=False) # alpha=None, -> 输出argmax-onehot, log_probs=False, 输出softmax
            # b_ni K, logits
            cluster_output = self.cluster_model(detached_code_nq, alpha=1./0.07, log_probs=True, is_direct=False) 
            linear_output = linear_output.split(nquery_splits) #list[ni k], batch
            cluster_nqs = cluster_output[1].split(nquery_splits) #list[ni k], batch
            cluster_loss = cluster_output[0]
            assert len(linear_output) == len(after_nquery_attns)
            linear_mask_pred = torch.stack([torch.einsum('hwn,nk->khw', nq_attn.softmax(-1), feat)  \
                                            for feat, nq_attn in zip(linear_output, after_nquery_attns)],dim=0)
            cluster_mask_pred = torch.stack([torch.einsum('hwn,nk->khw', nq_attn.softmax(-1), feat)  \
                                             for feat, nq_attn in zip(cluster_nqs, after_nquery_attns)],dim=0)
            loss, loss_dict, corr_dict = self.criterion(model_input=model_input,
                                                    model_output=model_output,
                                                    linear_output=linear_mask_pred,
                                                    cluster_output=(cluster_loss, cluster_mask_pred)
                                                    )

            loss = loss + loss_supcon + loss_consistency * self.configs['alpha'] + loss_nq


        self.scaler.scale(loss).backward()

        if self.freeze_encoder_bn:
            zero_grad_bn(self.net_model)
        if 0 < self.freeze_all_bn <= current_epoch:
            zero_grad_bn(self.net_model)

        self.scaler.unscale_(self.net_optimizer)

        g_norm = nn.utils.clip_grad_norm_(self.net_model.parameters(), self.grad_norm)
        self.scaler.step(self.net_optimizer)

        self.scaler.step(self.linear_probe_optimizer)
        self.scaler.step(self.cluster_probe_optimizer)
        self.scaler.step(self.head_optimizer)

        self.scaler.update()


        loss_dict['loss'] = loss.cpu().item()
        loss_dict['loss_supcon'] = loss_supcon.cpu().item()
        loss_dict['loss_consistency'] = loss_consistency.cpu().item()
        loss_dict['loss_nq'] = loss_nq.cpu().item()
        return loss_dict

    @torch.no_grad()
    def sample_v2(self, batch_dict, visualize_all=False):
        assert not self.training
        img: torch.Tensor = batch_dict['images'].to(self.device, non_blocking=True)
        img = (img - self.pixel_mean) / self.pixel_std
        B, _, H, W = img.shape
        label: torch.Tensor = batch_dict['masks'].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            output = self.net_model(img)

        backbone_feats = output[0] # b c h w
        head_code = output[1].float() # b c h w

        sampled_points, similarities, backbone_similarities = None, None, None
        kmeans_preds, num_kmeans_classes, kmeans_preds_backbone = None, None, None,
        if visualize_all:
            sampled_points, similarities, backbone_similarities = self.sample_point_similarities(code_features=head_code,
                                                                                                 backbone_features=backbone_feats, 
                                                                                                 num_points=10)
            kmeans_preds, _ , num_kmeans_classes = self.self_cluster(head_code, label)
            kmeans_preds_backbone, _ , _ = self.self_cluster(backbone_feats, label)

        nquery_feat, code_query, nquery_splits, after_nquery_attns = output[3],output[4],output[5],output[7]
        
        with torch.cuda.amp.autocast(enabled=True):
            linear_preds = self.linear_model(code_query)
        with torch.cuda.amp.autocast(enabled=True):
            # c' logits -> K logits
            _, cluster_preds = self.cluster_model(code_query, alpha=1./0.07, log_probs=True, is_direct=self.configs["eval"]["is_direct"])

        linear_preds = linear_preds.float().split(nquery_splits) # list[ni k], batch
        cluster_preds = cluster_preds.split(nquery_splits) # list[ni k], batch
        after_nquery_attns = [foo.softmax(-1).float() for foo in after_nquery_attns] # list[h w ni], batch, logits

        linear_preds = torch.stack([torch.einsum('nk,hwn->khw', linear_preds[batch_idx], after_nquery_attns[batch_idx]) \
            for batch_idx in range(B)], dim=0)
        cluster_preds = torch.stack([torch.einsum('nk,hwn->khw', cluster_preds[batch_idx], after_nquery_attns[batch_idx]) \
            for batch_idx in range(B)], dim=0)       
        
        linear_preds = F.interpolate(linear_preds, size=(H, W), mode='bilinear', align_corners=False)
        cluster_preds = F.interpolate(cluster_preds, size=(H, W), mode='bilinear', align_corners=False)
        return  {
            'linear_preds': linear_preds,
            'cluster_preds': cluster_preds,

            'sampled_points': sampled_points,
            'similarities': similarities,
            'backbone_similarities': backbone_similarities,

            'kmeans_preds': kmeans_preds,
            'kmeans_preds_bb': kmeans_preds_backbone,
            'num_kmeans_classes': num_kmeans_classes,
        }

    def forward_backward(self, batch_dict):
        trainingiter = batch_dict['num_iterations']
        current_epoch = batch_dict['num_epoch']
        device = self.device
        assert self.training
        
        if trainingiter <= self.configs["model"]["warmup"]:
            lmbd = 0
        else:
            lmbd = (trainingiter - self.configs["model"]["warmup"]) / (self.maxiter - self.configs["model"]["warmup"])

        if trainingiter % self.configs["renew_interval"] == 0 and trainingiter!= 0:
            self.Pool_sp = renew_reference_pool(self.net_model, self.train_loader_memory, self.configs, self.device,
                                                pixel_mean=self.pixel_mean, pixel_std=self.pixel_std)
            
        img: torch.Tensor = batch_dict['img'].to(device, non_blocking=True)
        label: torch.Tensor = batch_dict['label'].to(device, non_blocking=True)

        img_aug = batch_dict['img_aug'].to(device, non_blocking=True)

        with torch.no_grad():
            img = (img - self.pixel_mean) / self.pixel_std

        if self.freeze_encoder_bn:
            freeze_bn(self.net_model.model)
        if 0 < self.freeze_all_bn <= current_epoch:
            freeze_bn(self.net_model)

        batch_size = img.shape[0]
        self.net_optimizer.zero_grad(set_to_none=True)
        self.linear_probe_optimizer.zero_grad(set_to_none=True)
        self.cluster_probe_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)

        model_input = (img, label)

        with torch.cuda.amp.autocast(enabled=True):
            model_output = self.net_model(img, train=True)
            model_output_aug = self.net_model(img_aug)

        modeloutput_f = model_output[0].clone().detach().permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        modeloutput_f = F.normalize(modeloutput_f, dim=1)

        modeloutput_s = model_output[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        modeloutput_s_aug = model_output_aug[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z = self.project_head(modeloutput_s)
            modeloutput_z_aug = self.project_head(modeloutput_s_aug)
        modeloutput_z = F.normalize(modeloutput_z, dim=1)
        modeloutput_z_aug = F.normalize(modeloutput_z_aug, dim=1)

        loss_consistency = torch.mean(self.pd(modeloutput_z, modeloutput_z_aug))

        modeloutput_s_mix = model_output[3].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z_mix = self.project_head(modeloutput_s_mix)
        modeloutput_z_mix = F.normalize(modeloutput_z_mix, dim=1)

        modeloutput_s_pr = model_output[2].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        modeloutput_s_pr = F.normalize(modeloutput_s_pr, dim=1)

        loss_supcon = self.supcon_criterion(modeloutput_z, modeloutput_s_pr=modeloutput_s_pr, modeloutput_f=modeloutput_f,
                                Pool_ag=self.Pool_ag, Pool_sp=self.Pool_sp,
                                opt=self.configs, lmbd=lmbd, modeloutput_z_mix=modeloutput_z_mix)


        detached_code = torch.clone(model_output[1].detach())
        with torch.cuda.amp.autocast(enabled=True):
            linear_output = self.linear_model(detached_code)
            cluster_output = self.cluster_model(detached_code, None, is_direct=False)

            loss, loss_dict, corr_dict = self.criterion(model_input=model_input,
                                                    model_output=model_output,
                                                    linear_output=linear_output,
                                                    cluster_output=cluster_output
                                                    )

            loss = loss + loss_supcon + loss_consistency * self.configs['alpha'] 


        self.scaler.scale(loss).backward()

        if self.freeze_encoder_bn:
            zero_grad_bn(self.net_model)
        if 0 < self.freeze_all_bn <= current_epoch:
            zero_grad_bn(self.net_model)

        self.scaler.unscale_(self.net_optimizer)

        g_norm = nn.utils.clip_grad_norm_(self.net_model.parameters(), self.grad_norm)
        self.scaler.step(self.net_optimizer)

        self.scaler.step(self.linear_probe_optimizer)
        self.scaler.step(self.cluster_probe_optimizer)
        self.scaler.step(self.head_optimizer)

        self.scaler.update()


        loss_dict['loss'] = loss.cpu().item()
        loss_dict['loss_supcon'] = loss_supcon.cpu().item()
        loss_dict['loss_consistency'] = loss_consistency.cpu().item()
        return loss_dict

    @torch.no_grad()
    def sample(self, batch_dict, visualize_all=False):
        assert not self.training
        img: torch.Tensor = batch_dict['images'].to(self.device, non_blocking=True)
        img = (img - self.pixel_mean) / self.pixel_std
        label: torch.Tensor = batch_dict['masks'].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            output = self.net_model(img)

        backbone_feats = output[0] # b c h w
        head_code = output[1].float() # b c h w

        sampled_points, similarities, backbone_similarities = None, None, None
        kmeans_preds, num_kmeans_classes, kmeans_preds_backbone = None, None, None,
        if visualize_all:
            sampled_points, similarities, backbone_similarities = self.sample_point_similarities(code_features=head_code,
                                                                                                 backbone_features=backbone_feats, 
                                                                                                 num_points=10)
            kmeans_preds, _ , num_kmeans_classes = self.self_cluster(head_code, label)
            kmeans_preds_backbone, _ , _ = self.self_cluster(backbone_feats, label)

        head_code = F.interpolate(head_code, label.shape[-2:], mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=True):
            linear_preds = self.linear_model(head_code)
        with torch.cuda.amp.autocast(enabled=True):
            _, cluster_preds = self.cluster_model(head_code, None, is_direct=self.configs["eval"]["is_direct"])

        return  {
            'linear_preds': linear_preds,
            'cluster_preds': cluster_preds,

            'sampled_points': sampled_points,
            'similarities': similarities,
            'backbone_similarities': backbone_similarities,

            'kmeans_preds': kmeans_preds,
            'kmeans_preds_bb': kmeans_preds_backbone,
            'num_kmeans_classes': num_kmeans_classes,
        }

    def optimize_state_dict(self,):
        return {
            'net_optim': self.net_optimizer.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.net_optimizer.load_state_dict(state_dict=state_dict['net_optim'])


    def get_lr_group_dicts(self, ):
        return  {f'lr_net': self.net_optimizer.param_groups[0]["lr"],
                 f'lr_cluster': self.cluster_probe_optimizer.param_groups[0]["lr"],
                 f'lr_linear': self.linear_probe_optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def sample_point_similarities(self, backbone_features, code_features, num_points):
        # b c h w, num_points
        H_P, W_P = code_features.shape[-2:]
        H, W = H_P * self.patch_size, W_P * self.patch_size
        sampled_points = torch.rand(num_points, 2)
        sampled_points[:, 0] = sampled_points[:, 0] * H_P
        sampled_points[:, 1] = sampled_points[:, 1] * W_P
        sampled_points = sampled_points.long()
        sampled_points[:, 0].clamp_(0, H_P-1)
        sampled_points[:, 1].clamp_(0, W_P-1)
        similarities = []
        for point in sampled_points:
            query = code_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(code_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            similarities.append(sim)

        backbone_similarities = []
        for point in sampled_points:
            query = backbone_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(backbone_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            backbone_similarities.append(sim)

        sampled_points = sampled_points * self.patch_size
        return sampled_points, similarities, backbone_similarities


    @torch.no_grad()
    def self_cluster(self, features, gt_masks):
        from models.UN_IMG_SEM.kmeans.kmeans import kmeans
        # b c h w -> b
        _, _, H_P, W_P = features.shape
        assert features.shape[0] == 1
        if self.kmeans_strategy == 'adaptive':
            num_image_classes = len(set(gt_masks.unique().tolist()) - set([-1]))
        else:
            raise ValueError()
        features = features.permute(0, 2,3,1).flatten(0,2) # bhw c
        _, cluster_centers = kmeans(X=features, num_clusters=num_image_classes, device=self.device) # num c
        cluster_logits = torch.einsum('sc,nc->sn', 
                                    F.normalize(features, dim=-1, eps=1e-10),
                                    F.normalize(cluster_centers.to(self.device), dim=-1, eps=1e-10))
        # 把和cluster_center最近的点标注出来
        cluster_logits = rearrange(cluster_logits, '(b h w) n -> b n h w', b=1,h=H_P, w=W_P)
        cluster_logits = F.interpolate(cluster_logits, size=(H_P*self.patch_size, W_P*self.patch_size), mode='bilinear', align_corners=False)
        cluster_ids = cluster_logits.max(dim=1)[1].cpu()
        return cluster_ids, cluster_logits, num_image_classes

@register_model
def hp_agg_cluster(configs, device):
    # 假设只有一个训练数据集
    scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    train_loader_memory, _ = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    batch_size = configs['optim']['batch_size']
    eval_dataset_name = list(configs['data']['evaluate'].keys())[0]
    train_size = configs['data']['train'][train_dataset_name]['mapper']['res']
    eval_size = configs['data']['evaluate'][eval_dataset_name]['mapper']['res']
    model = HP_AGG(configs, num_classes=num_classes, train_loader_memory=train_loader_memory, device=device,
                   batch_size=batch_size,
                   train_size=train_size,
                   eval_size=eval_size)
    model.to(device)
    model.optimize_setup(configs)
    from .make_reference_pool import initialize_reference_pool
    model.Pool_ag, model.Pool_sp = initialize_reference_pool(model.net_model, 
                                                            train_loader_memory, configs, 
                                                            model.feat_dim, device, model.pixel_mean, model.pixel_std)
    model.scaler = scaler
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

