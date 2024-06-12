
import torch
import os

# class HilbertScan(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x: torch.Tensor):
#         # b c h w -> b k c hw
#         if scan_order_by_scale is None:
#             scan_order_by_scale = {
#                     fixed_H//stride: generate_hilbert_curve2D(height=fixed_H // stride, width=fixed_W // stride,
#                                                                device=x.device) for stride in [4, 8, 16, 32]
#             }

#         B, C, H, W = x.shape
#         ctx.shape = (B, C, H, W)

#         hil, hil1_rev, hil2, hil2_rev = scan_order_by_scale[int(H)] 
#         x_hw = x.flatten(2).contiguous() # b c hw
#         x_hil = x_hw.index_select(dim=-1, index=hil) # b c hil
#         x_hil_rever = x_hw.index_select(dim=-1, index=hil1_rev) # b c hil
#         x_hil2 = x_hw.index_select(dim=-1, index=hil2)
#         x_hil2_rever = x_hw.index_select(dim=-1, index=hil2_rev) # b c hil
#         xs = torch.stack([x_hil, x_hil_rever, x_hil2, x_hil2_rever], dim=1) # b k d l
#         return xs
    
#     @staticmethod
#     def backward(ctx, ys: torch.Tensor):
#         # b k c hw -> b c h w
#         B, C, H, W = ctx.shape
#         L = H * W
#         hil, hil1_rev, hil2, hil2_rev = scan_order_by_scale[int(H)]
#         # b k c hil
#         hil_feat, rev_hil_feat, hil2_feat, rev_hil2_feat = ys.unbind(dim=1) # b c Hil  
#         hil = hil[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil1_rev = hil1_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2 = hil2[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2_rev = hil2_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         assert hil_feat.shape == hil.shape
#         assert rev_hil_feat.shape == hil1_rev.shape
#         assert hil2_feat.shape == hil2.shape
#         assert rev_hil2_feat.shape == hil2_rev.shape
#         sum_out = torch.zeros_like(hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil, src=hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil1_rev, src=rev_hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2, src=hil2_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2_rev, src=rev_hil2_feat)  # b c Hil   
        
#         y = y.view(B, -1, H, W)
#         return y


# class HilbertMerge(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ys: torch.Tensor):
#         # b k c h w -> b c hw
#         B, K, D, H, W = ys.shape
#         ctx.shape = (H, W)
#         hil, hil1_rev, hil2, hil2_rev = scan_order_by_scale[int(H)] 
#         hil_feat, rev_hil_feat, hil2_feat, rev_hil2_feat = ys.flatten(3).unbind(dim=1) # b c Hil
#         hil = hil[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil1_rev = hil1_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2 = hil2[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2_rev = hil2_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         assert hil_feat.shape == hil.shape
#         assert rev_hil_feat.shape == hil1_rev.shape
#         assert hil2_feat.shape == hil2.shape
#         assert rev_hil2_feat.shape == hil2_rev.shape
#         sum_out = torch.zeros_like(hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil, src=hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil1_rev, src=rev_hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2, src=hil2_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2_rev, src=rev_hil2_feat)  # b c Hil   
#         y = sum_out
#         return y
    
#     @staticmethod
#     def backward(ctx, x: torch.Tensor):
#         # b c hil -> b k c h w
#         H, W = ctx.shape
#         B, C, L = x.shape
#         hil, hil1_rev, hil2, hil2_rev = scan_order_by_scale[int(H)] 
#         x_hw = x
#         x_hil = x_hw.index_select(dim=-1, index=hil) # b c hil
#         x_hil_rever = x_hw.index_select(dim=-1, index=hil1_rev) # b c hil
#         x_hil2 = x_hw.index_select(dim=-1, index=hil2)
#         x_hil2_rever = x_hw.index_select(dim=-1, index=hil2_rev) # b c hil
#         xs = torch.stack([x_hil, x_hil_rever, x_hil2, x_hil2_rever], dim=1) # b k d l
#         xs = xs.view(B, 4, C, H, W)
#         return xs


# class Hilbert:
#     def __init__(self) -> None:
#         self.scan_order_by_scale = None

#     def scan(self, x: torch.Tensor):
#         # b c h w -> b k c hw
#         if self.scan_order_by_scale is None:
#             print('Using hilbert*************************')
#             self.scan_order_by_scale = {
#                     512//stride: generate_hilbert_curve2D(height=512 // stride, width=512 // stride,
#                                                                device=x.device) for stride in [4, 8, 16, 32]
#             }
#         B, C, H, W = x.shape
#         hil, hil1_rev, hil2, hil2_rev = self.scan_order_by_scale[int(H)] 
#         x_hw = x.flatten(2).contiguous() # b c hw
#         x_hil = x_hw.index_select(dim=-1, index=hil) # b c hil
#         x_hil_rever = x_hw.index_select(dim=-1, index=hil1_rev) # b c hil
#         x_hil2 = x_hw.index_select(dim=-1, index=hil2)
#         x_hil2_rever = x_hw.index_select(dim=-1, index=hil2_rev) # b c hil
#         xs = torch.stack([x_hil, x_hil_rever, x_hil2, x_hil2_rever], dim=1) # b k d l
#         return xs    

#     def merge(self, ys: torch.Tensor):
#         # b k c h w -> b c hw
#         B, K, D, H, W = ys.shape
#         if self.scan_order_by_scale is None:
#             print('Using hilbert*************************')
#             self.scan_order_by_scale = {
#                     512//stride: generate_hilbert_curve2D(height=512 // stride, width=512 // stride,
#                                                                device=ys.device) for stride in [4, 8, 16, 32]
#             }
#         hil, hil1_rev, hil2, hil2_rev = self.scan_order_by_scale[int(H)] 
#         hil_feat, rev_hil_feat, hil2_feat, rev_hil2_feat = ys.flatten(3).unbind(dim=1) # b c Hil
#         hil = hil[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil1_rev = hil1_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2 = hil2[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         hil2_rev = hil2_rev[None, None, :].repeat(B, hil_feat.shape[1], 1)
#         assert hil_feat.shape == hil.shape
#         assert rev_hil_feat.shape == hil1_rev.shape
#         assert hil2_feat.shape == hil2.shape
#         assert rev_hil2_feat.shape == hil2_rev.shape
#         sum_out = torch.zeros_like(hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil, src=hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil1_rev, src=rev_hil_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2, src=hil2_feat)
#         sum_out.scatter_add_(dim=-1, index=hil2_rev, src=rev_hil2_feat)  # b c Hil   
#         y = sum_out
#         return y
    


# def generate_hilbert_curve2D(height, width, device):
#     # 从[0, 0,]横向切入, major=width
#     hil1 = list(gilbert2d(width=width, height=height,)) # x, y 平着走, 
#     # 从[W, H]横向切入
#     hil1_rev = [[width - haosen[0] - 1, height-haosen[1] - 1] for haosen in hil1]

#     # hw展开
#     hil1_ind = torch.tensor(hil1).long().to(device)
#     hil1_ind = hil1_ind[:, 1] * width + hil1_ind[:, 0]

#     hil1_rev_ind = torch.tensor(hil1_rev).long().to(device)
#     hil1_rev_ind = hil1_rev_ind[:, 1] * width + hil1_rev_ind[:, 0]

#     # 从[0,0]纵向切入, major=height
#     hil2_o = list(gilbert2d(width=height, height=width,)) # y, x 竖着走
#     # 从(W, 0)纵向切入
#     hil2 = [[haosen[0], width - haosen[1] - 1 ] for haosen in hil2_o]
#     # 从(0, H)纵向切入
#     hil2_rev = [[height - haosen[0] - 1, haosen[1]] for haosen in hil2_o]

#     # hw展开的ind
#     hil2_ind = torch.tensor(hil2).long().to(device)
#     hil2_ind = hil2_ind[:, 0] * width + hil2_ind[:, 1]

#     hil2_rev_ind = torch.tensor(hil2_rev).long().to(device)
#     hil2_rev_ind = hil2_rev_ind[:, 0] * width + hil2_rev_ind[:, 1]

#     return hil1_ind, hil1_rev_ind, hil2_ind, hil2_rev_ind

#     # hil1 = torch.tensor(hil1).long().to(device)
#     # hil2 = torch.tenosr(hil2).long().to(device)

#     # hil1_hw_ind = hil1[:, 1] * width

#     # hilbert_curve = torch.tensor(hilbert_curve).long().to(device)
#     # sequence_indices = hilbert_curve[:, 1] * width + hilbert_curve[:, 0] # tensor
    
#     # hilbert_curve2 = torch.tensor(hilbert_curve2).long().to(device)
#     # sequence_indices2 = hilbert_curve2[:, 0] * width + hilbert_curve2[:, 1] # tensor

#     # return sequence_indices, sequence_indices2

# def gilbert2d(width, height): # major = width
#     """
#     Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
#     2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
#     of size (width x height).
#     """

#     # if width >= height:
#     yield from generate2d(0, 0, width, 0, 0, height)
#     # else:
#     #     yield from generate2d(0, 0, 0, height, width, 0)

# def sgn(x):
#     return -1 if x < 0 else (1 if x > 0 else 0)



class Hilbert2D:
    def __init__(self,) -> None:
        self.scan_order_by_scale = {}  
        self.cache_file = './pre_hil_2d_cache.pt'
        # 4, 每个点的主方向都是H/W; 形成一个2D环
        if os.path.exists(self.cache_file):
            pre_gen = torch.load(self.cache_file)
            self.scan_order_by_scale = pre_gen

    def scan(self, x: torch.Tensor): 
        B, C, H, W, device = *x.shape, x.device
        # b c h w -> b k c hw
        scan_key = f'{H}_{W}'
        if scan_key not in self.scan_order_by_scale:
            self.scan_order_by_scale[scan_key] = self.generate_curves(H=H, W=W)
            torch.save(self.scan_order_by_scale, self.cache_file)
        
        # 4 L, hw对应Index
        curve_idxs = self.scan_order_by_scale[scan_key].to(device)
        
        x = x.flatten(2).contiguous() # b c hw
        
        # list[b c L], 8
        x_8 = [x.index_select(-1, index=haosen).contiguous()  for haosen in curve_idxs]
        xs = torch.stack(x_8, dim=1) # b 4 c L
        return xs    

    def merge(self, ys: torch.Tensor):
        # b k c h w -> b c hw
        B, K, C, H, W, device = *ys.shape, ys.device
        scan_key = f'{H}_{W}'
        curve_idxs = self.scan_order_by_scale[scan_key].to(device) # 8 L, int
        curve_idxs = curve_idxs[None, None, :, :].repeat(B, C, 1, 1) # b c 8 L
        # b k c h w -> b k c hw -> list[b c hw], k
        feat1, feat2, feat3, feat4 = ys.flatten(3).unbind(dim=1) # b c hw
        hil1, hil2, hil3, hil4 = curve_idxs.unbind(2) # list[b c L]
        assert feat1.shape == hil1.shape
        assert feat2.shape == hil2.shape
        assert feat3.shape == hil3.shape
        assert feat4.shape == hil4.shape
        
        sum_out = torch.zeros_like(feat1)
        sum_out.scatter_add_(dim=-1, index=hil1, src=feat1)
        sum_out.scatter_add_(dim=-1, index=hil2, src=feat2)
        sum_out.scatter_add_(dim=-1, index=hil3, src=feat3)
        sum_out.scatter_add_(dim=-1, index=hil4, src=feat4)
        y = sum_out # b c L
        return y

    def generate_curves(self, H, W):
        # x index W; y index H, z index T
        curve_1 = torch.tensor(list(generate2d(0, 0,  # 从0, 0开始, 横向
                                               ax=W, ay=0, bx=0, by=H)))
        curve_2 = torch.tensor(list(generate2d(W-1, 0,  # 从 W,0 开始, 纵向
                                               ax=0, ay=H, bx=-W, by=0)))
        curve_3 = torch.tensor(list(generate2d(W-1, H-1,  # 从 W, H 开始, 横向
                                               ax=-W, ay=0, bx=0, by=-H)))
        curve_4 = torch.tensor(list(generate2d(0, H-1,  # 从0, H开始, 纵向
                                               ax=0, ay=-H, bx=W, by=0, )))
  
        curves = torch.stack([curve_1, curve_2, curve_3, curve_4], dim=0).long()  # 4 wh 2 
        
        # hw flatten
        #  h * W + w
        curve_idxs = curves[:, :, 1] * W + curves[:, :, 0]

        return curve_idxs # 4 L

def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))
        


class Hilbert3D:
    def __init__(self, version='spatial_first') -> None:
        self.scan_order_by_scale = {}  
        if version == 'spatial_first':
            self.cache_file = './pre_hil_sf_cache.pt'
            # 8, 每个点的主方向都是H/W; 每4个点(同一frame)形成一个2D环
            # "t hw"的方式
            self.generate_curves = self.generate_curves_spatial_first

        elif version == 'time_first':
            self.cache_file = './pre_hil_tf_cache.pt'
            # 8, 每个点的主方向都是T/-T, 每2个点(同一pixel)形成一个1D环
            # "hw t"的方式
            self.generate_curves = self.generate_curves_time_first
        else:
            raise ValueError()  

        if os.path.exists(self.cache_file):
            pre_gen = torch.load(self.cache_file)
            self.scan_order_by_scale = pre_gen

    def scan(self, x: torch.Tensor): 
        B, C, T, H, W, device = *x.shape, x.device
        # b c t h w -> b k c thw
        scan_key = f'{T}_{H}_{W}'
        if scan_key not in self.scan_order_by_scale:
            saved_sob = torch.load(self.cache_file)
            if scan_key in saved_sob.keys():
                self.scan_order_by_scale[scan_key] = saved_sob[scan_key] 
            else:
                self.scan_order_by_scale[scan_key] = self.generate_curves(H=H, W=W, T=T)
                saved_sob[scan_key] = self.scan_order_by_scale[scan_key]
                torch.save(saved_sob, self.cache_file)
        
        # 8 L, thw对应的index
        curve_idxs = self.scan_order_by_scale[scan_key].to(device)
        
        # thw flatten
        x = x.flatten(2).contiguous() # b c thw
        
        # list[b c L], 8
        x_8 = [x.index_select(-1, index=haosen).contiguous()  for haosen in curve_idxs]
        xs = torch.stack(x_8, dim=1) # b 8 c L
        return xs    

    def merge(self, ys: torch.Tensor):
        # b k c h w -> b c hw
        B, K, C, T, H, W, device = *ys.shape, ys.device
        scan_key = f'{T}_{H}_{W}'
        
        curve_idxs = self.scan_order_by_scale[scan_key].to(device) # 8 L, int
        curve_idxs = curve_idxs[None, None, :, :].repeat(B, C, 1, 1) # b c 8 L
        # b k c t h w -> b k c thw -> list[b c thw], k
        feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8 = ys.flatten(3).unbind(dim=1) # b c thw
        hil1, hil2, hil3, hil4, hil5, hil6, hil7, hil8 = curve_idxs.unbind(2) # list[b c L]
        assert feat1.shape == hil1.shape
        assert feat2.shape == hil2.shape
        assert feat3.shape == hil3.shape
        assert feat4.shape == hil4.shape
        assert feat5.shape == hil5.shape
        assert feat6.shape == hil6.shape
        assert feat7.shape == hil7.shape
        assert feat8.shape == hil8.shape
        
        sum_out = torch.zeros_like(feat1)
        sum_out.scatter_add_(dim=-1, index=hil1, src=feat1)
        sum_out.scatter_add_(dim=-1, index=hil2, src=feat2)
        sum_out.scatter_add_(dim=-1, index=hil3, src=feat3)
        sum_out.scatter_add_(dim=-1, index=hil4, src=feat4)
        sum_out.scatter_add_(dim=-1, index=hil5, src=feat5)
        sum_out.scatter_add_(dim=-1, index=hil6, src=feat6)
        sum_out.scatter_add_(dim=-1, index=hil7, src=feat7)
        sum_out.scatter_add_(dim=-1, index=hil8, src=feat8)
        y = sum_out # b c L
        return y

    def generate_curves_spatial_first(self, H, W, T):
        # x index W; y index H, z index T
        curve_1 = torch.tensor(list(generate3d(0, 0, 0, # 0,0,0 主方向是横
                                  ax=W, ay=0, az=0,
                                  bx=0, by=H, bz=0,
                                  cx=0, cy=0, cz=T
                                  )))
        curve_2 = torch.tensor(list(generate3d(W-1, 0, 0, # W,0,0 主方向是纵
                                  ax=0, ay=H, az=0,
                                  bx=-W, by=0, bz=0,
                                  cx=0, cy=0, cz=T
                                  )))
        curve_3 = torch.tensor(list(generate3d(W-1, H-1, 0, # W, H, 0 主方向是反横
                                  ax=-W, ay=0, az=0,
                                  bx=0, by=-H, bz=0,
                                  cx=0, cy=0, cz=T
                                  )))
        curve_4 = torch.tensor(list(generate3d(0, H-1, 0, # 0, H, 0 主方向是反纵
                                  ax=0, ay=-H, az=0,
                                  bx=W, by=0, bz=0,
                                  cx=0, cy=0, cz=T
                                  )))
        curve_5 = torch.tensor(list(generate3d(0, 0, T-1,
                                  ax=W, ay=0, az=0,
                                  bx=0, by=H, bz=0,
                                  cx=0, cy=0, cz=-T
                                  )))
        curve_6 = torch.tensor(list(generate3d(W-1, 0, T-1,
                                  ax=0, ay=H, az=0,
                                  bx=-W, by=0, bz=0,
                                  cx=0, cy=0, cz=-T
                                  )))
        curve_7 = torch.tensor(list(generate3d(W-1, H-1, T-1, 
                                  ax=-W, ay=0, az=0,
                                  bx=0, by=-H, bz=0,
                                  cx=0, cy=0, cz=-T
                                  )))
        curve_8 = torch.tensor(list(generate3d(0, H-1, T-1, 
                                  ax=0, ay=-H, az=0,
                                  bx=W, by=0, bz=0,
                                  cx=0, cy=0, cz=-T
                                  ))) 
        curves = torch.stack([curve_1, curve_2, curve_3, curve_4, curve_5, curve_6, curve_7, curve_8], dim=0).long()  # 8 wht 3 
        
        # thw flatten
        # t * HW + h * W + w
        curve_idxs = curves[:, :, 2] * H * W + curves[:, :, 1] * W + curves[:, :, 0]

        # hwt flatten
        # (h * W + w) * T + t
        # curve_idxs = (curves[:, :, 1] * W + curves[:, :, 0]) * T + curves[:, :, 2]
        
        return curve_idxs # 8 L, int


    def generate_curves_time_first(self, H, W, T):
        curve_1 = torch.tensor(list(generate3d(0, 0, 0, # 0,0,0 次方向是横
                                            ax=0, ay=0, az=T,
                                            bx=W, by=0, bz=0,
                                            cx=0, cy=H, cz=0
                                                )))
        curve_2 = torch.tensor(list(generate3d(W-1, 0, 0, # W,0,0 次方向是纵
                                                ax=0, ay=0, az=T,
                                                bx=0, by=H, bz=0,
                                                cx=-W, cy=0, cz=0
                                                )))
        curve_3 = torch.tensor(list(generate3d(W-1, H-1, 0, # w h 0 次方向是-横
                                                ax=0, ay=0, az=T,
                                                bx=-W, by=0, bz=0,
                                                cx=0, cy=-H, cz=0
                                                )))
        curve_4 = torch.tensor(list(generate3d(0, H-1, 0,  # 0,h ,0 次方向是-纵
                                                ax=0, ay=0, az=T,
                                                bx=0, by=-H, bz=0,
                                                cx=W, cy=0, cz=0
                                            )))
        curve_5 = torch.tensor(list(generate3d(0, 0, T-1,
                                                ax=0, ay=0, az=-T,
                                                bx=W, by=0, bz=0,
                                                cx=0, cy=H, cz=0
                                                )))
        curve_6 = torch.tensor(list(generate3d(W-1, 0, T-1,
                                                ax=0, ay=0, az=-T,
                                                bx=0, by=H, bz=0,
                                                cx=-W, cy=0, cz=0
                                                )))
        curve_7 = torch.tensor(list(generate3d(W-1, H-1, T-1, 
                                                ax=0, ay=0, az=-T,
                                               bx=-W, by=0, bz=0,
                                                cx=0, cy=-H, cz=0
                                                )))
        curve_8 = torch.tensor(list(generate3d(0, H-1, T-1, 
                                                ax=0, ay=0, az=-T,
                                                bx=0, by=-H, bz=0,
                                                cx=W, cy=0, cz=0
                                                ))) # wht 3
        curves = torch.stack([curve_1,  curve_2, curve_3, curve_4, curve_5, curve_6, curve_7, curve_8], dim=0).long()  # 8 wht 3 ???wht
        
        # thw flatten
        # t * HW + h * W + w
        curve_idxs = curves[:, :, 2] * H * W + curves[:, :, 1] * W + curves[:, :, 0]

        # hwt flatten
        # (h * W + w) * T + t
        # curve_idxs = (curves[:, :, 1] * W + curves[:, :, 0]) * T + curves[:, :, 2]
        
        return curve_idxs # 8 L, thw flatten对应的序列


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

# T H W 
def generate3d(x, y, z,
               ax, ay, az,
               bx, by, bz,
               cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az)) # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz)) # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz)) # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        for i in range(0, w):
            yield(x, y, z)
            (x, y, z) = (x + dax, y + day, z + daz)
        return

    if w == 1 and d == 1:
        for i in range(0, h):
            yield(x, y, z)
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return

    if w == 1 and h == 1:
        for i in range(0, d):
            yield(x, y, z)
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
       yield from generate3d(x, y, z,
                             ax2, ay2, az2,
                             bx, by, bz,
                             cx, cy, cz)

       yield from generate3d(x+ax2, y+ay2, z+az2,
                             ax-ax2, ay-ay2, az-az2,
                             bx, by, bz,
                             cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
       yield from generate3d(x, y, z,
                             bx2, by2, bz2,
                             cx, cy, cz,
                             ax2, ay2, az2)

       yield from generate3d(x+bx2, y+by2, z+bz2,
                             ax, ay, az,
                             bx-bx2, by-by2, bz-bz2,
                             cx, cy, cz)

       yield from generate3d(x+(ax-dax)+(bx2-dbx),
                             y+(ay-day)+(by2-dby),
                             z+(az-daz)+(bz2-dbz),
                             -bx2, -by2, -bz2,
                             cx, cy, cz,
                             -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
       yield from generate3d(x, y, z,
                             cx2, cy2, cz2,
                             ax2, ay2, az2,
                             bx, by, bz)

       yield from generate3d(x+cx2, y+cy2, z+cz2,
                             ax, ay, az,
                             bx, by, bz,
                             cx-cx2, cy-cy2, cz-cz2)

       yield from generate3d(x+(ax-dax)+(cx2-dcx),
                             y+(ay-day)+(cy2-dcy),
                             z+(az-daz)+(cz2-dcz),
                             -cx2, -cy2, -cz2,
                             -(ax-ax2), -(ay-ay2), -(az-az2),
                             bx, by, bz)

    # regular case, split in all w/h/d
    else:
       yield from generate3d(x, y, z,
                             bx2, by2, bz2,
                             cx2, cy2, cz2,
                             ax2, ay2, az2)

       yield from generate3d(x+bx2, y+by2, z+bz2,
                             cx, cy, cz,
                             ax2, ay2, az2,
                             bx-bx2, by-by2, bz-bz2)

       yield from generate3d(x+(bx2-dbx)+(cx-dcx),
                             y+(by2-dby)+(cy-dcy),
                             z+(bz2-dbz)+(cz-dcz),
                             ax, ay, az,
                             -bx2, -by2, -bz2,
                             -(cx-cx2), -(cy-cy2), -(cz-cz2))

       yield from generate3d(x+(ax-dax)+bx2+(cx-dcx),
                             y+(ay-day)+by2+(cy-dcy),
                             z+(az-daz)+bz2+(cz-dcz),
                             -cx, -cy, -cz,
                             -(ax-ax2), -(ay-ay2), -(az-az2),
                             bx-bx2, by-by2, bz-bz2)

       yield from generate3d(x+(ax-dax)+(bx2-dbx),
                             y+(ay-day)+(by2-dby),
                             z+(az-daz)+(bz2-dbz),
                             -bx2, -by2, -bz2,
                             cx2, cy2, cz2,
                             -(ax-ax2), -(ay-ay2), -(az-az2))

