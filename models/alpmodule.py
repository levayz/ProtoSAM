"""
ALPModule
"""
import torch
import time
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
# for unit test from spatial_similarity_module import NONLocalBlock2D, LayerNorm

def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
    x_norm = torch.norm(x, p = p, dim = dim) # .detach()
    x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
    x = x.div(x_norm.unsqueeze(1).expand_as(x))
    return x


class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, embed_dim=768, use_attention=False, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map

        """
        super(MultiProtoAsConv, self).__init__()
        self.feature_hw = feature_hw
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ]
        self.kernel_size = kernel_size
        print(f"MultiProtoAsConv: kernel_size: {kernel_size}")
        self.avg_pool_op = nn.AvgPool2d( kernel_size  )
        
        if use_attention:
            self.proto_fg_attnetion = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12 if embed_dim == 768 else 8, batch_first=True)
            self.proto_bg_attnetion = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12 if embed_dim == 768 else 8, batch_first=True)
            self.fg_mask_projection = nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.bg_mask_projection = nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )
            
    def get_prediction_from_prototypes(self, prototypes, query, mode, vis_sim=False ):
        if mode == 'mask':
            pred_mask = F.cosine_similarity(query, prototypes[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]
            # incase there are more than one prototypes in the same location, take the max
            pred_mask = pred_mask.max(dim = 0)[0].unsqueeze(0)
            vis_dict = {'proto_assign': pred_mask} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]
            
        elif mode == 'gridconv':
            dists = F.conv2d(query, prototypes[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float().detach()

            vis_dict = {'proto_assign': debug_assign} # things to visualize

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()
            return pred_grid, [debug_assign], vis_dict
        
        elif mode == 'gridconv+':
            dists = F.conv2d(query, prototypes[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            # raw_local_sims = dists.det ach()

            debug_assign = dists.argmax(dim = 1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'mask', 'gridconv', or 'gridconv+'.")
        
        
    def get_prototypes(self, sup_x, sup_y, mode, val_wsize, thresh, isval = False):
        if mode == 'mask':
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            pro_n = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pro_n = proto
            proto_grid = sup_y.clone().detach() # a single prototype for the whole image
            resized_proto_grid = proto_grid
            non_zero = torch.nonzero(proto_grid)

        elif mode == 'gridconv':
            nch = sup_x.shape[1]

            sup_nshot = sup_x.shape[0]
            # if len(sup_x.shape) > 4:
            #     sup_x = sup_x.squeeze()
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            
            # get a grid of prototypes
            proto_grid = sup_y_g.clone().detach()
            proto_grid[proto_grid < thresh] = 0
            # interpolate the grid to the original size
            non_zero = torch.nonzero(proto_grid)
            
            resized_proto_grid = torch.zeros([1, 1, proto_grid.shape[2]*val_wsize, proto_grid.shape[3]*val_wsize])
            for index in non_zero:
                resized_proto_grid[0, 0, index[2]*val_wsize:index[2]*val_wsize + val_wsize, index[3]*val_wsize:index[3]*val_wsize + 2] = proto_grid[0, 0, index[2], index[3]]
            
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc
            pro_n = safe_norm(protos)
            
        elif mode == 'gridconv+':
            nch = sup_x.shape[1]
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )
            sup_nshot = sup_x.shape[0]
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            
            # get a grid of prototypes
            proto_grid = sup_y_g.clone().detach()
            proto_grid[proto_grid < thresh] = 0
            non_zero = torch.nonzero(proto_grid)
            for i, idx in enumerate(non_zero):
                proto_grid[0, idx[1], idx[2], idx[3]] = i + 1
            resized_proto_grid = torch.zeros([1, 1, proto_grid.shape[2]*val_wsize, proto_grid.shape[3]*val_wsize])
            for index in non_zero:
                resized_proto_grid[0, 0, index[2]*val_wsize:index[2]*val_wsize + val_wsize, index[3]*val_wsize:index[3]*val_wsize + 2] = proto_grid[0, 0, index[2], index[3]]
            
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            protos = n_sup_x[sup_y_g > thresh, :]

            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            pro_n = safe_norm(torch.cat( [protos, glb_proto], dim = 0 ))
        return pro_n, resized_proto_grid, non_zero

    def forward(self, qry, sup_x, sup_y, mode, thresh, isval = False, val_wsize = None, vis_sim = False, get_prototypes=False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """

        qry = qry.squeeze(1) # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1) # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0) # [nshot, 1, h, w]

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
        if val_wsize is None:
            val_wsize = self.avg_pool_op.kernel_size
            if isinstance(val_wsize, (tuple, list)):
                val_wsize = val_wsize[0] 
        sup_y = sup_y.reshape(sup_x.shape[0], 1, sup_x.shape[-2], sup_x.shape[-1]) 
        pro_n, proto_grid, proto_indices = self.get_prototypes(sup_x, sup_y, mode, val_wsize, thresh, isval) 
        if 0 in pro_n.shape:
            print("failed to find prototypes")
        qry_n = qry if mode == 'mask' else safe_norm(qry)
        pred_grid, debug_assign, vis_dict = self.get_prediction_from_prototypes(pro_n, qry_n, mode, vis_sim=vis_sim) 

        return pred_grid, debug_assign, vis_dict, proto_grid

