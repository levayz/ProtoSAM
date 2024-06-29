"""
ALPNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .alpmodule import MultiProtoAsConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from util.consts import DEFAULT_FEATURE_SIZE
from util.lora import inject_trainable_lora
# from util.utils import load_config_from_url, plot_dinov2_fts
import math

# Specify a local path to the repository (or use installed package instead)
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
# FG_PROT_MODE = 'mask'
# using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)
BG_PROT_MODE = 'gridconv'

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95


class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """

    def __init__(self, image_size, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self.config = cfg or {
            'align': False, 'debug': False}
        self.get_encoder()
        self.get_cls()
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path), strict=True)
            print(
                f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_encoder(self):
        self.config['feature_hw'] = [DEFAULT_FEATURE_SIZE,
                                     DEFAULT_FEATURE_SIZE]  # default feature map size
        if self.config['which_model'] == 'dlfcn_res101' or self.config['which_model'] == 'default':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
            self.config['feature_hw'] = [
                math.ceil(self.image_size/8), math.ceil(self.image_size/8)]
        elif self.config['which_model'] == 'dinov2_l14':
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitl14')
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        elif self.config['which_model'] == 'dinov2_l14_reg':
            try:
                self.encoder = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vitl14_reg')
            except RuntimeError as e:
                self.encoder = torch.hub.load(
                    'facebookresearch/dino', 'dinov2_vitl14_reg', force_reload=True)
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        elif self.config['which_model'] == 'dinov2_b14':
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14')
            self.config['feature_hw'] = [max(
                self.image_size//14, DEFAULT_FEATURE_SIZE), max(self.image_size//14, DEFAULT_FEATURE_SIZE)]
        else:
            raise NotImplementedError(
                f'Backbone network {self.config["which_model"]} not implemented')

        if self.config['lora'] > 0:
            self.encoder.requires_grad_(False)
            print(f'Injecting LoRA with rank:{self.config["lora"]}')
            encoder_lora_params = inject_trainable_lora(
                self.encoder, r=self.config['lora'])

    def get_features(self, imgs_concat):
        if self.config['which_model'] == 'dlfcn_res101':
            img_fts = self.encoder(imgs_concat, low_level=False)
        elif 'dino' in self.config['which_model']:
            # resize imgs_concat to the closest size that is divisble by 14
            imgs_concat = F.interpolate(imgs_concat, size=(
                self.image_size // 14 * 14, self.image_size // 14 * 14), mode='bilinear')
            dino_fts = self.encoder.forward_features(imgs_concat)
            img_fts = dino_fts["x_norm_patchtokens"]  # B, HW, C
            img_fts = img_fts.permute(0, 2, 1)  # B, C, HW
            C, HW = img_fts.shape[-2:]
            img_fts = img_fts.view(-1, C, int(HW**0.5),
                                   int(HW**0.5))  # B, C, H, W
            if HW < DEFAULT_FEATURE_SIZE ** 2:
                img_fts = F.interpolate(img_fts, size=(
                    DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode='bilinear')  # this is if h,w < (32,32)
        else:
            raise NotImplementedError(
                f'Backbone network {self.config["which_model"]} not implemented')
        
        return img_fts

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]

        if self.config['cls_name'] == 'grid_proto':
            embed_dim = 256
            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024
            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_hw, proto_hw], feature_hw=self.config["feature_hw"], embed_dim=embed_dim)  # when treating it as ordinary prototype
            print(f"cls unit feature hw: {self.cls_unit.feature_hw}")
        else:
            raise NotImplementedError(
                f'Classifier {self.config["cls_name"]} not implemented')

    def forward_resolutions(self, resolutions, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None):
        predictions = []
        for res in resolutions:
            supp_imgs_resized = [[F.interpolate(supp_img[0], size=(
                res, res), mode='bilinear') for supp_img in supp_imgs]] if supp_imgs[0][0].shape[-1] != res else supp_imgs
            fore_mask_resized = [[F.interpolate(fore_mask_way[0].unsqueeze(0), size=(res, res), mode='bilinear')[
                0] for fore_mask_way in fore_mask]] if fore_mask[0][0].shape[-1] != res else fore_mask
            back_mask_resized = [[F.interpolate(back_mask_way[0].unsqueeze(0), size=(res, res), mode='bilinear')[
                0] for back_mask_way in back_mask]] if back_mask[0][0].shape[-1] != res else back_mask
            qry_imgs_resized = [F.interpolate(qry_img, size=(res, res), mode='bilinear')
                                for qry_img in qry_imgs] if qry_imgs[0][0].shape[-1] != res else qry_imgs

            pred = self.forward(supp_imgs_resized, fore_mask_resized, back_mask_resized,
                                qry_imgs_resized, isval, val_wsize, show_viz, supp_fts)[0]
            predictions.append(pred)

    def resize_inputs_to_image_size(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        supp_imgs = [[F.interpolate(supp_img, size=(
            self.image_size, self.image_size), mode='bilinear') for supp_img in supp_imgs_way] for supp_imgs_way in supp_imgs]
        fore_mask = [[F.interpolate(fore_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear')[
            0] for fore_mask_way in fore_mask]] if fore_mask[0][0].shape[-1] != self.image_size else fore_mask
        back_mask = [[F.interpolate(back_mask_way[0].unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear')[
            0] for back_mask_way in back_mask]] if back_mask[0][0].shape[-1] != self.image_size else back_mask
        qry_imgs = [F.interpolate(qry_img, size=(self.image_size, self.image_size), mode='bilinear')
                    for qry_img in qry_imgs] if qry_imgs[0][0].shape[-1] != self.image_size else qry_imgs
        return supp_imgs, fore_mask, back_mask, qry_imgs

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False, supp_fts=None):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        # supp_imgs, fore_mask, back_mask, qry_imgs = self.resize_inputs_to_image_size(
        #     supp_imgs, fore_mask, back_mask, qry_imgs)
        
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        # NOTE: actual shot in support goes in batch dimension
        assert n_ways == 1, "Multi-shot has not been implemented yet"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        if self.config["cls_name"] == 'grid_proto_3d':
            img_size = supp_imgs[0][0].shape[-3:]
        qry_bsize = qry_imgs[0].shape[0]

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.get_features(imgs_concat)
        if len(img_fts.shape) == 5:  # for 3D
            fts_size = img_fts.shape[-3:]
        else:
            fts_size = img_fts.shape[-2:]
        if supp_fts is None:
            supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
                n_ways, n_shots, sup_bsize, -1, *fts_size)  # wa x sh x b x c x h' x w'
            qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
                n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        else:
            # N x B x C x H' x W'
            qry_fts = img_fts.view(n_queries, qry_bsize, -1, *fts_size)

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = []  # the buffer for visualization

        for epi in range(1):  # batch dimension, fixed to 1
            fg_masks = []  # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            if len(fts_size) == 3:  # TODO make more generic
                res_fg_msk = torch.stack([F.interpolate(fore_mask[0][0].unsqueeze(
                    0), size=fts_size, mode='nearest')], dim=0)  # [nway, ns, nb, nd', nh', nw'])
                res_bg_msk = torch.stack([F.interpolate(back_mask[0][0].unsqueeze(
                    0), size=fts_size, mode='nearest')], dim=0)  # [nway, ns, nb, nd', nh', nw'])
            else:
                res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='nearest')
                                         for fore_mask_w in fore_mask], dim=0)  # [nway, ns, nb, nh', nw']
                res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='nearest')
                                         for back_mask_w in back_mask], dim=0)  # [nway, ns, nb, nh', nw']

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []
            bg_mode = BG_PROT_MODE

            _raw_score, _, aux_attr, _ = self.cls_unit(
                qry_fts, supp_fts, res_bg_msk, mode=bg_mode, thresh=BG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            
            for way, _msks in enumerate(res_fg_msk):
                raw_scores = []
                for i, _msk in enumerate(_msks):
                    _msk = _msk.unsqueeze(0)
                    supp_ft = supp_fts[:, i].unsqueeze(0)
                    if self.config["cls_name"] == 'grid_proto_3d':  # 3D
                        k_size = self.cls_unit.kernel_size
                        fg_mode = FG_PROT_MODE if F.avg_pool3d(_msk, k_size).max(
                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'  # TODO figure out kernel size
                    else:
                        k_size = self.cls_unit.kernel_size
                        fg_mode = FG_PROT_MODE if F.avg_pool2d(_msk, k_size).max(
                        ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                        # TODO figure out kernel size
                    _raw_score, _, aux_attr, proto_grid = self.cls_unit(qry_fts, supp_ft, _msk.unsqueeze(
                        0), mode=fg_mode, thresh=FG_THRESH, isval=isval, val_wsize=val_wsize, vis_sim=show_viz)
                    raw_scores.append(_raw_score)

                # create a score where each feature is the max of the raw_score
                _raw_score = torch.stack(raw_scores, dim=1).max(dim=1)[
                    0] 
                scores.append(_raw_score)
                assign_maps.append(aux_attr['proto_assign'])
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])
            # print(f"Time for fg: {time.time() - start_time}")
            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            interpolate_mode = 'bilinear'
            outputs.append(F.interpolate(
                pred, size=img_size, mode=interpolate_mode))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        grid_shape = output.shape[2:]
        if self.config["cls_name"] == 'grid_proto_3d':
            grid_shape = output.shape[2:]
        output = output.view(-1, *grid_shape)
        assign_maps = torch.stack(assign_maps, dim=1) if show_viz else None
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps, proto_grid, supp_fts, qry_fts


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        # added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(
            2)  # added to nway(1) and nb(1)
        # end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]
                img_fts = supp_fts[way: way + 1, shot: shot + 1]
                size = img_fts.shape[-2:]
                mode = 'bilinear'
                if self.config["cls_name"] == 'grid_proto_3d':
                    size = img_fts.shape[-3:]
                    mode = 'trilinear'
                qry_pred_fg_msk = F.interpolate(
                    binary_masks[way + 1].float(), size=size, mode=mode)  # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(
                    binary_masks[0].float(), size=size, mode=mode)  # 1, n, h ,w
                scores = []

                bg_mode = BG_PROT_MODE
                _raw_score_bg, _, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_bg_msk.unsqueeze(-3), mode=bg_mode, thresh=BG_THRESH)

                scores.append(_raw_score_bg)
                if self.config["cls_name"] == 'grid_proto_3d':
                    fg_mode = FG_PROT_MODE if F.avg_pool3d(qry_pred_fg_msk, 4).max(
                    ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                else:
                    fg_mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max(
                    ) >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask'
                _raw_score_fg, _, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts, sup_y=qry_pred_fg_msk.unsqueeze(2), mode=fg_mode, thresh=FG_THRESH)
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                size = fore_mask.shape[-2:]
                if self.config["cls_name"] == 'grid_proto_3d':
                    size = fore_mask.shape[-3:]
                supp_pred = F.interpolate(supp_pred, size=size, mode=mode)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append(F.cross_entropy(
                    supp_pred.float(), supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum(torch.stack(loss))

    def dino_cls_loss(self, teacher_cls_tokens, student_cls_tokens):
        cls_loss_weight = 0.1
        student_temp = 1
        teacher_cls_tokens = self.sinkhorn_knopp_teacher(teacher_cls_tokens)
        lsm = F.log_softmax(student_cls_tokens / student_temp, dim=-1)
        cls_loss = torch.sum(teacher_cls_tokens * lsm, dim=-1)

        return -cls_loss.mean() * cls_loss_weight

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp=1, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(teacher_output / teacher_temp).t()
        # B = Q.shape[1] * world_size # number of samples to assign
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def dino_patch_loss(self, features, masked_features, masks):
        # for both supp and query features perform the patch wise loss
        loss = 0.0
        weight = 0.1
        B = features.shape[0]
        for (f, mf, mask) in zip(features, masked_features, masks):
            # TODO sinkhorn knopp center features
            f = f[mask]
            f = self.sinkhorn_knopp_teacher(f)
            mf = mf[mask]
            loss += torch.sum(f * F.log_softmax(mf / 1,
                              dim=-1), dim=-1) / mask.sum()

        return -loss.sum() * weight / B
