import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.ProtoSAM import ModelWrapper
from segment_anything import sam_model_registry
from util.utils import rotate_tensor_no_crop, reverse_tensor, need_softmax, get_confidence_from_logits, get_connected_components, cca, plot_connected_components

class ProtoMedSAM(nn.Module):
    def __init__(self, image_size, coarse_segmentation_model:ModelWrapper, sam_pretrained_path="pretrained_model/medsam_vit_b.pth", debug=False, use_cca=False,  coarse_pred_only=False):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.coarse_segmentation_model = coarse_segmentation_model
        self.get_sam(sam_pretrained_path)
        self.coarse_pred_only = coarse_pred_only
        self.debug = debug
        self.use_cca = use_cca
        
    
    def get_sam(self, checkpoint_path):
        model_type="vit_b" # TODO make generic?
        if 'vit_h' in checkpoint_path:
            model_type = "vit_h"
        self.medsam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval()

        
    torch.no_grad()
    def medsam_inference(self, img_embed, box_1024, H, W, query_label=None):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.medsam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, conf = self.medsam.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.medsam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True if query_label is not None else False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu()  # (256, 256)
        
        low_res_pred = low_res_pred.numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        
        if query_label is not None:
            medsam_seg = self.get_best_mask(medsam_seg, query_label)[None, :]
        
        return medsam_seg, conf.cpu().detach().numpy()
    
    def get_iou(self, pred, label):
        """
        pred np array shape h,w type uint8
        label np array shpae h,w type uiint8
        """
        tp = np.logical_and(pred, label).sum()
        fp = np.logical_and(pred, 1-label).sum()
        fn = np.logical_and(1-pred, label).sum()
        iou = tp / (tp + fp + fn)
        return iou
    
    def get_best_mask(self, masks, labels):
        """
        masks np shape ( B, h, w)
        labels torch shape (1, H, W)
        """
        np_labels = labels[0].clone().detach().cpu().numpy()
        best_iou, best_mask = 0, None
        for mask in masks:
            iou = self.get_iou(mask, np_labels)
            if iou > best_iou:
                best_iou = iou
                best_mask = mask
                
        return best_mask
    
    def get_bbox(self, pred):
        """
        pred is tensor of shape (H,W) - 1 is fg, 0 is bg.
        return bbox of pred s.t np.array([xmin, y_min, xmax, ymax])
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if pred.max() == 0:
            return None
        indices = torch.nonzero(pred)
        ymin, xmin = indices.min(dim=0)[0]
        ymax, xmax = indices.max(dim=0)[0]
        return np.array([xmin, ymin, xmax, ymax])
            
    
    def get_bbox_per_cc(self, conn_components):
        """
        conn_components: output of cca function
        return list of bboxes per connected component, each bbox is a list of 2d points
        """
        bboxes = []
        for i in range(1, conn_components[0]):
            # get the indices of the foreground points
            pred = torch.tensor(conn_components[1] == i, dtype=torch.uint8)
            bboxes.append(self.get_bbox(pred))

        bboxes = np.array(bboxes)
        return bboxes

    def forward(self, query_image, coarse_model_input, degrees_rotate=0):
        """
        query_image: 3d tensor of shape (1, 3, H, W)
        images should be normalized with mean and std but not to [0, 1]?
        """
        original_size = query_image.shape[-2]
        # rotate query_image by degrees_rotate
        rotated_img, (rot_h, rot_w) = rotate_tensor_no_crop(query_image, degrees_rotate)
        # print(f"rotating query image took {time.time() - start_time} seconds")
        coarse_model_input.set_query_images(rotated_img)
        output_logits_rot = self.coarse_segmentation_model(coarse_model_input)
        # print(f"ALPNet took {time.time() - start_time} seconds")
       
        if degrees_rotate != 0:
            output_logits = reverse_tensor(output_logits_rot, rot_h, rot_w, -degrees_rotate)
            # print(f"reversing rotated output_logits took {time.time() - start_time} seconds")
        else:
            output_logits = output_logits_rot
        
        # check if softmax is needed 
        # output_p = output_logits.softmax(dim=1)
        output_p = output_logits
        pred = output_logits.argmax(dim=1)[0]
        if self.debug:
            _pred = np.array(output_logits.argmax(dim=1)[0].detach().cpu())
            plt.subplot(132)
            plt.imshow(query_image[0,0].detach().cpu())
            plt.imshow(_pred, alpha=0.5)
            plt.subplot(131)
            # plot heatmap of prob of being fg
            plt.imshow(output_p[0, 1].detach().cpu())
            # plot rotated query image and rotated pred
            output_p_rot = output_logits_rot.softmax(dim=1)
            _pred_rot = np.array(output_p_rot.argmax(dim=1)[0].detach().cpu())
            _pred_rot = F.interpolate(torch.tensor(_pred_rot).unsqueeze(0).unsqueeze(0).float(), size=original_size, mode='nearest')[0][0]
            plt.subplot(133)
            plt.imshow(rotated_img[0, 0].detach().cpu())
            plt.imshow(_pred_rot, alpha=0.5)
            plt.savefig('debug/coarse_pred.png')
            plt.close()
             
        if self.coarse_pred_only: 
            output_logits = F.interpolate(output_logits, size=original_size, mode='bilinear') if output_logits.shape[-2:] != original_size else output_logits
            pred = output_logits.argmax(dim=1)[0]
            conf = get_confidence_from_logits(output_logits) 
            if self.use_cca:
                _pred = np.array(pred.detach().cpu())
                _pred, conf = cca(_pred, output_logits, return_conf=True)
                pred = torch.from_numpy(_pred)
            if self.training:
                return output_logits, [conf]
            return pred, [conf]
        
        if query_image.shape[-2:] != self.image_size:
            query_image = F.interpolate(query_image, size=self.image_size, mode='bilinear')
            output_logits = F.interpolate(output_logits, size=self.image_size, mode='bilinear')
        if need_softmax(output_logits):
            output_logits = output_logits.softmax(dim=1)
        
        output_p = output_logits
        pred = output_p.argmax(dim=1)[0]
       
        _pred = np.array(output_p.argmax(dim=1)[0].detach().cpu()) 
        if self.use_cca:
            conn_components = cca(_pred, output_logits, return_cc=True)
            conf=None
        else:
            conn_components, conf = get_connected_components(_pred, output_logits, return_conf=True)
        if self.debug:
            plot_connected_components(conn_components, query_image[0,0].detach().cpu(), conf)
        # print(f"connected components took {time.time() - start_time} seconds")
        
        if _pred.max() == 0:
            if output_p.shape[-2:] != original_size:
                output_p = F.interpolate(output_p, size=original_size, mode='bilinear')
            return output_p.argmax(dim=1)[0], [0]

        H, W = query_image.shape[-2:]
        # bbox = self.get_bbox(_pred)
        bbox = self.get_bbox_per_cc(conn_components)
        bbox = bbox / np.array([W, H, W, H]) * max(self.image_size)
        query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min())
        with torch.no_grad():
            image_embedding = self.medsam.image_encoder(query_image)
            
        medsam_seg, conf= self.medsam_inference(image_embedding, bbox, H, W)
        
        if self.debug:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(query_image[0].permute(1,2,0).detach().cpu())
            show_mask(medsam_seg, ax[0])
            ax[1].imshow(query_image[0].permute(1,2,0).detach().cpu())
            show_box(bbox[0], ax[1])
            plt.savefig('debug/medsam_pred.png')
            plt.close()

        medsam_seg = torch.tensor(medsam_seg, device=image_embedding.device)
        if medsam_seg.shape[-2:] != original_size:
            medsam_seg = F.interpolate(medsam_seg.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]

        return medsam_seg, [conf]
    
    def segment_all(self, query_image, query_label):
        H, W = query_image.shape[-2:]
        # bbox = self.get_bbox(_pred)
        # bbox = self.get_bbox_per_cc(conn_components)
        # bbox = bbox / np.array([W, H, W, H]) * max(self.image_size)
        bbox = np.array([[0, 0, W, H]])
        query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min())
        with torch.no_grad():
            image_embedding = self.medsam.image_encoder(query_image)
            
        medsam_seg, conf= self.medsam_inference(image_embedding, bbox, H, W, query_label)
        
        if self.debug:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(query_image[0].permute(1,2,0).detach().cpu())
            show_mask(medsam_seg, ax[0])
            ax[1].imshow(query_image[0].permute(1,2,0).detach().cpu())
            show_box(bbox[0], ax[1])
            plt.savefig('debug/medsam_pred.png')
            plt.close()

        medsam_seg = torch.tensor(medsam_seg, device=image_embedding.device)
        if medsam_seg.shape[-2:] != (H, W):
            medsam_seg = F.interpolate(medsam_seg.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest')[0][0]

        return medsam_seg.view(H,W), [conf]

    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
