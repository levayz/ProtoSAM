import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from models.grid_proto_fewshot import FewShotSeg
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from models.SamWrapper import SamWrapper
from util.utils import cca, get_connected_components, rotate_tensor_no_crop, reverse_tensor, get_confidence_from_logits
from util.lora import inject_trainable_lora
from models.segment_anything.utils.transforms import ResizeLongestSide
import cv2
import time
from abc import ABC, abstractmethod

CONF_MODE="conf"
CENTROID_MODE="centroid"
BOTH_MODE="both"
POINT_MODES=(CONF_MODE, CENTROID_MODE, BOTH_MODE)

TYPE_ALPNET="alpnet"
TYPE_SAM="sam"

def plot_connected_components(cca_output, original_image, confidences:dict=None, title="debug/connected_components.png"):
    num_labels, labels, stats, centroids = cca_output
    # Create an output image with random colors for each component
    output_image = np.zeros((labels.shape[0], labels.shape[1], 3), np.uint8)
    for label in range(1, num_labels):  # Start from 1 to skip the background
        mask = labels == label
        output_image[mask] = np.random.randint(0, 255, size=3)

    # Plotting the original and the colored components image
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(original_image), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), plt.title('Connected Components')
    if confidences is not None:
        # Plot the axes color chart with the confidences, use the same colors as the connected components
        plt.subplot(122)
        scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=list(confidences.values()), cmap='jet')
        plt.colorbar(scatter)

    plt.savefig(title)
    plt.close()

class SegmentationInput(ABC):
    @abstractmethod
    def set_query_images(self, query_images):
        pass
    
    def to(self, device):
        pass
    
class SegmentationOutput(ABC):
    @abstractmethod
    def get_prediction(self):
        pass

class ALPNetInput(SegmentationInput): # for alpnet
    def __init__(self, support_images:list, support_labels:list, query_images:torch.Tensor, isval, val_wsize, show_viz=False, supp_fts=None):
        self.supp_imgs = [support_images]
        self.fore_mask = [support_labels]
        self.back_mask = [[1 - sup_labels for sup_labels in support_labels]]
        self.qry_imgs = [query_images]
        self.isval = isval
        self.val_wsize = val_wsize
        self.show_viz = show_viz
        self.supp_fts = supp_fts
        
    def set_query_images(self, query_images):
        self.qry_imgs = [query_images]
        
    def to(self, device):
        self.supp_imgs = [[supp_img.to(device) for way in self.supp_imgs for supp_img in way]]
        self.fore_mask = [[fore_mask.to(device) for way in self.fore_mask for fore_mask in way]]
        self.back_mask = [[back_mask.to(device) for way in self.back_mask for back_mask in way]]
        self.qry_imgs = [qry_img.to(device) for qry_img in self.qry_imgs]
        if self.supp_fts is not None:
            self.supp_fts = self.supp_fts.to(device)

class ALPNetOutput(SegmentationOutput):
    def __init__(self, pred, align_loss, sim_maps, assign_maps, proto_grid, supp_fts, qry_fts):
        self.pred = pred
        self.align_loss = align_loss
        self.sim_maps = sim_maps
        self.assign_maps = assign_maps
        self.proto_grid = proto_grid
        self.supp_fts = supp_fts
        self.qry_fts = qry_fts
        
    def get_prediction(self):
        return self.pred
        
class SAMWrapperInput(SegmentationInput):
    def __init__(self, image, image_labels):
        self.image = image
        self.image_labels = image_labels
        
    def set_query_images(self, query_images):
        B, C, H, W = query_images.shape
        if isinstance(query_images, torch.Tensor):
            query_images = query_images.cpu().detach().numpy()
        assert B == 1, "batch size must be 1"
        query_images = (query_images - query_images.min()) / (query_images.max() - query_images.min()) * 255
        query_images = query_images.astype(np.uint8)
        self.image = np.transpose(query_images[0], (1, 2, 0))
        
    def to(self, device):
        pass


class InputFactory(ABC):
    @staticmethod
    def create_input(input_type, query_image, support_images=None, support_labels=None, isval=False, val_wsize=None, show_viz=False, supp_fts=None, original_sz=None, img_sz=None, gts=None):
        
        if input_type == TYPE_ALPNET:
            return ALPNetInput(support_images, support_labels, query_image, isval, val_wsize, show_viz, supp_fts)
        elif input_type == TYPE_SAM:
            qimg = np.array(query_image.detach().cpu())
            B,C,H,W = qimg.shape
            assert B == 1, "batch size must be 1"
            gts = np.array(gts.detach().cpu()).astype(np.uint8).reshape(H,W)
            assert np.unique(gts).shape[0] <= 2, "support labels must be binary"
            gts[gts > 0] = 1
            qimg = qimg.reshape(H,W,C)
            qimg = (qimg - qimg.min()) / (qimg.max() - qimg.min()) * 255
            qimg = qimg.astype(np.uint8)
            return SAMWrapperInput(qimg, gts)
        else:
            raise ValueError(f"input_type not supported")
 
    
class ModelWrapper(ABC):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, input_data: SegmentationInput)->SegmentationOutput:
        pass

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train() 
    
    def parameters(self):
        pass
        
class ALPNetWrapper(ModelWrapper):
    def __init__(self, model: FewShotSeg):
        super().__init__(model)
        
    def __call__(self, input_data: ALPNetInput):
        output = self.model(**input_data.__dict__)
        output = ALPNetOutput(*output)
        return output.pred
    
    def parameters(self):
        return self.model.encoder.parameters()
    
    def train(self):
        self.model.encoder.train()
        
class SamWrapperWrapper(ModelWrapper):
    def __init__(self, model:SamWrapper):
        super().__init__(model)
        
    def __call__(self, input_data: SAMWrapperInput):
        pred = self.model(**input_data.__dict__)
        # make pred look like logits
        pred = torch.tensor(pred).float()[None, None, ...]
        pred = torch.cat([1-pred, pred], dim=1)
        return pred

    def to(self, device):
        self.model.sam.to(device)
    
class ProtoSAM(nn.Module):
    def __init__(self, image_size, coarse_segmentation_model:ModelWrapper, sam_pretrained_path="pretrained_model/sam_default.pth", num_points_for_sam=1, use_points=True, use_bbox=False, use_mask=False, debug=False, use_cca=False, point_mode=CONF_MODE, use_sam_trans=True, coarse_pred_only=False, alpnet_image_size=None, use_neg_points=False, ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.coarse_segmentation_model = coarse_segmentation_model
        self.get_sam(sam_pretrained_path, use_sam_trans)
        self.num_points_for_sam = num_points_for_sam
        self.use_points = use_points
        self.use_bbox = use_bbox # if False then uses points
        self.use_mask = use_mask
        self.use_neg_points = use_neg_points
        assert self.use_bbox or self.use_points or self.use_mask, "must use at least one of bbox, points, or mask"
        self.use_cca = use_cca
        self.point_mode = point_mode
        if self.point_mode not in POINT_MODES:
            raise ValueError(f"point mode must be one of {POINT_MODES}")
        self.debug=debug
        self.coarse_pred_only = coarse_pred_only
         
    def get_sam(self, checkpoint_path, use_sam_trans):
        model_type="vit_b" # TODO make generic?
        if 'vit_h' in checkpoint_path:
            model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval()
        self.predictor = SamPredictor(self.sam)
        self.sam.requires_grad_(False)
        if use_sam_trans:
            # sam_trans = ResizeLongestSide(self.sam.image_encoder.img_size, pixel_mean=[0], pixel_std=[1])
            sam_trans = ResizeLongestSide(self.sam.image_encoder.img_size)
            sam_trans.pixel_mean = torch.tensor([0, 0, 0]).view(3, 1, 1)
            sam_trans.pixel_std = torch.tensor([1, 1, 1]).view(3, 1, 1)
        else:
            sam_trans = None
            
        self.sam_trans = sam_trans
        
    def get_bbox(self, pred):
        '''
        pred tensor of shape (H, W) where 1 represents foreground and 0 represents background
        returns a list of 2d points representing the bbox
        ''' 
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred)
        # get the indices of the foreground points
        indices = torch.nonzero(pred)
        # get the min and max of the indices
        min_x = indices[:, 1].min()
        max_x = indices[:, 1].max()
        min_y = indices[:, 0].min()
        max_y = indices[:, 0].max()
        # get the bbox
        bbox = [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]
        
         
        return bbox  
    
    def get_bbox_per_cc(self, conn_components):
        """
        conn_components: output of cca function
        return list of bboxes per connected component, each bbox is a list of 2d points
        """
        bboxes = []
        for i in range(1, conn_components[0]):
            # get the indices of the foreground points
            indices = torch.nonzero(torch.tensor(conn_components[1] == i))
            # get the min and max of the indices
            min_x = indices[:, 1].min()
            max_x = indices[:, 1].max()
            min_y = indices[:, 0].min()
            max_y = indices[:, 0].max()
            # get the bbox
            # bbox = [[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]
            # bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            # bbox should be in a XYXY format
            bbox = [min_x, min_y, max_x, max_y]
            bboxes.append(bbox)

        bboxes = np.array(bboxes)
        return bboxes
    
    def get_most_conf_points(self, output_p_fg, pred, k):
        '''
        get the k most confident points from pred
        output_p: 3d tensor of shape (H, W)
        pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
        '''
        # Create a mask where pred is 1
        mask = pred.bool()

        # Apply the mask to output_p_fg
        masked_output_p_fg = output_p_fg[mask]
        if masked_output_p_fg.numel() == 0:
            return None, None
        # Get the top k probabilities and their indices
        confidences, indices = torch.topk(masked_output_p_fg, k)

        # Get the locations of the top k points in xy format
        locations = torch.nonzero(mask)[indices]
        # convert locations to xy format
        locations = locations[:, [1, 0]]
        # convert locations to list of lists
        # points = [loc.tolist() for loc in locations]
        
        return locations.numpy(), [float(conf.item()) for conf in confidences]
    
    
    def plot_most_conf_points(self, points, confidences, pred, image, bboxes=None, title=None):
        '''
        points: np array of shape (N, 2) where each row is a point in xy format
        pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
        image: 2d tensor of shape (H,W) representing the image
        bbox: list or np array of shape (N, 4) where each row is a bbox in xyxy format
        '''
        warnings.filterwarnings('ignore', category=UserWarning)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        if title is None:
            title="debug/most_conf_points.png"
            
        fig = plt.figure()
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)
        plt.imshow(pred, alpha=0.5)
        for i, point in enumerate(points):
            plt.scatter(point[0][0], point[0][1], cmap='viridis', marker='*', c='red')
            if confidences is not None:
                plt.text(point[0], point[1], f"{confidences[i]:.3f}", fontsize=12, color='red')
        # assume points is a list of lists
        if bboxes is not None:
            for bbox in bboxes:
                if bbox is None:
                    continue
                bbox = np.array(bbox)
                # plt.scatter(bbox[:, 1], bbox[:, 0], c='red')
                # plot a line connecting the points
                box = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
                box = np.vstack([box, box[0]])
                plt.plot(box[:, 0], box[:, 1], c='green')
        plt.colorbar()
        fig.savefig(title)
        plt.close(fig)
        
    def plot_sam_preds(self, masks, scores, image, input_point, input_label, input_box=None):
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            if input_point is not None:
                show_points(input_point, input_label, plt.gca())
            if input_box is not None:
                show_box(input_box, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            # plt.axis('off')
            plt.savefig(f'debug/sam_mask_{i+1}.png')
            plt.close()
            if i > 5:
                break
        
    def get_sam_input_points(self, conn_components, output_p, get_neg_points=False, l=1): 
        """
        args:
        conn_components: output of cca function
        output_p: 3d tensor of shape (1, 2, H, W)
        get_neg_points: bool, if True then return the negative points
        l: int, number of negative points to get
        """
        sam_input_points = []
        sam_neg_points = []
        fg_p = output_p[0, 1].detach().cpu()
        
        if get_neg_points:
            # get global negative points
            bg_p = output_p[0, 0].detach().cpu()
            bg_p[bg_p < 0.95] = 0
            bg_pred = torch.where(bg_p > 0, 1, 0)
            glob_neg_points, _ = self.get_most_conf_points(bg_p, bg_pred, 1)
            if self.debug:
                # plot the bg_p as a heatmap
                plt.figure()
                plt.imshow(bg_p)
                plt.colorbar()
                plt.savefig('debug/bg_p_heatmap.png')
                plt.close()
        
        for i, cc_id in enumerate(np.unique(conn_components[1])):
            # get self.num_points_for_sam most confident points from pred
            if cc_id == 0:
                continue  # skip background
            pred = torch.tensor(conn_components[1] == cc_id).float()

            if self.point_mode == CONF_MODE:
                points, confidences = self.get_most_conf_points(fg_p, pred, self.num_points_for_sam)  # (N, 2)
            elif self.point_mode == CENTROID_MODE:
                points = conn_components[3][cc_id][None, :]  # (1, 2)
                confidences = [1 for _ in range(len(points))]
            elif self.point_mode == BOTH_MODE:
                points, confidences = self.get_most_conf_points(fg_p, pred, self.num_points_for_sam)
                point = conn_components[3][cc_id][None, :]
                points = np.vstack([points, point])  # (N+1, 2)
                confidences.append(1)
            else:
                raise NotImplementedError(f"point mode {self.point_mode} not implemented")
            sam_input_points.append(np.array(points))
            
            if get_neg_points:
                pred_uint8 = (pred.numpy() * 255).astype(np.uint8)

                # Dilate the mask to expand it
                kernel_size = 3  # Size of the dilation kernel, adjust accordingly
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilation_iterations = 10  # Number of times dilation is applied, adjust as needed
                dilated_mask = cv2.dilate(pred_uint8, kernel, iterations=dilation_iterations)

                # Subtract the original mask from the dilated mask
                # This will give a boundary that is only outside the original mask
                outside_boundary = dilated_mask - pred_uint8

                # Convert back to torch tensor and normalize
                boundary = torch.tensor(outside_boundary).float() / 255
                try:
                    bg_p = output_p[0, 0].detach().cpu()
                    neg_points, neg_confidences = self.get_most_conf_points(bg_p, boundary, l)
                except RuntimeError as e:
                    # make each point (None, None)
                    neg_points = None
                # append global negative points to the negative points
                if neg_points is not None and glob_neg_points is not None:
                    neg_points = np.vstack([neg_points, glob_neg_points])
                else:
                    neg_points = glob_neg_points if neg_points is None else neg_points
                if self.debug and neg_points is not None:
                    # draw an image with 2 subplots, one is the pred and the other is the boundary
                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(pred)
                    plt.imshow(boundary, alpha=0.5)
                    # plot the neg points
                    plt.scatter(neg_points[:, 0], neg_points[:, 1], cmap='viridis', marker='*', c='red')
                    plt.subplot(122)
                    plt.imshow(pred)
                    plt.scatter(neg_points[:, 0], neg_points[:, 1], cmap='viridis', marker='*', c='red')
                    plt.savefig('debug/pred_and_boundary.png')
                    plt.close()
                sam_neg_points.append(neg_points)
            else:
                # create a list of None same shape as points
                sam_neg_points = [None for _ in range(len(sam_input_points))]

        sam_input_labels = np.array([l+1 for l, cc_points in enumerate(sam_input_points) for _ in range(len(cc_points))])
        sam_input_points = np.stack(sam_input_points)  # should be of shape (num_connected_components, num_points_for_sam, 2)
        # if get_neg_points:
        sam_neg_input_points = np.stack(sam_neg_points) if sam_neg_points is not None else None
        if sam_neg_input_points is not None:
            sam_neg_input_points = sam_neg_points
            sam_neg_input_labels = np.array([0] * len(sam_neg_input_points) )
        else:
            sam_neg_input_points = None
            sam_neg_input_labels = None

        return sam_input_points, sam_input_labels, sam_neg_input_points, sam_neg_input_labels
    
    def get_sam_input_mask(self, conn_components):
        sam_input_masks = []
        sam_input_mask_lables = []
        for i, cc_id in enumerate(np.unique(conn_components[1])):
            # get self.num_points_for_sam most confident points from pred
            if cc_id == 0:
                continue
            pred = torch.tensor(conn_components[1] == cc_id).float()
            sam_input_masks.append(pred)
            sam_input_mask_lables.append(cc_id)

        sam_input_masks = np.stack(sam_input_masks)
        sam_input_mask_lables = np.array(sam_input_mask_lables)
        
        return sam_input_masks, sam_input_mask_lables

    def predict_w_masks(self, sam_input_masks, qry_img, original_size):
        masks = []
        scores = []
        for in_mask in sam_input_masks:
            in_mask = cv2.resize(in_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            in_mask[in_mask == 1] = 10
            in_mask[in_mask == 0] = -8
            assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8   
            self.predictor.set_image(qry_img)
            mask, score, _ = self.predictor.predict(
                mask_input=in_mask[None, ...].astype(np.uint8),
                multimask_output=True)
            # get max index from score
            if self.debug:
                # plot each channel of mask
                fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                for i in range(mask.shape[0]):
                    ax[i].imshow(qry_img)
                    ax[i].imshow(mask[i], alpha=0.5)
                    ax[i].set_title(f"Mask {i+1}, Score: {score[i]:.3f}", fontsize=18)
                    # ax[i].axis('off')
                ax[-1].imshow(cv2.resize(in_mask, original_size, interpolation=cv2.INTER_NEAREST))
                fig.savefig(f'debug/sam_mask_from_mask_prompts.png')
                plt.close(fig)
                           
                        
            max_index = score.argmax()
            masks.append(mask[max_index])
            scores.append(score[max_index])
        
        return masks, scores
    
    def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points, qry_img, pred, return_logits=False):
        masks, scores = [], []
        self.predictor.set_image(qry_img)
        # if sam_input_points is None:
        #     sam_input_points = [None for _ in range(len(bboxes))]
        for point, bbox_xyxy, neg_point in zip(sam_input_points, bboxes, sam_neg_input_points): 
            assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8   
            points = point
            point_labels = np.array([1] * len(point)) if point is not None else None
            if self.use_neg_points:
                neg_points = [npoint for npoint in neg_point if None not in npoint] 
                points = np.vstack([point, *neg_points])
                point_labels = np.array([1] * len(point) + [0] * len(neg_points))
            if self.debug: 
                self.plot_most_conf_points(points[:, None, ...], None, pred, qry_img, bboxes=bbox_xyxy[None,...] if bbox_xyxy is not None else None, title="debug/pos_neg_points.png") # TODO add plots for all points not just the first set of points
            mask, score, _ = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                # box=bbox_xyxy[None, :] if bbox_xyxy is not None else None,
                box = bbox_xyxy if bbox_xyxy is not None else None,
                # mask_input=sam_mask_input,
                return_logits=return_logits,
                multimask_output=False if self.use_cca else True
            )
            # best_pred_idx = np.argmax(score)
            best_pred_idx = 0
            masks.append(mask[best_pred_idx])
            scores.append(score[best_pred_idx])
        
        if self.debug:
            # pass
            self.plot_sam_preds(mask, score, qry_img[...,0], points.reshape(-1,2) if sam_input_points is not None else None, point_labels, input_box=bbox_xyxy if bbox_xyxy is not None else None)

        return masks, scores
    
    
    def forward(self, query_image, coarse_model_input, degrees_rotate=0):
        """
        query_image: 3d tensor of shape (1, 3, H, W)
        images should be normalized with mean and std but not to [0, 1]?
        """
        original_size = query_image.shape[-2]
        # rotate query_image by degrees_rotate
        start_time = time.time()
        rotated_img, (rot_h, rot_w) = rotate_tensor_no_crop(query_image, degrees_rotate)
        # print(f"rotating query image took {time.time() - start_time} seconds")
        start_time = time.time()
        coarse_model_input.set_query_images(rotated_img)
        output_logits_rot = self.coarse_segmentation_model(coarse_model_input)
        # print(f"ALPNet took {time.time() - start_time} seconds")
       
        if degrees_rotate != 0:
            start_time = time.time()
            output_logits = reverse_tensor(output_logits_rot, rot_h, rot_w, -degrees_rotate)
            # print(f"reversing rotated output_logits took {time.time() - start_time} seconds")
        else:
            output_logits = output_logits_rot
        
        # check if softmax is needed 
        output_p = output_logits.softmax(dim=1)
        # output_p = output_logits
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
        # if need_softmax(output_logits):
        # output_logits = output_logits.softmax(dim=1)
        
        # output_p = output_logits
        output_p = output_logits.softmax(dim=1)
        pred = output_p.argmax(dim=1)[0]
       
        _pred = np.array(output_p.argmax(dim=1)[0].detach().cpu()) 
        start_time = time.time()
        if self.use_cca:
            conn_components = cca(_pred, output_logits, return_cc=True)
            conf=None
        else:
            conn_components, conf = get_connected_components(_pred, output_logits, return_conf=True)
        if self.debug:
            plot_connected_components(conn_components, query_image[0,0].detach().cpu(), conf)
        # print(f"connected components took {time.time() - start_time} seconds")
        if _pred.max() == 0:
            return output_p.argmax(dim=1)[0], [0]
        
        # get bbox from pred
        if self.use_bbox:
            start_time = time.time()
            try:
                bboxes = self.get_bbox_per_cc(conn_components) 
            except:
                bboxes = [None] * conn_components[0]
        else:
            bboxes = [None] * conn_components[0]
        # print(f"getting bboxes took {time.time() - start_time} seconds")


        start_time = time.time()
        if self.use_points:
            sam_input_points, sam_input_point_labels, sam_neg_input_points, sam_neg_input_labels = self.get_sam_input_points(conn_components, output_p, get_neg_points=self.use_neg_points, l=1)
        else:
            sam_input_points = [None] * conn_components[0]
            sam_input_point_labels = [None] * conn_components[0]
            sam_neg_input_points = [None] * conn_components[0]
            sam_neg_input_labels = [None] * conn_components[0]
        # print(f"getting sam input points took {time.time() - start_time} seconds")
        
        if self.use_mask:
            sam_input_masks, sam_input_mask_labels = self.get_sam_input_mask(conn_components) 
        else:
            sam_input_masks = None
            sam_input_mask_labels = None
            
        if self.debug and sam_input_points is not None:
            title = f'debug/most_conf_points.png'
            if self.use_cca:
                title = f'debug/most_conf_points_cca.png'
            # convert points to a list where each item is a list of 2 elements in xy format
            self.plot_most_conf_points(sam_input_points, None, _pred, query_image[0, 0].detach().cpu(), bboxes=bboxes, title=title) # TODO add plots for all points not just the first set of points
        
        # self.sam_trans = None
        if self.sam_trans is None:
            query_image = query_image.permute(1, 2, 0).detach().cpu().numpy() 
        else:
            query_image = self.sam_trans.apply_image_torch(query_image[0])
            query_image = self.sam_trans.preprocess(query_image)
            query_image = query_image.permute(1, 2, 0).detach().cpu().numpy()
            # mask = self.sam_trans.preprocess(mask) 
        
        
        query_image = ((query_image - query_image.min()) / (query_image.max() - query_image.min()) * 255).astype(np.uint8)
        if self.use_mask:
            masks, scores = self.predict_w_masks(sam_input_masks, query_image, original_size)
        
        start_time = time.time()
        if self.use_points or self.use_bbox:
            masks, scores = self.predict_w_points_bbox(sam_input_points, bboxes, sam_neg_input_points, query_image, pred, return_logits=True if self.training else False)
        # print(f"predicting w points/bbox took {time.time() - start_time} seconds")
            
        pred = sum(masks)
        if not self.training:
            pred = pred > 0
        pred = torch.tensor(pred).float().to(output_p.device)
        
        # pred = torch.tensor(masks[0]).float().cuda()
        # resize pred to the size of the input
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest')[0][0]
        
        return pred, scores
    
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def need_softmax(tensor, dim=1):
    return not torch.all(torch.isclose(tensor.sum(dim=dim), torch.ones_like(tensor.sum(dim=dim))) & (tensor >= 0))

        
        
        
        