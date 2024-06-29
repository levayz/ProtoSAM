import torch
import torch.nn as nn
import numpy as np
from models.segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from models.segment_anything.utils.transforms import ResizeLongestSide
import cv2

def get_iou(mask, label):
    tp = (mask * label).sum()
    fp = (mask * (1-label)).sum()
    fn = ((1-mask) * label).sum()
    iou = tp / (tp + fp + fn)
    return iou

class SamWrapper(nn.Module):
    def __init__(self,sam_args):
        """
                sam_args: dict should include the following
        {
            "model_type": "vit_h",
            "sam_checkpoint": "path to checkpoint" pretrained_model/sam_vit_h.pth
        }
        """
        super().__init__()
        self.sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        
    def forward(self, image, image_labels):
        """
        generate masks for a batch of images
        return mask that has the largest iou with the image label
        Args: 
            images (np.ndarray): The image to generate masks for, in HWC uint8 format.
            image_labels (np.ndarray): The image labels to generate masks for, in HWC uint8 format. assuming binary labels
        """
        image = self.transform.apply_image(image)
        masks = self.mask_generator.generate(image)
        
        best_index, best_iou = None, 0
        for i, mask in enumerate(masks): 
            segmentation = mask['segmentation']
            iou = get_iou(segmentation.astype(np.uint8), image_labels)
            if best_index is None or iou > best_iou:
                best_index = i
                best_iou = iou
                
        return masks[best_index]['segmentation']

    def to(self, device):
        self.sam.to(device)
        self.mask_generator.to(device)
        self.mask_generator.predictor.to(device)
             
            

if __name__ == "__main__":
    sam_args = {
        "model_type": "vit_h",
        "sam_checkpoint": "pretrained_model/sam_vit_h.pth"
    }
    sam_wrapper = SamWrapper(sam_args).cuda()
    image = cv2.imread("./Kheops-Pyramid.jpg")
    image = np.array(image).astype('uint8')
    image_labels = torch.rand(1,3,224,224)
    sam_wrapper(image, image_labels)
        
        