import torch
import numpy as np
import matplotlib.pyplot as plt

"""
simple dataset, gets the images and masks as list together with a transform function that
shoudl receive both the image and the mask.
loop means how many times to loop the dataset per epoch
"""

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, mask_list, transform=None, norm_func=None, loops=10, modality="", debug=False, image_size=None):
        self.image_list = image_list
        if image_size is not None:
            if len(image_size) == 1:
                image_size = (image_size, image_size)
            self.image_size = image_size
        else:
            self.image_size = image_list[0].shape[-2:] 
        self.mask_list = mask_list
        self.transform = transform
        self.norm_func = norm_func
        self.loops = loops
        self.modality = modality
        self.debug = debug
        
    def __len__(self):
        return len(self.image_list) * self.loops

    def __getitem__(self, idx):
        idx = idx % (len(self.image_list))
        image = self.image_list[idx].numpy()
        mask = self.mask_list[idx].to(dtype=torch.uint8).numpy()
        if self.modality == "CT":
            image = image.astype(np.uint8)
            if self.transform:
                image, mask = self.transform(image, mask)
        else:
            # mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
            if self.transform:
                image, mask = self.transform(image, mask)

        if self.norm_func:
            image = self.norm_func(image)
        
        mask[mask != 0] = 1
        
        if self.image_size != image.shape[-2:]:
            image = torch.nn.functional.interpolate(torch.tensor(image).unsqueeze(0), self.image_size, mode='bilinear').squeeze(0)
            mask = torch.nn.functional.interpolate(torch.tensor(mask).unsqueeze(0).unsqueeze(0), self.image_size, mode='nearest').squeeze(0).squeeze(0)
        
        # plot image and mask
        if self.debug:
            fig = plt.figure()
            plt.imshow((image[0]- image.min()) / (image.max() - image.min()))
            plt.imshow(mask, alpha=0.5)
            plt.savefig("debug/support_image_mask.png")
            plt.close(fig)
         
        image_size = torch.tensor(tuple(image.shape[-2:]))
        return image, mask