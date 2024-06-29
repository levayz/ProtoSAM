"""
Copied from https://github.com/talshaharabany/AutoSAM
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from dataloaders.PolypTransforms import get_polyp_transform
import cv2
KVASIR = "Kvasir"
CLINIC_DB = "CVC-ClinicDB"
COLON_DB = "CVC-ColonDB"
ETIS_DB = "ETIS-LaribPolypDB"
CVC300 = "CVC-300"

DATASETS = (KVASIR, CLINIC_DB, COLON_DB, ETIS_DB)
EXCLUDE_DS = (CVC300, )


def create_suppport_set_for_polyps(n_support=10):
    """
    create a text file contating n_support_images for each dataset
    """
    root_dir = "/disk4/Lev/Projects/Self-supervised-Fewshot-Medical-Image-Segmentation/data/PolypDataset/TrainDataset"
    supp_images = []
    supp_masks = []
    
    image_dir = os.path.join(root_dir, "images")
    mask_dir = os.path.join(root_dir, "masks")
        # randonly sample n_support images and masks
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(
        image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(
        mask_dir) if f.endswith('.png')])
        
    while len(supp_images) < n_support:
        index = random.randint(0, len(image_paths) - 1)
        # check that the index is not already in the support set
        if image_paths[index] in supp_images:
            continue
        supp_images.append(image_paths[index])
        supp_masks.append(mask_paths[index])
                
    with open(os.path.join(root_dir, "support.txt"), 'w') as file:
        for image_path, mask_path in zip(supp_images, supp_masks):
            file.write(f"{image_path} {mask_path}\n")

def create_train_val_test_split_for_polyps():
    root_dir = "/disk4/Lev/Projects/Self-supervised-Fewshot-Medical-Image-Segmentation/data/PolypDataset/"
    # for each subdir in root_dir, create a split file
    num_train_images_per_dataset = {
        "CVC-ClinicDB": 548, "Kvasir": 900, "CVC-300": 0, "CVC-ColonDB": 0}

    num_test_images_per_dataset = {
        "CVC-ClinicDB": 64, "Kvasir": 100, "CVC-300": 60, "CVC-ColonDB": 380}

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            split_file = os.path.join(subdir_path, "split.txt")
            image_dir = os.path.join(subdir_path, "images")
            create_train_val_test_split(
                image_dir, split_file, train_number=num_train_images_per_dataset[subdir], test_number=num_test_images_per_dataset[subdir])


def create_train_val_test_split(root, split_file, train_number=100, test_number=20):
    """
    Create a train, val, test split file for a dataset
    root: root directory of dataset
    split_file: name of split file to create
    train_ratio: ratio of train set
    val_ratio: ratio of val set
    test_ratio: ratio of test set
    """
    # Get all files in root directory
    files = os.listdir(root)
    # Filter out non-image files, remove suffix
    files = [f.split('.')[0]
             for f in files if f.endswith('.jpg') or f.endswith('.png')]
    # Shuffle files
    random.shuffle(files)

    # Calculate number of files for each split
    num_files = len(files)
    num_train = train_number
    num_test = test_number
    num_val = num_files - num_train - num_test
    print(f"num_train: {num_train}, num_val: {num_val}, num_test: {num_test}")
    # Create splits
    train = files[:num_train]
    val = files[num_train:num_train + num_val]
    test = files[num_train + num_val:]

    # Write splits to file
    with open(split_file, 'w') as file:
        file.write("train\n")
        for f in train:
            file.write(f + "\n")
        file.write("val\n")
        for f in val:
            file.write(f + "\n")
        file.write("test\n")
        for f in test:
            file.write(f + "\n")


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, root, image_root=None, gt_root=None, trainsize=352, augmentations=None, train=True, sam_trans=None, datasets=DATASETS, image_size=(1024, 1024), ds_mean=None, ds_std=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.datasets = datasets
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        if image_root is not None and gt_root is not None:
            self.images = [
                os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [
                os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
            # also look in subdirectories
            for subdir in os.listdir(image_root):
                # if not dir, continue
                if not os.path.isdir(os.path.join(image_root, subdir)):
                    continue
                subdir_image_root = os.path.join(image_root, subdir)
                subdir_gt_root = os.path.join(gt_root, subdir)
                self.images.extend([os.path.join(subdir_image_root, f) for f in os.listdir(
                    subdir_image_root) if f.endswith('.jpg') or f.endswith('.png')])
                self.gts.extend([os.path.join(subdir_gt_root, f) for f in os.listdir(
                    subdir_gt_root) if f.endswith('.png')])
                
        else:
            self.images, self.gts = self.get_image_gt_pairs(
                root, split="train" if train else "test", datasets=self.datasets)
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if not 'VPS' in root:
            self.filter_files_and_get_ds_mean_and_std()
        if ds_mean is not None and ds_std is not None:
            self.mean, self.std = ds_mean, ds_std
        self.size = len(self.images)
        self.train = train
        self.sam_trans = sam_trans
        if self.sam_trans is not None:
            # sam trans takes care of norm
            self.mean, self.std = 0 , 1

    def get_image_gt_pairs(self, dir_root: str, split="train", datasets: tuple = DATASETS):
        """
        for each folder in dir root, get all image-gt pairs. Assumes each subdir has a split.txt file
        dir_root: root directory of all subdirectories, each subdirectory contains images and masks folders
        split: train, val, or test
        """
        image_paths = []
        gt_paths = []
        for folder in os.listdir(dir_root):
            if folder not in datasets:
                continue
            split_file = os.path.join(dir_root, folder, "split.txt")
            if os.path.isfile(split_file):
                image_root = os.path.join(dir_root, folder, "images")
                gt_root = os.path.join(dir_root, folder, "masks")
                image_paths_tmp, gt_paths_tmp = self.get_image_gt_pairs_from_text_file(
                    image_root, gt_root, split_file, split=split)
                image_paths.extend(image_paths_tmp)
                gt_paths.extend(gt_paths_tmp)
            else:
                print(
                    f"No split.txt file found in {os.path.join(dir_root, folder)}")

        return image_paths, gt_paths

    def get_image_gt_pairs_from_text_file(self, image_root: str, gt_root: str, text_file: str, split: str = "train"):
        """
        image_root: root directory of images
        gt_root: root directory of ground truth
        text_file: text file containing train, val, test split with the following format:
        train:
        image1
        image2
        ...
        val:
        image1
        image2
        ...
        test:
        image1
        image2
        ...

        split: train, val, or test
        """
        #  Initialize a dictionary to hold file names for each split
        splits = {"train": [], "val": [], "test": []}
        current_split = None

        # Read the text file and categorize file names under each split
        with open(text_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line in splits:
                    current_split = line
                elif line and current_split:
                    splits[current_split].append(line)

        # Get the file names for the requested split
        file_names = splits[split]

        # Create image-ground truth pairs
        image_paths = []
        gt_paths = []
        for name in file_names:
            image_path = os.path.join(image_root, name + '.png')
            gt_path = os.path.join(gt_root, name + '.png')
            image_paths.append(image_path)
            gt_paths.append(gt_path)

        return image_paths, gt_paths

    def get_support_from_dirs(self, support_image_dir, support_mask_dir, n_support=1):
        support_images = []
        support_labels = []
        # get all images and masks
        support_image_paths = sorted([os.path.join(support_image_dir, f) for f in os.listdir(
            support_image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        support_mask_paths = sorted([os.path.join(support_mask_dir, f) for f in os.listdir(
            support_mask_dir) if f.endswith('.png')])
        # sample n_support images and masks
        for i in range(n_support):
            index = random.randint(0, len(support_image_paths) - 1)
            support_img = self.cv2_loader(
                support_image_paths[index], is_mask=False)
            support_mask = self.cv2_loader(
                support_mask_paths[index], is_mask=True)
            support_images.append(support_img)
            support_labels.append(support_mask)

        if self.augmentations:
            support_images = [self.augmentations(
                img, mask)[0] for img, mask in zip(support_images, support_labels)]
            support_labels = [self.augmentations(
                img, mask)[1] for img, mask in zip(support_images, support_labels)]
        
        support_images = [(support_image - self.mean) / self.std  if support_image.max() == 255 and support_image.min() == 0 else support_image for support_image in support_images]
        
        if self.sam_trans is not None:
            support_images = [self.sam_trans.preprocess(
                img).squeeze(0) for img in support_images]
            support_labels = [self.sam_trans.preprocess(
                mask) for mask in support_labels]
        else:
            image_size = self.image_size
            support_images = [torch.nn.functional.interpolate(img.unsqueeze(
                0), size=image_size, mode='bilinear', align_corners=False).squeeze(0) for img in support_images]
            support_labels = [torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(
                0), size=image_size, mode='nearest').squeeze(0).squeeze(0) for mask in support_labels]

        return torch.stack(support_images), torch.stack(support_labels)
    
    def get_support_from_text_file(self, text_file, n_support=1):
        """
        each row in the file has 2 paths divided by space, the first is the image path and the second is the mask path
        """
        support_images = []
        support_labels = []
        with open(text_file, 'r') as file:
            for line in file:
                image_path, mask_path = line.strip().split()
                support_images.append(image_path)
                support_labels.append(mask_path)
                
        # indices = random.choices(range(len(support_images)), k=n_support)
        if n_support > len(support_images):
            raise ValueError(f"n_support ({n_support}) is larger than the number of images in the text file ({len(support_images)})")
        
        n_support_images = support_images[:n_support]
        n_support_labels = support_labels[:n_support]
                
        return n_support_images, n_support_labels

    def get_support(self, n_support=1, support_image_dir=None, support_mask_dir=None, text_file=None):
        """
        Get support set from specified directories, text file or from the dataset itself
        """
        if support_image_dir is not None and support_mask_dir:
            return self.get_support_from_dirs(support_image_dir, support_mask_dir, n_support=n_support)
        elif text_file is not None:
            support_image_paths, support_gt_paths = self.get_support_from_text_file(text_file, n_support=n_support)
        else:
            # randomly sample n_support images and masks from the dataset
            indices = random.choices(range(self.size), k=n_support)
            # indices = list(range(n_support))
            print(f"support indices:{indices}")
            support_image_paths = [self.images[index] for index in indices]
            support_gt_paths = [self.gts[index] for index in indices]
            
        support_images = []
        support_gts = []
        
        for image_path, gt_path in zip(support_image_paths, support_gt_paths):
            support_img = self.cv2_loader(image_path, is_mask=False)
            support_mask = self.cv2_loader(gt_path, is_mask=True)
            out = self.process_image_gt(support_img, support_mask)
            support_images.append(out['image'].unsqueeze(0))
            support_gts.append(out['label'].unsqueeze(0))
            if len(support_images) >= n_support:
                break
        return support_images, support_gts, out['case']
        # return torch.stack(support_images), torch.stack(support_gts), out['case']
    
    def process_image_gt(self, image, gt, dataset=""):
        """
        image and gt are expected to be output from self.cv2_loader
        """
        original_size = tuple(image.shape[-2:])
        if self.augmentations:
            image, mask = self.augmentations(image, gt)
        
        if self.sam_trans:
            image, mask = self.sam_trans.apply_image_torch(
                image.unsqueeze(0)), self.sam_trans.apply_image_torch(mask)
        elif image.max() <= 255 and image.min() >= 0:
            image = (image - self.mean) / self.std
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # image_size = tuple(img.shape[-2:])

        image_size = self.image_size
        if self.sam_trans is None:
            image = torch.nn.functional.interpolate(image.unsqueeze(
                0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(
                0), size=image_size, mode='nearest').squeeze(0).squeeze(0)
            # img = (img - img.min()) / (img.max() - img.min()) # TODO uncomment this if results get worse

        return {'image': self.sam_trans.preprocess(image).squeeze(0) if self.sam_trans else image,
                'label': self.sam_trans.preprocess(mask) if self.sam_trans else mask,
                'original_size': torch.Tensor(original_size),
                'image_size': torch.Tensor(image_size),
                'case': dataset} # case to be compatible with polyp video dataset
    
    def get_dataset_name_from_path(self, path):
        for dataset in self.datasets:
            if dataset in path:
                return dataset
        return ""
        
    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=True)
        dataset = self.get_dataset_name_from_path(self.images[index])
        return self.process_image_gt(image, gt, dataset)

    def filter_files_and_get_ds_mean_and_std(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        ds_mean = 0
        ds_std = 0
        for img_path, gt_path in zip(self.images, self.gts):
            if any([ex_ds in img_path for ex_ds in EXCLUDE_DS]):
                continue
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            ds_mean += np.array(img).mean()
            ds_std += np.array(img).std()
        self.images = images
        self.gts = gts
        self.mean = ds_mean / len(self.images)
        self.std = ds_std / len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        # with open(path, 'rb') as f:
        # img = Image.open(f)
        # return img.convert('1')
        img = cv2.imread(path, 0)
        return img

    def cv2_loader(self, path, is_mask):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(
                path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        # return 32
        return self.size
    

class SuperpixPolypDataset(PolypDataset):
    def __init__(self, root, image_root=None, gt_root=None, trainsize=352, augmentations=None, train=True, sam_trans=None, datasets=DATASETS, image_size=(1024, 1024), ds_mean=None, ds_std=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.datasets = datasets
        self.image_size = image_size
        # print(self.augmentations)
        if image_root is not None and gt_root is not None:
            self.images = [
                os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [
                os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') and 'superpix' in f]
            # also look in subdirectories
            for subdir in os.listdir(image_root):
                # if not dir, continue
                if not os.path.isdir(os.path.join(image_root, subdir)):
                    continue
                subdir_image_root = os.path.join(image_root, subdir)
                subdir_gt_root = os.path.join(gt_root, subdir)
                self.images.extend([os.path.join(subdir_image_root, f) for f in os.listdir(
                    subdir_image_root) if f.endswith('.jpg') or f.endswith('.png')])
                self.gts.extend([os.path.join(subdir_gt_root, f) for f in os.listdir(
                    subdir_gt_root) if f.endswith('.png')])
                
        else:
            self.images, self.gts = self.get_image_gt_pairs(
                root, split="train" if train else "test", datasets=self.datasets)
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if not 'VPS' in root:
            self.filter_files_and_get_ds_mean_and_std()
        if ds_mean is not None and ds_std is not None:
            self.mean, self.std = ds_mean, ds_std
        self.size = len(self.images)
        self.train = train
        self.sam_trans = sam_trans
        if self.sam_trans is not None:
            # sam trans takes care of norm
            self.mean, self.std = 0 , 1
            
            
    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=False)
        gt = gt[:, :, 0]
        fgpath = os.path.basename(self.gts[index]).split('.png')[0].split('superpix-MIDDLE_')
        fgpath = os.path.join(os.path.dirname(self.gts[index]), 'fgmask_' + fgpath[1] + '.png')
        fg = self.cv2_loader(fgpath, is_mask=True)
        dataset = self.get_dataset_name_from_path(self.images[index])

        # randomly choose a superpixels from the gt
        gt[1-fg] = 0
        sp_id = random.choice(np.unique(gt)[1:])
        sp = (gt == sp_id).astype(np.uint8)
        
        
        out = self.process_image_gt(image, gt, dataset)
        support_image, support_sp, dataset = out["image"], out["label"], out["case"]
        
        out = self.process_image_gt(image, sp, dataset)
        query_image, query_sp, dataset = out["image"], out["label"], out["case"]

        # TODO tile the masks to have 3 channels?
      
        support_bg_mask = 1 - support_sp
        support_masks = {"fg_mask": support_sp, "bg_mask": support_bg_mask}
        
        batch = {"support_images" : [[support_image]],
                "support_mask" : [[support_masks]],
                "query_images" : [query_image],
                "query_labels" : [query_sp],
                "scan_id" : [dataset]
        }
        
        return batch


def get_superpix_polyp_dataset(image_size:tuple=(1024,1024), sam_trans=None):
    transform_train, transform_test = get_polyp_transform()
    image_root = './data/PolypDataset/TrainDataset/images/'
    gt_root = './data/PolypDataset/TrainDataset/superpixels/'
    ds_train = SuperpixPolypDataset(root=image_root, image_root=image_root, gt_root=gt_root,
                            augmentations=transform_train,
                            sam_trans=sam_trans,
                            image_size=image_size)
    
    return ds_train

def get_polyp_dataset(image_size, sam_trans=None):
    transform_train, transform_test = get_polyp_transform()
    image_root = './data/PolypDataset/TrainDataset/images/'
    gt_root = './data/PolypDataset/TrainDataset/masks/'
    ds_train = PolypDataset(root=image_root, image_root=image_root, gt_root=gt_root,
                            augmentations=transform_test, sam_trans=sam_trans, train=True, image_size=image_size)
    image_root = './data/PolypDataset/TestDataset/test/images/'
    gt_root = './data/PolypDataset/TestDataset/test/masks/'
    ds_test = PolypDataset(root=image_root, image_root=image_root, gt_root=gt_root, train=False,
                           augmentations=transform_test, sam_trans=sam_trans, image_size=image_size)
    return ds_train, ds_test


def get_tests_polyp_dataset(sam_trans):
    transform_train, transform_test = get_polyp_transform()

    image_root = './data/polyp/TestDataset/Kvasir/images/'
    gt_root = './data/polyp/TestDataset/Kvasir/masks/'
    ds_Kvasir = PolypDataset(
        image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = './data/polyp/TestDataset/CVC-ClinicDB/images/'
    gt_root = './data/polyp/TestDataset/CVC-ClinicDB/masks/'
    ds_ClinicDB = PolypDataset(
        image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = './data/polyp/TestDataset/CVC-ColonDB/images/'
    gt_root = './data/polyp/TestDataset/CVC-ColonDB/masks/'
    ds_ColonDB = PolypDataset(
        image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    image_root = './data/polyp/TestDataset/ETIS-LaribPolypDB/images/'
    gt_root = './data/polyp/TestDataset/ETIS-LaribPolypDB/masks/'
    ds_ETIS = PolypDataset(
        image_root, gt_root, augmentations=transform_test, train=False, sam_trans=sam_trans)

    return ds_Kvasir, ds_ClinicDB, ds_ColonDB, ds_ETIS


if __name__ == '__main__':
    # create_train_val_test_split_for_polyps()
    create_suppport_set_for_polyps()
