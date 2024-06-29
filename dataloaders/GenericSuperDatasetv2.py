"""
Dataset for training with pseudolabels
TODO:
1. Merge with manual annotated dataset
2. superpixel_scale -> superpix_config, feed like a dict
"""
import glob
import numpy as np
import dataloaders.augutils as myaug
import torch
import random
import os
import copy
import platform
import json
import re
import cv2
from dataloaders.common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList
from util.consts import IMG_SIZE

class SuperpixelDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, image_size, transforms, scan_per_load, num_rep = 2, min_fg = '', nsup = 1, fix_length = None, tile_z_dim = 3, exclude_list = [], train_list = [], superpix_scale = 'SMALL', norm_mean=None, norm_std=None, supervised_train=False, use_3_slices=False, **kwargs):
        """
        Pseudolabel dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            nsup:               number of scans used as support. currently idle for superpixel dataset
            transforms:         data transform (augmentation) function
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            num_rep:            Number of augmentation applied for a same pseudolabel
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            superpix_scale:     config of superpixels
        """
        super(SuperpixelDataset, self).__init__(base_dir)

        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.pseu_label_name = DATASET_INFO[which_dataset]['PSEU_LABEL_NAME']
        self.real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

        self.image_size = image_size
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.supervised_train = supervised_train
        if self.supervised_train and len(train_list) == 0:
            raise Exception('Please provide training labels')
        # assert mode == 'train'
        self.fix_length = fix_length
        if self.supervised_train:
            # self.nclass = len(self.real_label_name)
            self.nclass = len(self.pseu_label_name)
        else:
            self.nclass = len(self.pseu_label_name)
        self.num_rep = num_rep
        self.tile_z_dim = tile_z_dim
        self.use_3_slices = use_3_slices
        if tile_z_dim > 1 and self.use_3_slices:
            raise Exception("tile_z_dim and use_3_slices shouldn't be used together")

        # find scans in the data folder
        self.nsup = nsup
        self.base_dir = base_dir
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x)))

        # experiment configs
        self.exclude_lbs = exclude_list
        self.train_list = train_list
        self.superpix_scale = superpix_scale
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')
        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)
        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold
        self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()], ct_mean=norm_mean, ct_std=norm_std)

        if self.is_train:
            if scan_per_load > 0: # if the dataset is too large, only reload a subset in each sub-epoch
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        else:
            raise Exception
        
        self.use_clahe = False
        if kwargs['use_clahe']:
            self.use_clahe = True
            clip_limit = 4.0 if self.img_modality == 'MR' else 2.0
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(7,7))
            
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()

        print("###### Initial scans loaded: ######")
        print(self.pid_curr_load)

    def get_scanids(self, mode, idx_split):
        """
        Load scans by train-test split
        leaving one additional scan as the support scan. if the last fold, taking scan 0 as the additional one
        Args:
            idx_split: index for spliting cross-validation folds
        """
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a only portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing notiong")
            return -1

        del self.actual_dataset
        del self.info_by_scan

        self.pid_curr_load = np.random.choice( self.scan_ids, size = self.scan_per_load, replace = False )
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid  = os.path.join(self.base_dir, f'superpix-{self.superpix_scale}_{curr_id}.nii.gz')
            _gt_lb_fid = os.path.join(self.base_dir, f'label_{curr_id}.nii.gz')

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            curr_dict["gt_lbs_fid"] = _gt_lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def read_dataset(self):
        """
        Read images into memory and store them in 2D
        Build tables for the position of an individual 2D slice in the entire dataset
        """
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
            # read connected graph of labels
            if self.use_clahe:
                # img = nself.clahe.apply(img.astype(np.uint8))
                if self.img_modality == 'MR':
                    img = np.stack([((slice - slice.min()) / (slice.max() - slice.min())) * 255 for slice in img], axis=0)
                img = np.stack([self.clahe.apply(slice.astype(np.uint8)) for slice in img], axis=0)

            img = img.transpose(1,2,0)
            self.info_by_scan[scan_id] = _info
            
            img = np.float32(img)
            img = self.norm_func(img)
            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]

            if self.supervised_train:
                lb = read_nii_bysitk(itm["gt_lbs_fid"])
            else:
                lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1,2,0)
            lb = np.int32(lb)

            # resize img and lb to self.image_size
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            lb = cv2.resize(lb, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
             
            # format of slices: [axial_H x axial_W x Z]
            if self.supervised_train:
                # remove all images that dont have the training labels
                del_indices = [i for i in range(img.shape[-1]) if not np.any(np.isin(lb[..., i], self.train_list))]
                # create an new img and lb without indices in del_indices
                new_img = np.zeros((img.shape[0], img.shape[1], img.shape[2] - len(del_indices)))
                new_lb = np.zeros((lb.shape[0], lb.shape[1], lb.shape[2] - len(del_indices)))
                new_img = img[..., ~np.isin(np.arange(img.shape[-1]), del_indices)]
                new_lb = lb[..., ~np.isin(np.arange(lb.shape[-1]), del_indices)]
                        
                img = new_img
                lb = new_lb
                a = [i for i in range(img.shape[-1]) if lb[...,i].max() == 0]

            nframes = img.shape[-1]
            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            # re-organize 3D images into 2D slices and record essential information for each slice
            out_list.append( {"img": img[..., 0: 1],
                           "lb":lb[..., 0: 0 + 1],
                           "sup_max_cls": lb[..., 0: 0 + 1].max(),
                           "is_start": True,
                           "is_end": False,
                           "nframe": nframes,
                           "scan_id": scan_id,
                           "z_id":0,
                           })

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii + 1],
                           "is_start": False,
                           "is_end": False,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": nframes,
                           "scan_id": scan_id,
                           "z_id": ii,
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last slice of a 3D volume
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii+ 1],
                           "is_start": False,
                            "is_end": True,
                           "sup_max_cls": lb[..., ii: ii + 1].max(),
                           "nframe": nframes,
                           "scan_id": scan_id,
                           "z_id": ii,
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

        return out_list

    def read_classfiles(self):
        """
        Load the scan-slice-class indexing file
        """
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map
    
    def get_superpixels_similarity(self, sp1, sp2):
        pass

    def supcls_pick_binarize(self, super_map, sup_max_cls, bi_val=None, conn_graph=None, img=None):
        if bi_val is None:
            # bi_val = np.random.randint(1, sup_max_cls)
            bi_val = random.choice(list(np.unique(super_map)))
        if conn_graph is not None and img is not None:
            # get number of neighbors of bi_val
            neighbors = conn_graph[bi_val]
            # pick a random number of neighbors and merge them
            n_neighbors = np.random.randint(0, len(neighbors))
            try:
                neighbors = random.sample(neighbors, n_neighbors)
            except TypeError:
                neighbors = []
            # merge neighbors
            super_map = np.where(np.isin(super_map, neighbors), bi_val, super_map)
        return np.float32(super_map == bi_val)
    
    def supcls_pick(self, super_map):
        return random.choice(list(np.unique(super_map)))
    
    def get_3_slice_adjacent_image(self, image_t, index):
        curr_dict = self.actual_dataset[index]
        prev_image = np.zeros_like(image_t) 
        
        if index > 1 and not curr_dict["is_start"]:
            prev_dict = self.actual_dataset[index - 1]
            prev_image = prev_dict["img"]
            
        next_image = np.zeros_like(image_t)
        if index < len(self.actual_dataset) - 1 and not curr_dict["is_end"]:
            next_dict = self.actual_dataset[index + 1] 
            next_image = next_dict["img"]
            
        image_t = np.concatenate([prev_image, image_t, next_image], axis=-1)

        return image_t

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        sup_max_cls = curr_dict['sup_max_cls']
        if sup_max_cls < 1:
            return self.__getitem__(index + 1)

        image_t = curr_dict["img"]
        label_raw = curr_dict["lb"]

        if self.use_3_slices:
            image_t = self.get_3_slice_adjacent_image(image_t, index)
             
        for _ex_cls in self.exclude_lbs:
            if curr_dict["z_id"] in self.tp1_cls_map[self.real_label_name[_ex_cls]][curr_dict["scan_id"]]: # if using setting 1, this slice need to be excluded since it contains label which is supposed to be unseen
                return self.__getitem__(torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))
        
        if self.supervised_train:
            superpix_label = -1
            label_t = np.float32(label_raw)
            
            lb_id = random.choice(list(set(np.unique(label_raw)) & set(self.train_list)))
            label_t[label_t != lb_id] = 0
            label_t[label_t == lb_id] = 1
            
        else:
            superpix_label = self.supcls_pick(label_raw)
            label_t = np.float32(label_raw == superpix_label)

        pair_buffer = []

        comp = np.concatenate( [image_t, label_t], axis = -1 )
        
        for ii in range(self.num_rep):
            if self.transforms is not None:
                img, lb = self.transforms(comp, c_img = image_t.shape[-1], c_label = 1, nclass = self.nclass,  is_train = True, use_onehot = False)
            else:
                img, lb = comp[:, :, 0:1], comp[:, :, 1:2]
            # if ii % 2 == 0:
            #     label_raw = lb
            # lb = lb == superpix_label

            img = torch.from_numpy( np.transpose( img, (2, 0, 1)) ).float()
            lb  = torch.from_numpy( lb.squeeze(-1)).float()

            img = img.repeat( [ self.tile_z_dim, 1, 1] )

            is_start = curr_dict["is_start"]
            is_end = curr_dict["is_end"]
            nframe = np.int32(curr_dict["nframe"])
            scan_id = curr_dict["scan_id"]
            z_id    = curr_dict["z_id"]

            sample = {"image": img,
                    "label":lb,
                    "is_start": is_start,
                    "is_end": is_end,
                    "nframe": nframe,
                    "scan_id": scan_id,
                    "z_id": z_id
                    }

            # Add auxiliary attributes
            if self.aux_attrib is not None:
                for key_prefix in self.aux_attrib:
                    # Process the data sample, create new attributes and save them in a dictionary
                    aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                    for key_suffix in aux_attrib_val:
                        # one function may create multiple attributes, so we need suffix to distinguish them
                        sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]
            pair_buffer.append(sample)

        support_images = []
        support_mask = []
        support_class = []

        query_images = []
        query_labels = []
        query_class = []

        for idx, itm in enumerate(pair_buffer):
            if idx % 2 == 0:
                support_images.append(itm["image"])
                support_class.append(1) # pseudolabel class
                support_mask.append(  self.getMaskMedImg( itm["label"], 1, [1]  ))
            else:
                query_images.append(itm["image"])
                query_class.append(1)
                query_labels.append(  itm["label"])

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'superpix_label': superpix_label, 
            'superpix_label_raw': label_raw[:,:,0],
            'support_mask': [support_mask],
            'query_images': query_images, #
            'query_labels': query_labels,
            'scan_id': scan_id,
            'z_id': z_id,
            'nframe': nframe,
        }


    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length != None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
            class_ids:      all class id in this episode
        """
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}
