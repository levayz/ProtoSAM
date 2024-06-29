"""
Manually labeled dataset
TODO: 
1. Merge with superpixel dataset
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
from dataloaders.common import BaseDataset, Subset, ValidationDataset
# from common import BaseDataset, Subset
from dataloaders.dataset_utils import*
from pdb import set_trace
from util.utils import CircularList
from util.consts import IMG_SIZE

MODE_DEFAULT = "default"
MODE_FULL_SCAN = "full_scan"

class ManualAnnoDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, image_size, transforms, scan_per_load, min_fg = '', fix_length = None, tile_z_dim = 3, nsup = 1, exclude_list = [], extern_normalize_func = None, **kwargs):
        """
        Manually labeled dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'. 
            transforms:         data transform (augmentation) function
            min_fg:             minimum number of positive pixels in a 2D slice, mainly for stablize training when trained on manually labeled dataset
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            nsup:               number of support scans
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            extern_normalize_function:  normalization function used for data pre-processing  
        """
        super(ManualAnnoDataset, self).__init__(base_dir)
        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']
        self.image_size = image_size
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.fix_length = fix_length
        self.all_label_names = self.label_name
        self.nclass = len(self.label_name)
        self.tile_z_dim = tile_z_dim
        self.base_dir = base_dir
        self.nsup = nsup
        self.img_pids = [ re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz") ]
        self.img_pids = CircularList(sorted( self.img_pids, key = lambda x: int(x))) # make it circular for the ease of spliting folds
        if 'use_clahe' not in kwargs:
            self.use_clahe = False
        else:
            self.use_clahe = kwargs['use_clahe']
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
           
        self.use_3_slices = kwargs["use_3_slices"] if 'use_3_slices' in kwargs else False 
        if self.use_3_slices:
            self.tile_z_dim=1
        
        self.get_item_mode = MODE_DEFAULT
        if 'get_item_mode' in kwargs:
            self.get_item_mode = kwargs['get_item_mode']

        self.exclude_lbs = exclude_list
        if len(exclude_list) > 0:
            print(f'###### Dataset: the following classes has been excluded {exclude_list}######')

        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split) # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)

        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids() # information of scans of the entire fold

        if extern_normalize_func is not None: # helps to keep consistent between training and testing dataset.
            self.norm_func = extern_normalize_func
            print(f'###### Dataset: using external normalization statistics ######')
        else:
            self.norm_func = get_normalize_op(self.img_modality, [ fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])
            print(f'###### Dataset: using normalization statistics calculated from loaded data ######')

        if self.is_train:
            if scan_per_load > 0: # buffer needed
                self.pid_curr_load = np.random.choice( self.scan_ids, replace = False, size = self.scan_per_load)
            else: # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
            self.potential_support_sid = []
        else:
            raise Exception
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()
        self.update_subclass_lookup()

    def get_scanids(self, mode, idx_split):
        val_ids  = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        self.potential_support_sid = val_ids[-self.nsup:] # this is actual file scan id, not index
        if mode == 'train':
            return [ ii for ii in self.img_pids if ii not in val_ids ]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a portion of the entire dataset, if the dataset is too large
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
            _lb_fid  = os.path.join(self.base_dir, f'label_{curr_id}.nii.gz')

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def read_dataset(self):
        """
        Build index pointers to individual slices
        Also keep a look-up table from scan_id, slice to index
        """
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out

            img = img.transpose(1,2,0)

            self.info_by_scan[scan_id] = _info

            if self.use_clahe:
                img = np.stack([self.clahe.apply(slice.astype(np.uint8)) for slice in img], axis=0)
            
            img = np.float32(img)
            img = self.norm_func(img)

            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]

            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1,2,0)

            lb = np.float32(lb)

            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            lb = cv2.resize(lb, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice

            # write the beginning frame
            out_list.append( {"img": img[..., 0: 1],
                           "lb":lb[..., 0: 0 + 1],
                           "is_start": True,
                           "is_end": False,
                           "nframe": img.shape[-1],
                           "scan_id": scan_id,
                           "z_id":0})

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii + 1],
                           "is_start": False,
                           "is_end": False,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last frame, note the is_end flag
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii+ 1],
                           "is_start": False,
                           "is_end": True,
                           "nframe": -1,
                           "scan_id": scan_id,
                           "z_id": ii
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

        return out_list

    def read_classfiles(self):
        with open(   os.path.join(self.base_dir, f'classmap_{self.min_fg}.json') , 'r' ) as fopen:
            cls_map =  json.load( fopen)
            fopen.close()

        with open(   os.path.join(self.base_dir, 'classmap_1.json') , 'r' ) as fopen:
            self.tp1_cls_map =  json.load( fopen)
            fopen.close()

        return cls_map

    def __getitem__(self, index):
        if self.get_item_mode == MODE_DEFAULT:
            return self.__getitem_default__(index)
        elif self.get_item_mode == MODE_FULL_SCAN:
            return self.__get_ct_scan___(index)
        else:
            raise NotImplementedError("Unknown mode")
            

    def __get_ct_scan___(self, index):
        scan_n = index % len(self.scan_z_idx)
        scan_id = list(self.scan_z_idx.keys())[scan_n]
        scan_slices = self.scan_z_idx[scan_id]
        
        scan_imgs = np.concatenate([self.actual_dataset[_idx]["img"] for _idx in scan_slices], axis = -1).transpose(2, 0, 1)
        
        scan_lbs = np.concatenate([self.actual_dataset[_idx]["lb"] for _idx in scan_slices], axis = -1).transpose(2, 0, 1)
        
        scan_imgs = np.float32(scan_imgs)
        scan_lbs  = np.float32(scan_lbs)
        
        scan_imgs = torch.from_numpy(scan_imgs).unsqueeze(0)
        scan_lbs = torch.from_numpy(scan_lbs)
        
        if self.tile_z_dim:
            scan_imgs = scan_imgs.repeat(self.tile_z_dim, 1, 1, 1)
            assert scan_imgs.ndimension() == 4, f'actual dim {scan_imgs.ndimension()}'
        
        # # reshape to C, D, H, W
        # scan_imgs = scan_imgs.permute(1, 0, 2, 3)
        # scan_lbs = scan_lbs.permute(1, 0, 2, 3)
         
        sample = {"image": scan_imgs,
                "label":scan_lbs,
                "scan_id": scan_id,
                }
        
        return sample
            
    
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


    def __getitem_default__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train:
            if len(self.exclude_lbs) > 0:
                for _ex_cls in self.exclude_lbs:
                    if curr_dict["z_id"] in self.tp1_cls_map[self.label_name[_ex_cls]][curr_dict["scan_id"]]: # this slice need to be excluded since it contains label which is supposed to be unseen
                        return self.__getitem__(index + torch.randint(low = 0, high = self.__len__() - 1, size = (1,)))

            comp = np.concatenate( [curr_dict["img"], curr_dict["lb"]], axis = -1 )
            if self.transforms is not None:
                img, lb = self.transforms(comp, c_img = 1, c_label = 1, nclass = self.nclass, use_onehot = False)
            else:
                raise Exception("No transform function is provided")
            
        else:
            img = curr_dict['img']
            lb = curr_dict['lb']


        img = np.float32(img)
        lb = np.float32(lb).squeeze(-1) # NOTE: to be suitable for the PANet structure
        if self.use_3_slices:
            img = self.get_3_slice_adjacent_image(img, index)
        
        img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
        lb  = torch.from_numpy( lb)

        if self.tile_z_dim:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

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

        return sample

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.get_item_mode == MODE_FULL_SCAN:
            return len(self.scan_z_idx)
        
        if self.fix_length != None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def update_subclass_lookup(self):
        """
        Updating the class-slice indexing list
        Args:
            [internal] overall_slice_by_cls:
                {
                    class1: {pid1: [slice1, slice2, ....],
                                pid2: [slice1, slice2]},
                                ...}
                    class2:
                    ...
                }
        out[internal]:
                {
                    class1: [ idx1, idx2, ...  ],
                    class2: [ idx1, idx2, ...  ],
                    ...
                }

        """
        # delete previous ones if any
        assert self.overall_slice_by_cls is not None

        if not hasattr(self, 'idx_by_class'):
             self.idx_by_class = {}
        # filter the new one given the actual list
        for cls in self.label_name:
            if cls not in self.idx_by_class.keys():
                self.idx_by_class[cls] = []
            else:
                del self.idx_by_class[cls][:]
        for cls, dict_by_pid in self.overall_slice_by_cls.items():
            for pid, slice_list in dict_by_pid.items():
                if pid not in self.pid_curr_load:
                    continue
                self.idx_by_class[cls] += [ self.scan_z_idx[pid][_sli] for _sli in slice_list ]
        print("###### index-by-class table has been reloaded ######")

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask. Used when getting the support
        """
        # Dense Mask
        fg_mask = torch.where(label == class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id,
                              torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0

        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}

    def subsets(self, sub_args_lst=None):
        """
        Override base-class subset method
        Create subsets by scan_ids

        output: list [[<fid in each class>] <class1>, <class2>     ]
        """

        if sub_args_lst is not None:
            subsets = []
            ii = 0
            for cls_name, index_list in self.idx_by_class.items():
                subsets.append( Subset(dataset = self, indices = index_list, sub_attrib_args = sub_args_lst[ii])  )
                ii += 1
        else:
            subsets = [Subset(dataset=self, indices=index_list) for _, index_list in self.idx_by_class.items()]
        return subsets

    def get_support(self, curr_class: int, class_idx: list, scan_idx: list, npart: int):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        assert curr_class != 0; assert 0 not in class_idx
        # assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        # print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        # print(f'###### Parts percentage: {pcts} ######')

        # norm_func = get_normalize_op(modality='MR', fids=None)
        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                _zlist = self.tp1_cls_map[self.label_name[curr_class]][_scan_id] # list of indices
                _zid = _zlist[int(pcts[_part] * len(_zlist))]
                _glb_idx = self.scan_z_idx[_scan_id][_zid]

                # almost copy-paste __getitem__ but no augmentation
                curr_dict = self.actual_dataset[_glb_idx]
                img = curr_dict['img']
                lb = curr_dict['lb']

                if self.use_3_slices:
                    prev_image = np.zeros_like(img) 
                    if _glb_idx > 1 and not curr_dict["is_start"]:
                        prev_dict = self.actual_dataset[_glb_idx - 1]
                        prev_image = prev_dict["img"]
                
                    next_image = np.zeros_like(img)    
                    if _glb_idx < len(self.actual_dataset) - 1 and not curr_dict["is_end"]:
                        next_dict = self.actual_dataset[_glb_idx + 1] 
                        next_image = next_dict["img"]
                
                    img = np.concatenate([prev_image, img, next_image], axis=-1)
                
                img = np.float32(img)
                lb = np.float32(lb).squeeze(-1) # NOTE: to be suitable for the PANet structure

                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb )

                if self.tile_z_dim:
                    img = img.repeat( [ self.tile_z_dim, 1, 1] )
                    assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                is_start    = curr_dict["is_start"]
                is_end      = curr_dict["is_end"]
                nframe      = np.int32(curr_dict["nframe"])
                scan_id     = curr_dict["scan_id"]
                z_id        = curr_dict["z_id"]

                sample = {"image": img,
                        "label":lb,
                        "is_start": is_start,
                        "inst": None,
                        "scribble": None,
                        "is_end": is_end,
                        "nframe": nframe,
                        "scan_id": scan_id,
                        "z_id": z_id
                        }

                concat_buffer.append(sample)
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),

                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            support_mask.append(  self.getMaskMedImg( itm["label"], curr_class, class_idx  ))

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
        }

    def get_support_scan(self, curr_class: int, class_idx: list, scan_idx: list):
        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        # print(f'###### Using {len(scan_idx)} shot evaluation!')
        scan_slices = self.scan_z_idx[self.potential_support_sid[0]]
        scan_imgs = np.concatenate([self.actual_dataset[_idx]["img"] for _idx in scan_slices], axis = -1).transpose(2, 0, 1)
        
        scan_lbs = np.concatenate([self.actual_dataset[_idx]["lb"] for _idx in scan_slices], axis = -1).transpose(2, 0, 1)
        # binarize the labels
        scan_lbs[scan_lbs != curr_class] = 0
        scan_lbs[scan_lbs == curr_class] = 1
         
        scan_imgs = torch.from_numpy(np.float32(scan_imgs)).unsqueeze(0)
        scan_lbs  = torch.from_numpy(np.float32(scan_lbs))
        
        if self.tile_z_dim:
            scan_imgs = scan_imgs.repeat(self.tile_z_dim, 1, 1, 1)
            assert scan_imgs.ndimension() == 4, f'actual dim {scan_imgs.ndimension()}'
            
        # reshape to C, D, H, W
        sample = {"scan": scan_imgs,
                "labels":scan_lbs,
        }
        
        return sample
        
    
    def get_support_multiple_classes(self, classes: list, scan_idx: list, npart: int, use_3_slices=False):
        """
        getting (probably multi-shot) support set for evaluation
        sample from 50% (1shot) or 20 35 50 65 80 (5shot)
        Args:
            curr_cls:       current class to segment, starts from 1
            class_idx:      a list of all foreground class in nways, starts from 1
            npart:          how may chunks used to split the support
            scan_idx:       a list, indicating the current **i_th** (note this is idx not pid) training scan
        being served as support, in self.pid_curr_load
        """
        assert npart % 2 == 1
        # assert curr_class != 0; assert 0 not in class_idx
        # assert not self.is_train

        self.potential_support_sid = [self.pid_curr_load[ii] for ii in scan_idx ]
        # print(f'###### Using {len(scan_idx)} shot evaluation!')

        if npart == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (npart * 2)
            part_interval = (1.0 - 1.0 / npart) / (npart - 1)
            pcts = [ half_part + part_interval * ii for ii in range(npart) ]

        # print(f'###### Parts percentage: {pcts} ######')

        out_buffer = [] # [{scanid, img, lb}]
        for _part in range(npart):
            concat_buffer = [] # for each fold do a concat in image and mask in batch dimension
            for scan_order in scan_idx:
                _scan_id = self.pid_curr_load[ scan_order ]
                print(f'Using scan {_scan_id} as support!')

                # for _pc in pcts:
                zlist = []
                for curr_class in classes:
                    zlist.append(self.tp1_cls_map[self.label_name[curr_class]][_scan_id]) # list of indices
                # merge all the lists in zlist and keep only the unique elements
                # _zlist = sorted(list(set([item for sublist in zlist for item in sublist])))
                # take only the indices that appear in all of the sublist
                _zlist = sorted(list(set.intersection(*map(set, zlist))))
                _zid = _zlist[int(pcts[_part] * len(_zlist))]
                _glb_idx = self.scan_z_idx[_scan_id][_zid]

                # almost copy-paste __getitem__ but no augmentation
                curr_dict = self.actual_dataset[_glb_idx]
                img = curr_dict['img']
                lb = curr_dict['lb']
                
                if use_3_slices:
                    prev_image = np.zeros_like(img) 
                    if _glb_idx > 1 and not curr_dict["is_start"]:
                        prev_dict = self.actual_dataset[_glb_idx - 1]
                        assert prev_dict["scan_id"] == curr_dict["scan_id"]
                        assert prev_dict["z_id"] == curr_dict["z_id"] - 1
                        prev_image = prev_dict["img"]
                
                    next_image = np.zeros_like(img)    
                    if _glb_idx < len(self.actual_dataset) - 1 and not curr_dict["is_end"]:
                        next_dict = self.actual_dataset[_glb_idx + 1] 
                        assert next_dict["scan_id"] == curr_dict["scan_id"]
                        assert next_dict["z_id"] == curr_dict["z_id"] + 1
                        next_image = next_dict["img"]
                
                    img = np.concatenate([prev_image, img, next_image], axis=-1)
                
                img = np.float32(img)
                lb = np.float32(lb).squeeze(-1) # NOTE: to be suitable for the PANet structure
                # zero all labels that are not in the classes arg
                mask = np.zeros_like(lb)
                for cls in classes:
                    mask[lb == cls] = 1
                lb[~mask.astype(np.bool)] = 0
                
                img = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
                lb  = torch.from_numpy( lb )

                if self.tile_z_dim:
                    img = img.repeat( [ self.tile_z_dim, 1, 1] )
                    assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

                is_start    = curr_dict["is_start"]
                is_end      = curr_dict["is_end"]
                nframe      = np.int32(curr_dict["nframe"])
                scan_id     = curr_dict["scan_id"]
                z_id        = curr_dict["z_id"]

                sample = {"image": img,
                        "label":lb,
                        "is_start": is_start,
                        "inst": None,
                        "scribble": None,
                        "is_end": is_end,
                        "nframe": nframe,
                        "scan_id": scan_id,
                        "z_id": z_id
                        }

                concat_buffer.append(sample)
            out_buffer.append({
                "image": torch.stack([itm["image"] for itm in concat_buffer], dim = 0),
                "label": torch.stack([itm["label"] for itm in concat_buffer], dim = 0),

                })

            # do the concat, and add to output_buffer

        # post-processing, including keeping the foreground and suppressing background.
        support_images = []
        support_mask = []
        support_class = []
        for itm in out_buffer:
            support_images.append(itm["image"])
            support_class.append(curr_class)
            # support_mask.append(  self.getMaskMedImg( itm["label"], curr_class, class_idx  ))
            support_mask.append(itm["label"])

        return {'class_ids': [support_class],
            'support_images': [support_images], #
            'support_mask': [support_mask],
            'scan_id': scan_id
        }

def get_nii_dataset(config, image_size, **kwargs):
    organ_mapping = {
        "sabs":{
            "rk": 2,
            "lk": 3,
            "liver": 6,
            "spleen": 1
        },
        "chaost2":{
            "liver": 1,
            "rk": 2,
            "lk": 3,
            "spleen": 4
    }}
    
    transforms = None
    data_name = config['dataset']
    if data_name == 'SABS_Superpix' or data_name == 'SABS_Superpix_448' or data_name == 'SABS_Superpix_672':
        baseset_name = 'SABS'
        max_label = 13
        modality="CT"
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2_Superpix_672':
        baseset_name = 'CHAOST2'
        max_label = 4
        modality="MR"
    elif 'lits' in data_name.lower():
        baseset_name = 'LITS17'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')
     
    # norm_func = get_normalize_op(modality=modality, fids=None) # TODO add global statistics
    # norm_func = None
    
    test_label = organ_mapping[baseset_name.lower()][config["curr_cls"]]
    base_dir = config['path'][data_name]['data_dir']

    testdataset = ManualAnnoDataset(which_dataset=baseset_name,
                                    base_dir=base_dir,
                                    idx_split = config['eval_fold'],
                                    mode = 'val',
                                    scan_per_load = 1,
                                    transforms=transforms,
                                    min_fg=1,
                                    nsup = config["task"]["n_shots"],
                                    fix_length=None,
                                    image_size=image_size,
                                    # extern_normalize_func=norm_func
                                    **kwargs)
    
    testdataset = ValidationDataset(testdataset, test_classes = [test_label], npart = config["task"]["npart"])
    testdataset.set_curr_cls(test_label)
    
    traindataset = None # TODO make this the support set later

    return traindataset, testdataset