"""
Training the model
Extended from original implementation of ALPNet.
"""
from scipy.ndimage import distance_transform_edt as eucl_distance
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from models.grid_proto_fewshot import FewShotSeg
from torch.utils.tensorboard import SummaryWriter
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex
from tqdm.auto import tqdm
# import Tensor
from torch import Tensor
from typing import List, Tuple, Union, cast, Iterable, Set, Any, Callable, TypeVar

def get_dice_loss(prediction: torch.Tensor, target: torch.Tensor, smooth=1.0):
    '''
    prediction: (B, 1, H, W)
    target: (B, H, W)
    '''
    if prediction.shape[1] > 1:
        # use only the foreground prediction
        prediction = prediction[:, 1, :, :]
    prediction = torch.sigmoid(prediction)
    intersection = (prediction * target).sum(dim=(-2, -1))
    union = prediction.sum(dim=(-2, -1)) + target.sum(dim=(1, 2)) + smooth

    dice = (2.0 * intersection + smooth) / union
    dice_loss = 1.0 - dice.mean()

    return dice_loss


def get_train_transforms(_config):
    tr_transforms = myaug.transform_with_label(
        {'aug': myaug.get_aug(_config['which_aug'], _config['input_size'][0])})
    return tr_transforms

    
def get_dataset_base_name(data_name):
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'CHAOST2_Superpix_672':
        baseset_name = 'CHAOST2'
    elif data_name == 'SABS_Superpix_448':
        baseset_name = 'SABS'
    elif data_name == 'SABS_Superpix_672':
        baseset_name = 'SABS'
    elif 'lits' in data_name.lower():
        baseset_name = 'LITS17'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    return baseset_name

def get_nii_dataset(_config):
    data_name = _config['dataset']
    baseset_name = get_dataset_base_name(data_name)
    tr_transforms = get_train_transforms(_config)
    tr_parent = SuperpixelDataset(  # base dataset
        which_dataset=baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split=_config['eval_fold'],
        mode='train',
        # dummy entry for superpixel dataset
        min_fg=str(_config["min_fg_data"]),
        image_size=_config["input_size"][0],
        transforms=tr_transforms,
        nsup=_config['task']['n_shots'],
        scan_per_load=_config['scan_per_load'],
        exclude_list=_config["exclude_cls_list"],
        superpix_scale=_config["superpix_scale"],
        fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
            data_name == 'CHAOST2_Superpix') else _config["max_iters_per_load"],
        use_clahe=_config['use_clahe'],
        use_3_slices=_config["use_3_slices"],
        tile_z_dim=3 if not _config["use_3_slices"] else 1,
    )
    
    return tr_parent


def get_dataset(_config):
    return get_nii_dataset(_config)


@ex.automain
def main(_run, _config, _log):
    precision = torch.float32
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])

    writer = SummaryWriter(f'{_run.observers[0].dir}/logs')
    _log.info('###### Create model ######')
    if _config['reload_model_path'] != '':
        _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    else:
        _config['reload_model_path'] = None
    model = FewShotSeg(image_size=_config['input_size'][0], pretrained_path=_config['reload_model_path'], cfg=_config['model'])

    model = model.to(device, precision)
    model.train()
    
    _log.info('###### Load data ######')
    data_name = _config['dataset']
    tr_parent = get_dataset(_config)

    # dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=_config['lr'], eps=1e-5)
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(
        optimizer, milestones=_config['lr_milestones'],  gamma=_config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config['ignore_label'], weight=my_weight)

    i_iter = 0  # total number of iteration
    # number of times for reloading
    n_sub_epoches = max(1, _config['n_steps'] // _config['max_iters_per_load'], _config["epochs"])
    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    epoch_losses = []
    for sub_epoch in range(n_sub_epoches):
        _log.info(
            f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        pbar = tqdm(trainloader)
        optimizer.zero_grad()
        for idx, sample_batched in enumerate(tqdm(trainloader)):
            losses = []
            i_iter += 1
            support_images = [[shot.to(device, precision) for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().to(device, precision) for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().to(device, precision) for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.to(device, precision)
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().to(device) for query_label in sample_batched['query_labels']], dim=0)

            loss = 0.0
            try:
                out = model(support_images, support_fg_mask, support_bg_mask,
                        query_images, isval=False, val_wsize=None)
                query_pred, align_loss, _, _, _, _, _ = out
                # pred = np.array(query_pred.argmax(dim=1)[0].cpu())
            except Exception as e:
                print(f'faulty batch detected, skip: {e}')
                # offload cuda memory
                del support_images, support_fg_mask, support_bg_mask, query_images, query_labels
                continue
                 
            query_loss = criterion(query_pred.float(), query_labels.long())
            loss += query_loss + align_loss
            pbar.set_postfix({'loss': loss.item()})
            loss.backward()
            if (idx + 1) % _config['grad_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            losses.append(loss.item())
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)

            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                writer.add_scalar('loss', loss, i_iter)
                writer.add_scalar('query_loss', query_loss, i_iter)
                writer.add_scalar('align_loss', align_loss, i_iter)

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                print(
                    f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss},')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            if (i_iter - 1) >= _config['n_steps']:
                break  # finish up
        epoch_losses.append(np.mean(losses))
        print(f"Epoch {sub_epoch} loss: {np.mean(losses)}")
