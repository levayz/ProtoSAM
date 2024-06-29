"""Util functions
Extended from original PANet code
TODO: move part of dataset configurations to data_utils
"""
import random
import torch
import numpy as np
import operator
import cv2
import matplotlib.pyplot as plt
import kneed
import urllib
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
import torchvision.transforms.functional as F


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


def reverse_tensor(tensor, original_h, original_w, degrees):
    """
    tensor: tensor of shape (B, C, H, W) to be rotated
    original_h: int - original height of the tensor (after it was rotated)
    original_w: int - original width of the tensor (after it was rotated)
    degrees: int or float - angle in degrees couterclockwise
    """
    _, _, h, w = tensor.shape # this is the shape that we want to return to
    if tensor.shape[-2:] != (original_h, original_w):
        tensor = F.resize(tensor, (original_h, original_w), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        # print("interpolating")
        
    rotated_tensor = F.rotate(tensor, degrees, expand=False)
    # remove the black padding
    h_remove = abs(h - original_h) // 2
    w_remove = abs(w - original_w) // 2
    if h_remove > 0 and w_remove > 0:
        rotated_tensor = rotated_tensor[:, :, h_remove:-h_remove, w_remove:-w_remove]
    
    return rotated_tensor


def need_softmax(tensor, dim=1):
    return not torch.all(torch.isclose(tensor.sum(dim=dim), torch.ones_like(tensor.sum(dim=dim))) & (tensor >= 0))


def rotate_tensor_no_crop(image_tensor, degrees):
    """
    image_tensor: tensor of shape (B, C, H, W)
    degrees: int or float - angle in degrees couterclockwise
    returns: tensor of shape (B, C, H, W) rotated by degrees,
    """
    if degrees == 0:
        return image_tensor, image_tensor.shape[-2:]
    
    b, c, h, w = image_tensor.shape
    rotated_tensor = F.rotate(image_tensor, degrees, expand=True)
    
    interpolation_mode = F.InterpolationMode.BILINEAR
    if c == 1:
        interpolation_mode = F.InterpolationMode.NEAREST
    resized_tensor = F.resize(rotated_tensor, (h, w), interpolation=interpolation_mode, antialias=True)
         
    return resized_tensor, rotated_tensor.shape[-2:]

def plot_dinov2_fts(img_fts, title="debug/img_fts.png"):
    """
    Using PCA to reduce img_fts to 2D and plot it
    Args:
    img_fts: (B, C, H, W)
    """
    if isinstance(img_fts, torch.Tensor):
        img_fts = img_fts.cpu().detach().numpy()

    B, C, H, W = img_fts.shape
    
    img_fts_reshaped = img_fts.transpose(0, 2, 3, 1).reshape(-1, C)
    
    # Apply PCA to reduce dimensionality from C to 1
    pca = PCA(n_components=1)
    img_fts_pca = pca.fit_transform(img_fts_reshaped)
    
    # Reshape back to (B, 1, H, W)
    img_fts_reduced = img_fts_pca.reshape(B, H, W, 1).transpose(0, 3, 1, 2)
    
    # Plot the B images
    if B == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_fts_reduced[0, 0])
    else:
        fig, axes = plt.subplots(1, B, figsize=(B*5, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(img_fts_reduced[i, 0])
            # ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(title)
    plt.close(fig)


def move_to_device(dict_obj, device='cuda'):
    for key in dict_obj:
        value = dict_obj[key]
        if isinstance(value, torch.Tensor):
            dict_obj[key] = value.to(device)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    dict_obj[key][i] = item.to(device)


def validation_single_slice(model, support_images, support_fg_mask, support_bg_mask, query_images, _config, q_part=0):
    model.eval()
    
    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

    with torch.no_grad():
        query_pred_logits, _, _, assign_mats, _, _ = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

    query_pred = np.array(query_pred_logits.argmax(dim=1)[0].cpu().detach())
            
    if _config['do_cca']:
        query_pred = cca(query_pred, query_pred_logits)
    
    if _config["debug"]:
        # plot the support images, support fg mask, query image, query pred before cca and query pred after cca
        fig, ax = plt.subplots(3, 2, figsize=(15, 10))
        ax[0,0].imshow(support_images[0][q_part][0,0].cpu().numpy(), cmap='gray')
        ax[0,1].imshow(support_fg_mask[0][q_part][0].cpu().numpy(), cmap='gray')
        ax[1,0].imshow(query_images[0][0][0].cpu().numpy(), cmap='gray')
        ax[1,1].imshow(query_pred_logits.argmax(dim=1)[0].cpu().detach().numpy(), cmap='gray')
        ax[2,0].imshow(query_pred, cmap='gray')
        ax[2,1].imshow(query_pred_logits.argmax(dim=1)[0].cpu().detach().numpy(), cmap='gray')
        # remove all ticks
        for axi in ax.flat:
            axi.set_xticks([])
            axi.set_yticks([])
        fig.savefig("debug/cca_before_after.png")
        plt.close(fig)
    
    model.train()
    return query_pred, query_pred_logits

    
def validation_on_scans(model, curr_lb, support_images, support_fg_mask, support_bg_mask, testloader, te_parent, te_dataset, _config, sup_img_indx=1, save_pred_buffer=None):
    if save_pred_buffer is None:
        save_pred_buffer = {}
    lb_buffer = {}
    conf_buffer = {}
    # sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][sup_img_indx]]]   # way(1) x shot x [B(1) x C x H x W]
    # sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][sup_img_indx]]]
    # sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][sup_img_indx]]]
    for scan_idx, sample_batched in enumerate(testloader):
        print(f"Processing scan: {scan_idx + 1} / {len(testloader)}")
        _scan_id = sample_batched["scan_id"][0]
        if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
            print(f"Skipping support scan: {_scan_id}") # TODO delete
            continue
                    
        outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
        outsize = (_config['input_size'][0], _config['input_size'][1], outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
        _pred = np.zeros( outsize )
        _pred.fill(np.nan)
        conf_buffer[_scan_id] = []

        query_images = sample_batched['image'].cuda()
        z_min = sample_batched['z_min'][0]
        z_max = sample_batched['z_max'][0]
        # create an index list that starts with s_idx goes down to 0, then concat the indices from s_idx + 1 to the end
        # this is to make sure that the most similiar slice is the first one to be processed
        indices = list(range(len(query_images[0])))
        qpart = sup_img_indx 
        for idx, i in enumerate(tqdm(indices)):
            if _config["use_3_slices"]:
                # change the query to 3 slices (-1, 0, 1)
                if i == 0:
                    prev_q = torch.zeros_like(query_images[0, i]).unsqueeze(0)
                else:
                    prev_q = query_images[0, i - 1].unsqueeze(0)
                if i == len(query_images[0]) - 1:
                    next_q = torch.zeros_like(query_images[0, i]).unsqueeze(0)
                else:
                    next_q = query_images[0, i + 1].unsqueeze(0)
                                
                query = torch.cat([prev_q, query_images[0, i].unsqueeze(0), next_q], dim=1)
                            
            else:
                query = query_images[0, i].unsqueeze(0)

            
            query_pred, query_pred_logits = validation_single_slice(model, support_images, support_fg_mask, support_bg_mask, [query], _config, q_part=qpart)
            query_conf = get_confidence_from_logits(query_pred_logits, query_pred)
            conf_buffer[_scan_id].append(query_conf)
            _pred[..., i] = query_pred.copy()
        
        if _config['dataset'] != 'C0':
            lb_buffer[_scan_id] = _pred.transpose(2,0,1)
        else:
            lb_buffer[_scan_id] = _pred
    save_pred_buffer[str(curr_lb)] = lb_buffer
    
    return save_pred_buffer, conf_buffer
            
            
            
def validation(model, curr_lb, testloader, te_parent, te_dataset, _config, support_images, support_fg_mask, support_bg_mask, mar_val_metric_node=None, save_pred_buffer=None, do_validation=False, get_confidence=False):
    model.eval()
    with torch.no_grad():
        curr_scan_count = -1 # counting for current scan
        _lb_buffer = {} # indexed by scan
        _conf_buffer = {} # indexed by scan
        _has_label_buffer = {} # indexed by scan
        last_qpart = 0 # used as indicator for adding result to buffer

        for idx, sample_batched in enumerate(tqdm(testloader)):
            _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
            if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                continue
            if sample_batched["is_start"]:
                ii = 0
                curr_scan_count += 1
                if do_validation:
                    if curr_scan_count > 0:
                        break
                print(f"Processing scan {curr_scan_count + 1} / {len(te_dataset.dataset.pid_curr_load)}")
                _scan_id = sample_batched["scan_id"][0]
                outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                outsize = (te_dataset.dataset.image_size, te_dataset.dataset.image_size, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                _pred = np.zeros( outsize )
                _pred.fill(np.nan)
                _conf_buffer[_scan_id] = []
                _has_label_buffer[_scan_id] = []

            q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
            query_images = [sample_batched['image'].cuda()]
            query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)
            # if not 1 in query_labels:
            #     continue
            # [way, [part, [shot x C x H x W]]] ->
            query_pred, query_pred_logits = validation_single_slice(model, support_images, support_fg_mask, support_bg_mask, query_images, _config, q_part=q_part) 
            _pred[..., ii] = query_pred.copy()
            if 1 in query_labels:
                _has_label_buffer[_scan_id].append(True)
            else:
                _has_label_buffer[_scan_id].append(False)
                
            if get_confidence:
                # calc condfidence from logits and log it in the _conf_buffer
                query_conf = get_confidence_from_logits(query_pred_logits, query_pred)
                _conf_buffer[_scan_id].append(query_conf)
            
            if mar_val_metric_node is not None and ((sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin'])):
                mar_val_metric_node.record(query_pred, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
            else:
                pass

            ii += 1
            # now check data format
            if sample_batched["is_end"]:
                if _config['dataset'] != 'C0':
                    _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                else:
                    _lb_buffer[_scan_id] = _pred

        save_pred_buffer[str(curr_lb)] = _lb_buffer
    
    model.train()
    
    return save_pred_buffer, _conf_buffer, _has_label_buffer


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def save_pred_gt_fig(query_images, query_pred, query_labels, support_images=None, support_labels=None, path="debug/gt_vs_pred.png"):
    fig = plt.figure(figsize=(10, 5 if support_images is None else 10))
    ax1 = fig.add_subplot(2 if support_images is not None else 1, 2, 1)
    ax1.imshow(query_images[0][0, 1].cpu().numpy())
    ax1.imshow(query_labels[0].cpu().numpy(), alpha=0.5)
    ax1.set_title("Ground Truth")
    ax2 = fig.add_subplot(2 if support_images is not None else 1, 2, 2)
    ax2.imshow(query_images[0][0, 1].cpu().numpy())
    ax2.imshow(query_pred, alpha=0.5)
    ax2.set_title("Prediction")
    if support_images is not None:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(support_images[0][0, 1].cpu().numpy())
        ax3.imshow(support_labels[0].cpu().numpy(), alpha=0.5)
        ax3.set_title("Support")
    plt.savefig(path)
    plt.close('all')
    
    
def plot_heatmap_of_probs(probs, image, path=None):
    # normalize image values to be between 0 and 1, assume image doesnt have a specific range
    image = (image - image.min()) / (image.max() - image.min())
    rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rgb_image)
    ax.imshow(probs, alpha=0.5)
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)
    

def plot_3d_bar_probabilities(probabilities, labels, image, path=None):
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid of the x and y coordinates
    x, y = np.meshgrid(np.arange(probabilities.shape[1]), np.arange(probabilities.shape[0]))

    # Flatten the probabilities and labels data and convert them to 1D arrays
    z = probabilities.flatten()
    c = np.where(labels.flatten() == 1, 'g', 'r')

    # normaliize image values to be between 0 and 1, assume image doesnt have a specific range
    image = (image - image.min()) / (image.max() - image.min())
    rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    # ax.imshow(rgb_image, extent=[0, probabilities.shape[1], 0, probabilities.shape[0]], alpha=0.5)

    # Create the 3D bar plot
    ax.plot_surface(x, y, np.zeros_like(x), facecolors=rgb_image)
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=c, alpha=0.3)

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability')

    # Show the plot
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)

# def plot_3d_bar_probabilities(probabilities, labels, path=None):
#     # Create a 3D figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Create a meshgrid of the x and y coordinates
#     x, y = np.meshgrid(np.arange(probabilities.shape[1]), np.arange(probabilities.shape[0]))

#     # Flatten the probabilities and labels data and convert them to 1D arrays
#     z = probabilities.flatten()
#     c = np.where(labels.flatten() == 1, 'g', 'r')

#     # Create the 3D bar plot
#     ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, color=c)

#     # Set the axis labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Probability')

#     # Show the plot
#     if path is not None:
#         fig.savefig(path)
#     else:
#         plt.show()
#     plt.close(fig)


# def sliding_window_confidence_segmentation(query_pred_conf:np.array, window_size=3, threshold=0.5):
#     """
#     query_pred_conf: np.array, shape (B, H, W)
#     """
#     # slice window across the query_pred_conf, if the window has a mean confidence > 0.5, the center pixel is 1, otherwise 0
    
#     pred = np.zeros_like(query_pred_conf)
#     # slice the window
#     for i in range(query_pred_conf.shape[-1] - window_size + 1):
#         for j in range(query_pred_conf.shape[-2] - window_size + 1):
#             window = query_pred_conf[:, i:i+window_size, j:j+window_size]
#             if np.mean(window) > threshold:
#                 pred[:, i+window_size//2, j+window_size//2] = 1
    
#     return pred


def sliding_window_confidence_segmentation(query_pred_conf: np.array, window_size=3, threshold=0.5):
    """
    query_pred_conf: np.array, shape (B, H, W)
    """
    B, H, W = query_pred_conf.shape
    pad = window_size // 2
    padded_conf = np.pad(query_pred_conf, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Calculate the mean in sliding windows
    window_view = np.lib.stride_tricks.sliding_window_view(padded_conf, (B, window_size, window_size))
    mean_values = np.mean(window_view, axis=(-1, -2))

    pred = (mean_values > threshold).astype(int)

    return pred[..., 0]
 
    

def get_confidence_from_logits(query_pred_logits: torch.Tensor):
    query_probs = query_pred_logits.softmax(1)[:,1].flatten(1)
    query_pred = query_probs.clone()
    query_pred[query_probs < 0.5] = 0
    query_pred[query_probs >= 0.5] = 1
    return ((query_probs * query_pred).sum() / (query_pred.sum() + 1e-6)).item()    

def choose_threshold_kneedle(p):
    '''
    p - probabilities of prediction
    '''
    # use kneed to choose the threshold
    # create pdf from x
    n_bins = min(100, len(p))
    hist, bin_edges = np.histogram(p, bins=n_bins)
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    
    x = np.linspace(0, 1, n_bins)
    y = cdf
    # plot x, y in a fig and save the fig
    plt.figure()
    plt.plot(x, y)
    plt.savefig(f'debug/cdf.png')
    plt.figure()
    plt.plot(x, pdf)
    plt.savefig(f'debug/pdf.png')
    plt.close('all')
    kneedle = kneed.KneeLocator(x, y, curve='convex', direction='increasing')
    # get the value at the knee from the bin_edges
    threshold = bin_edges[int(kneedle.knee * n_bins)]
    
    return threshold

    
def plot_cca_output(cca_output):
    for j in range(cca_output[0]):
        if j == 0:
            continue
        plt.figure()
        plt.imshow(cca_output[1] == j)
        plt.savefig(f'debug/cca_{j}.png')
        plt.close('all')


def get_connected_components(query_pred_original, query_pred_logits, return_conf=False):
    """
    get all connected components
    """
    cca_output = cv2.connectedComponentsWithStats(query_pred_original.astype(np.uint8), connectivity=8) # TODO try 8
    
    # plot_cca_output(cca_output)    
    
    if return_conf:
        # calc confidence for each connected component
        cca_conf = {} # conf by id
        query_probs = query_pred_logits.softmax(1)[:,1].cpu().detach().numpy()
        for j in range(cca_output[0]):
            if j == 0:
                cca_conf[0] = 0 # background
                continue
            cca_conf[j] = ((query_probs.flatten() * (cca_output[1] == j).flatten()).sum()  / ((query_pred_original.flatten().sum() + 1e-6))) # take into account the area of the connected component
        
        return cca_output, cca_conf
    
    return cca_output, None

def cca(query_pred_original, query_pred_logits, return_conf=False, return_cc=False):
    '''
    Performs connected component analysis on the query_pred and returns the most confident connected component
    '''
    # cca_output = cv2.connectedComponentsWithStats(query_pred_original.astype(np.uint8), connectivity=8) # TODO try 8
    # # calc confidence for each connected component
    # cca_conf = []
    # for j in range(cca_output[0]):
    #     if j == 0:
    #         cca_conf.append(0) # background
    #         continue
    #     cca_conf.append((query_pred_logits.softmax(1)[:,1].flatten(1).cpu().detach().numpy() * (cca_output[1] == j).flatten()).sum() / ((cca_output[1] == j).flatten().sum() + 1e-6) * ((cca_output[1] == j).flatten().sum() / (query_pred_original.flatten().sum() + 1e-6))) # take into account the area of the connected component
    cca_output, cca_conf = get_connected_components(query_pred_original, query_pred_logits, return_conf=True)
    
    # find the most confident connected component, find max conf and its key
    max_conf = cca_conf[0]
    for k,v in cca_conf.items():
        if v > max_conf:
            max_conf = v
            max_key = k
        
    if max_conf == 0:
        # no connected component found, use zeros
        query_pred = np.zeros_like(query_pred_original)
    else:
        # zero out all other connected components
        new_cca_output = list(cca_output)
        new_cca_output[0] = 2  # bg + fg
        new_cca_output[1] = np.where(cca_output[1] != max_key, 0, 1)  # binarize the max_key
        new_cca_output[2] = cca_output[2][[0, max_key]]
        new_cca_output[3] = cca_output[3][[0, max_key]]
        cca_output = tuple(new_cca_output)

        query_pred = (cca_output[1] == 1).astype(np.uint8)
        # convert to binary mask
        query_pred = (query_pred > 0).astype(np.uint8) 
    
    if return_cc:
        return cca_output
    
    query_pred_original = query_pred_original * query_pred
    
    if return_conf:
        return query_pred_original, max_conf
    
    return query_pred_original

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'SABS': {
        'pa_all': set( [1,2,3,6]  ),
        0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
        1: set( [2,3] ), # lower_abdomen
    },
    'C0': {
        'pa_all': set(range(1, 4)),
        0: set([2,3]),
        1: set([1,3]),
        2: set([1,2]),
    },
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]), # lower_abdomen
    },
}

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()

def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)

def compose_wt_simple(is_wce, data_name):
    """
    Weights for cross-entropy loss
    """
    # if is_wce:
    #     if data_name in ['SABS', 'SABS_Superpix', 'SABS_448', 'SABS_Superpix_448', 'SABS_672', 'SABS_Superpix_672','C0', 'C0_Superpix', 'CHAOST2', 'CHAOST2_Superpix', 'CHAOST2_672', 'CHAOST2_Superpix_672', 'LITS17', 'LITS17_Superpix']:
    #         return torch.FloatTensor([0.05, 1.0]).cuda()
    #     else:
    #         raise NotImplementedError
    # else:
    #     return torch.FloatTensor([1.0, 1.0]).cuda()
    return torch.FloatTensor([0.05, 1.0]).cuda()


class CircularList(list):
    """
    Helper for spliting training and validation scans
    Originally: https://stackoverflow.com/questions/8951020/pythonic-circular-list/8951224
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)

