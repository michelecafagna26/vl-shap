import pickle
from pathlib import Path
from copy import copy

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from semshap.clipseg import CLIPDensePredT


def apply_mask(image, masks):
    """
    Inputs:
        image: PIL.Image
        masks: List[np.array]

    Return:
        PIL.Image
    """
    # assuming Image and mask have the same shape
    image_array = np.array(image)

    if not isinstance(masks, list):
        raise ValueError(f"'masks' is expected to be a list, not  {type(masks)}")
    if len(masks) == 0:
        raise ValueError("masks cannot be an empty list")

    mask = copy(masks[0])
    if len(masks) > 1:
        for m in masks[1:]:
            mask += m

    image_array[:, :, 0][~mask] = 0
    image_array[:, :, 1][~mask] = 0
    image_array[:, :, 2][~mask] = 0

    return Image.fromarray(image_array).convert('RGB')


def save_masks(masks, path, mode="pkl", img=None):
    # save the binary masks as pickle objects
    for i, m in enumerate(masks):

        if mode == "pkl":
            with open(Path(path, f"f{i}.pkl"), "wb") as fp:
                pickle.dump(m, fp)

        elif mode in {"jpg", "png"} and img is not None:

            masked_img = apply_mask(img, [m])
            masked_img.save(Path(path, f"f_{i}.{mode}"))
        else:
            raise ValueError(f"{mode} is not a valid mode! valid modes are 'pkl' | 'jpg' | 'png'")


def generate_dff_masks(visual_embeds, k=10, img_size=(256, 256), mask_th=20, random_state=0,
                       return_heatmaps=False):
    """
    Inputs:
        visual_embeds: torch.Tensor with shape (kernel_h, kernel_w, embed_dim)
        k : int,  number of feture masks to generate, because of the leftover mask we genererate k+1 masks, default 10
        img_size: tuple-like object [int, int], (width, height)
        mask_th: int, threshold to generate the binary masks, default 20
        random_state: int, default 0
        return_heatmaps: bool, if True the heatmaps are returned,default False
        reverse_resolution: bool, if True, we assume that the resolution is given (width, height), this is the default
                            behaviour of PIL.Image

    Return:
        Dict-like object: { "masks": List[numpy.array] , "heatmaps": List[numpy.array] if return_heatmaps is True}
    """

    # reshape the visual_embed to (h*w, embed_dim)
    ve_size = visual_embeds.shape
    visual_embeds = visual_embeds.reshape(-1, visual_embeds.size(2))

    # apply the NMF
    nmf = NMF(n_components=k, init='random', random_state=random_state)

    # W has shape (h*w, k)
    W = nmf.fit_transform(visual_embeds.numpy())
    # W has shape (k, embed_dim)
    # H = nmf.components_

    # let's reshape to get the heatmaps
    # now W has shape (h, w, k)
    W = W.reshape(ve_size[0], ve_size[1], -1)

    masks = []
    heatmaps = []
    leftover_mask = np.array([])
    for i in range(k):

        # manipulate it as a torch tensor
        heatmap = torch.tensor(W[..., i]).unsqueeze(0)

        # upsample to the image size
        heatmap = transforms.Resize((img_size[1], img_size[0]), interpolation=Image.BICUBIC)(heatmap)

        heatmap = (heatmap[0].numpy() * 255)
        # set the boundaries
        heatmap[heatmap > 255] = 255
        heatmap[heatmap < 0] = 0
        heatmaps.append(heatmap)

        # generate the binary mask
        mask = heatmap > mask_th
        masks.append(mask)

        if leftover_mask.size == 0:
            leftover_mask = np.copy(mask)
        else:
            leftover_mask += np.copy(mask)

    # add the leftover mask if it's not empty
    if np.sum(~leftover_mask) > 0:
        masks.append(~leftover_mask)

    out = {
        "masks": masks
    }
    if return_heatmaps:
        out['heatmaps'] = heatmaps

    return out


def generate_superpixel_masks(img_size, grid_shape=(4, 4)):
    """
    img_size: tuple-like object, image size of the image (width, height)
    grid_shape: tuple-like object, size of the grid used to partition the image (rows, columns)
    """
    patch_w = int(img_size[1] / grid_shape[0])
    patch_h = int(img_size[0] / grid_shape[1])
    masks = []

    for i in range(0, patch_w * grid_shape[0], patch_w):
        for j in range(0, patch_h * grid_shape[1], patch_h):
            mask = np.zeros((img_size[1], img_size[0]))

            if i + (2 * patch_w) > mask.shape[0]:
                i_end = mask.shape[0]
            else:
                i_end = i + patch_w

            if j + (2 * patch_h) > mask.shape[1]:
                j_end = mask.shape[1]
            else:
                j_end = j + patch_h

            mask[i:i_end, j:j_end] = 1
            masks.append(mask.astype(np.bool_))

    return {
        "masks": masks
    }


def generate_segmentation_masks(img, prompts, img_size=(256, 256), mask_th=100,
                           return_heatmaps=False, model_path='./semshap/clipseg/model/rd64-uni.pth', device=None):
    """
    Inputs:
        img: torch.Tensor with shape (kernel_h, kernel_w, embed_dim)
        labels: List[str], list of objects to segment
        img_size: tuple-like object [int, int], (width, height)
        mask_th: int, threshold to generate the binary masks, default 20
        return_heatmaps: bool, if True the heatmaps are returned,default False
        model_path: string, path to the model, default: ./semshap/clipseg/model/rd64-uni.pth'
        device: string "cpu" or "cuda", if None it will be assigned will be assigned with the following order: "cuda", "cpu"

    Return:
        Dict-like object: { "masks": List[numpy.array] , "heatmaps": List[numpy.array] if return_heatmaps is True}
    """

    if not device:
        device = "cuda" if torch.cuda.is_available() == True else "cpu"

    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)),
                          strict=False)

    # load and preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(img_size),
    ])
    img = transform(img).unsqueeze(0)

    masks = []
    heatmaps = []
    leftover_mask = np.array([])

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

    for i in range(len(preds)):

        # get the heatmap
        heatmap = torch.sigmoid(preds[i][0])

        # upsample to the image size
        heatmap = transforms.Resize((img_size[1], img_size[0]), interpolation=Image.BICUBIC)(heatmap)

        heatmap = (heatmap.numpy() * 255)

        # set the boundaries
        heatmap[heatmap > 255] = 255
        heatmap[heatmap < 0] = 0
        heatmaps.append(heatmap)

        # generate the binary mask
        mask = heatmap > mask_th
        masks.append(mask)

        if leftover_mask.size == 0:
            leftover_mask = np.copy(mask)
        else:
            leftover_mask += np.copy(mask)

    # add the leftover mask if it's not empty
    if np.sum(~leftover_mask) > 0:
        masks.append(~leftover_mask)

    out = {
        "masks": masks
    }
    if return_heatmaps:
        out['heatmaps'] = heatmaps

    return out


def genenerate_vit_masks(visual_embeds, img_size, k=10, mask_th=150, random_state=0,
                         return_heatmaps=False, reverse_resolution=True):
    """
    visual_embeds: numpy-like object shaped ( num_inputs, hidden_size ), this is usually the last hidden layer of ViT
    resolution: tuple-like object, image size of the original image (width, height)
    k: int, number of features masks to generate
    resolution: mask output resolution, for a ViT it usually corresponds to the number of patches
    mask_th: int, threshold used to generate the binary masks, default = 150
    random_state: int, default = 0
    return_heatmaps: bool, if 'True' the heatmaps are returned
    """

    # compute the feature index in the grid
    def extract_feature_idx(masks):
        return {idx: np.where(m[0, :] is True)[0].tolist() for idx, m in enumerate(masks)}

    # add together a set of masks
    def compose_mask(masks):
        out_mask = np.zeros(masks[0].shape, dtype=np.bool_)
        for m in masks:
            out_mask += m

        return out_mask

    # The first token is the BOS of the sequence
    visual_embeds = visual_embeds[..., 1:, :]

    resolution = (visual_embeds.shape[0], visual_embeds.shape[0])

    # visual_embeds may contain negative values (we cannot apply NMF), therefore we normalize
    scaler = MinMaxScaler()
    visual_embeds_norm = scaler.fit_transform(visual_embeds)

    dff_out = generate_dff_masks(torch.tensor(visual_embeds_norm).unsqueeze(0), k=k, return_heatmaps=return_heatmaps,
                                 img_size=resolution, mask_th=mask_th, random_state=random_state)

    feature_set = extract_feature_idx(dff_out['masks'])

    num_patches = np.sqrt(resolution[0]).astype(int)
    grid_shape = (num_patches, num_patches)

    # generate all the superpixel masks

    masks = generate_superpixel_masks(img_size, grid_shape=grid_shape)['masks']

    # Let's group them together
    new_masks = [compose_mask([masks[j] for j in feature_set[i]]) for i in feature_set if len(feature_set[i]) > 0]

    out = {
        "masks": new_masks
    }

    if "heatmaps" in dff_out:
        out['heatmaps'] = dff_out['heatmaps']

    return out
