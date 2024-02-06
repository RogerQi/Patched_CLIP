"""
Extract patch embeddings from CLIP
"""
import gc
import os
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from params_proto import PrefixProto, Proto
from PIL import Image
from torchvision.transforms import CenterCrop, Compose

from .modified_clip import clip
from .modified_clip.model import CLIP

# fmt: off
class CLIP_args(PrefixProto):
    clip_model_name: str = Proto(
        "ViT-L/14@336px",
        help="Name of CLIP model to use. You should only use ViT versions, "
             "of which CLIP supports ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px.", )

    visualize_embeddings: bool = Proto(True, help="Whether to visualize the CLIP embeddings.")
    visualize_every: int = Proto(5, help="Frequency of images to visualize (e.g. every 10th image).")

    batch_size: int = Proto(64, help="Batch size for image input to CLIP.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dir: str = Proto("feat", help="Name of directory to store features in.")
    visualize_dir: str = Proto("clip_viz", help="Name of directory to store visualizations in.")
    patch_size: int = Proto(14, help="Patch size for CLIP. Valid options: 14.")


def load_clip() -> Tuple[CLIP, Compose]:
    """
    Load CLIP and its preprocessing transforms.
    Cache it, so we only have to do it once.
    """
    model: CLIP
    model, preprocess = clip.load(
        name=CLIP_args.clip_model_name, device=CLIP_args.device
    )
    return model, preprocess


@torch.no_grad()
def get_clip_projection_matrix() -> np.ndarray:
    """Get the CLIP image encoder projection matrix"""
    model, _ = load_clip()
    projection_matrix = model.get_image_encoder_projection()
    projection_matrix_np = projection_matrix.cpu().numpy()
    return projection_matrix_np


def patch_clip_preprocess(preprocess):
    """Patch CLIP preprocess to remove center crop"""
    # Check there is exactly one center crop transform
    is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
    assert sum(is_center_crop) == 1, "There should be exactly one CenterCrop transform"
    # Create new preprocess without center crop
    preprocess = Compose([t for t in preprocess.transforms if not isinstance(t, CenterCrop)])
    return preprocess

@torch.no_grad()
def get_clip_embeddings_export(images: Union[List[Image.Image], List[torch.Tensor]],
                        model: CLIP = None, preprocess: Compose = None, to_cpu: bool = True,
                        skip_center_crop: bool = False) -> torch.Tensor:
    """
    Process a list of images and get their patch embeddings with CLIP.
    """
    batch_idx = 0
    aspect_ratio = images[batch_idx].shape[-1] / images[batch_idx].shape[-2]
    target_H = 336
    target_W = int(target_H * aspect_ratio)
    images[batch_idx] = F.interpolate(images[batch_idx].unsqueeze(0),
                                        size=(target_H, target_W),
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)
    preprocessed_images = torch.stack(images)
    preprocessed_images = preprocessed_images.to('cuda')
    # normalize
    rgb_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1, 3, 1, 1).to('cuda')
    rgb_std  = torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1, 3, 1, 1).to('cuda')
    preprocessed_images = (preprocessed_images - rgb_mean) / rgb_std

    # Get CLIP embeddings for the images
    patch_embeddings = model.get_patch_encodings(preprocessed_images)

    # Reshape embeddings to number of patches in height and width
    h_in, w_in = preprocessed_images.shape[-2:]
    h_out = h_in // model.visual.patch_size
    w_out = w_in // model.visual.patch_size
    embeddings = rearrange(patch_embeddings, "b (h w) d -> b d h w", h=h_out, w=w_out)
    
    return embeddings

@torch.no_grad()
def get_clip_embeddings(images: Union[List[Image.Image], List[torch.Tensor]],
                        model: CLIP = None, preprocess: Compose = None, to_cpu: bool = True,
                        skip_center_crop: bool = False) -> torch.Tensor:
    """
    Process a list of images and get their patch embeddings with CLIP.
    """
    assert CLIP_args.clip_model_name.startswith("ViT")

    # Load CLIP if not provided
    if model is None or preprocess is None:
        model, preprocess = load_clip()
    
    start_time = time.perf_counter()

    if isinstance(images[0], Image.Image):
        # TODO: support arbitrary user specified image sizes? Rather than the resize in the preprocess
        if skip_center_crop:
            preprocess = patch_clip_preprocess(preprocess)

        # Preprocess each image
        preprocessed_images = torch.stack([preprocess(image) for image in images])
        # (b, 3, H, W); usually H is the short edge and resized to 336
        # if skip_center_crop is set to True, then the aspect ratio is preserved (i.e., W can be larger than H=336)
        # otherwise, the image is center cropped to 336x336
        preprocessed_images = preprocessed_images.to(CLIP_args.device)
    else:
        assert isinstance(images, list)
        # Assume images is a list of CHW tensors normalized to 0-1 in RGB order
        for batch_idx in range(len(images)):
            # sanity check
            assert isinstance(images[batch_idx], torch.Tensor)
            assert images[batch_idx].shape[0] == 3
            assert images[batch_idx].max() <= 1.0
            assert images[batch_idx].min() >= 0.0

            # resize images
            if skip_center_crop:
                # TODO(roger): support portrait images
                # assert images[batch_idx].shape[-1] >= images[batch_idx].shape[-2], "W should be larger than H"
                aspect_ratio = images[batch_idx].shape[-1] / images[batch_idx].shape[-2]
                target_H = 336 // 2
                target_W = int(target_H * aspect_ratio)
                images[batch_idx] = F.interpolate(images[batch_idx].unsqueeze(0),
                                                  size=(target_H, target_W),
                                                  mode='bilinear',
                                                  align_corners=False).squeeze(0)
            else:
                assert images[batch_idx].shape[-1] == images[batch_idx].shape[-2]
                target_H, target_W = (336, 336)
                images[batch_idx] = F.interpolate(images[batch_idx].unsqueeze(0),
                                                    size=(target_H, target_W),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0)
        preprocessed_images = torch.stack(images)
        preprocessed_images = preprocessed_images.to(CLIP_args.device)
        # normalize
        rgb_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1, 3, 1, 1).to(preprocessed_images.device)
        rgb_std  = torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1, 3, 1, 1).to(preprocessed_images.device)
        preprocessed_images = (preprocessed_images - rgb_mean) / rgb_std

    # Get CLIP embeddings for the images
    patch_embeddings = []
    for i in range(0, len(images), CLIP_args.batch_size):
        batch = preprocessed_images[i: i + CLIP_args.batch_size]
        patch_embeddings.append(model.get_patch_encodings(batch))
    patch_embeddings = torch.cat(patch_embeddings, dim=0)

    # Reshape embeddings to number of patches in height and width
    h_in, w_in = preprocessed_images.shape[-2:]
    h_out = h_in // model.visual.patch_size
    w_out = w_in // model.visual.patch_size
    embeddings = rearrange(patch_embeddings, "b (h w) d -> b d h w", h=h_out, w=w_out)
    if to_cpu:
        embeddings = embeddings.cpu()

    end_time = time.perf_counter()
    mean_time = (end_time - start_time) / len(images)
    
    return embeddings
