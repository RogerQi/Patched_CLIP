import glob
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Normalize, CenterCrop

import torch
from einops import rearrange
from PIL import Image

_VALID_IMAGE_EXTENSIONS = "jpg jpeg png JPG JPEG PNG".split(" ")

def patch_clip_preprocess(preprocess):
    """Patch CLIP preprocess to remove center crop"""
    # Check there is exactly one center crop transform
    is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
    assert sum(is_center_crop) == 1, "There should be exactly one CenterCrop transform"
    # Create new preprocess without center crop
    preprocess = Compose([t for t in preprocess.transforms if not isinstance(t, CenterCrop)])
    return preprocess

@torch.no_grad()
# def get_pca(
#     feat: TensorType["B", "H", "W", "C"], proj=None, dim=None, q=None, center=None
# ) -> Tuple[TensorType["B", "H", "W", dim], TensorType["C", dim]]:
def get_pca(feat, proj=None, dim=None, q=None, center=None):
    """Getting the principal components of a batch of feature maps.

    Args:
        feat: input feature map, [B, H, W, C]
        dim: output dimension
        proj: projection matrix, [C, dim]
        q: low-rank approximation. q>= dim
        center: if torch.pca_lowrank sets the mean of feat to zero.

    Returns: feat_pca[B, H, W, dim], proj[C, dim]

    """
    assert dim is None or q >= dim, "q must be greater than dim"

    feat_flat = rearrange(feat, "b h w c->(b h w) c")
    if proj is None:
        u, diag, proj = torch.pca_lowrank(feat_flat, q=q, center=center)
    else:
        u = feat_flat @ proj

    b, h, w, c = feat.shape
    feat_pca = u[:, :dim].reshape(b, h, w, c if dim is None else dim)

    return feat_pca, proj[:, :dim]


@lru_cache
def get_images_in_dir(image_dir: str, recursive: bool = True) -> List[str]:
    """Finds and returns all images in the given directory."""
    images = []
    for ext in _VALID_IMAGE_EXTENSIONS:
        if recursive:
            glob_pattern = f"{image_dir}/**/*.{ext}"
        else:
            glob_pattern = f"{image_dir}/*.{ext}"
        images.extend(glob.glob(glob_pattern, recursive=recursive))
    return images


def get_source_dir(dataset_path: str) -> str:
    """
    Returns the path to the source directory of the given dataset.

    We assume the images have been written to the source/ or images/
    directory (the latter in case of CLIPort datasets) in a flat
    structure as expected by Fast NeRF.
    """
    dataset_path = os.path.expandvars(dataset_path)
    source_dir = os.path.join(dataset_path, "source")
    if os.path.isdir(source_dir):
        return source_dir

    # Check if dataset follows CLIPort directory structure
    source_dir = os.path.join(dataset_path, "images")
    transforms_json = os.path.join(dataset_path, "transforms.json")
    if os.path.exists(transforms_json) and os.path.isdir(source_dir):
        print(f"Detected CLIPort dataset for {dataset_path}")
        return source_dir

    raise ValueError(f"Could not determine source image directory for {dataset_path}")


def load_images(
    image_dir: str, minimum_images: int = 1, recursive: bool = False
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads all images in the given directory.
    If recursive is True, then all subdirectories are searched for images.

    Returns a tuple of with the list of images and the list of image paths.
    """
    image_paths = get_images_in_dir(image_dir, recursive=recursive)
    if len(image_paths) < minimum_images:
        raise ValueError(f"Found {len(image_paths)} images in {image_dir}. Need at least {minimum_images} images.")

    images = [Image.open(image_path) for image_path in image_paths]
    return images, image_paths

def visualize_embedding_pca(
    image_paths: List[str],
    images: List[Image.Image],
    embeddings: np.ndarray,
    preprocess: Compose,
    visualize_every: int,
    alpha: float = 0.5,
    show_plot: bool = False,
    patch_size: int = 16,
    skip_center_crop: bool = False,
):
    """
    Save visualized embeddings. We show the pre-processed image
    without normalization and the embeddings projected using PCA into
    three components (RGB).

    We blend the resulting images and save it to disk and show the
    combined plot if show_plot is True.
    """
    from ml_logger import logger
    from sklearn.decomposition import PCA

    if skip_center_crop:
        preprocess = patch_clip_preprocess(preprocess)

    # Embeddings takes in the torch format.
    embeddings = rearrange(embeddings, "n c h w -> n (h w) c")

    if visualize_every < 1:
        raise ValueError("visualize_every must be >= 1")

    # Flatten to shape num_patches * clip_embedding_dimensionality
    og_viz_embeddings = embeddings[::visualize_every]
    num_patches = og_viz_embeddings.shape[0] * og_viz_embeddings.shape[1]
    viz_embeddings = og_viz_embeddings.reshape(num_patches, -1)

    # PCA with RGB components
    pca = PCA(n_components=3)
    pca.fit(viz_embeddings)
    embeddings_rgb = pca.transform(viz_embeddings)

    # Normalize by min-max of each R G B channel (it doesn't matter too much)
    rgb_max = embeddings_rgb.max(0)
    rgb_min = embeddings_rgb.min(0)
    normalized_embeddings_rgb = (embeddings_rgb - rgb_min) / (rgb_max - rgb_min)
    normalized_embeddings_rgb = normalized_embeddings_rgb.reshape(*og_viz_embeddings.shape[:2], 3)

    # Blend the pre-processed image with the PCA embedding and save to disk
    blended_images = []
    for idx, (image_path, pil_image, embedding_rgb) in enumerate(
        zip(
            image_paths[::visualize_every],
            images[::visualize_every],
            normalized_embeddings_rgb,
        )
    ):
        # Apply all but last normalize pre-process transform
        assert isinstance(preprocess.transforms[-1], Normalize), "last transform should be Normalize."

        image = pil_image
        for t in preprocess.transforms[:-1]:
            image = t(image)
        image = image.permute(1, 2, 0).cpu().numpy()
        # This check is not necessary as we apply the transform above, and the PIL image should be the raw image
        # assert image.shape[:2] == pil_image.size, "image size should remain the same"

        # Reshape embeddings to pre-processed image shape
        h = image.shape[0] // patch_size
        w = image.shape[1] // patch_size
        embedding_rgb = embedding_rgb.reshape(h, w, 3)
        scaling_factor_h = int((image.shape[0] / h))
        scaling_factor_w = int((image.shape[1] / w))
        embedding_rgb = np.kron(embedding_rgb, np.ones((scaling_factor_h, scaling_factor_w, 1)))

        # Plot embedding as a transparent layer on image
        image_255 = (image * 255).astype(np.uint8)
        embedding_rgb_255 = (embedding_rgb * 255).astype(np.uint8)
        blended = Image.blend(Image.fromarray(image_255), Image.fromarray(embedding_rgb_255), alpha=alpha)
        blended_images.append(blended)

        if show_plot:
            plt.figure()
            plt.imshow(blended)
            plt.show()
        else:
            # Save to ml-logger
            path = logger.save_image(blended, f"figures/{Path(image_path).stem}_viz.png")
            logger.print(f"Saved image at {path}.", color="yellow")

    return blended_images
