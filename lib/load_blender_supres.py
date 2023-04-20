import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from .load_blender import trans_t, rot_phi, rot_theta, pose_spherical


# From Yuval
def im_resize(image,scale_factor: int):
    assert all([v%scale_factor==0 for v in image.shape[:2]]),'Not supporting non-integer downscaling factor.'
    return cv2.resize(
        image, dsize=(image.shape[1]//scale_factor,image.shape[0]//scale_factor), interpolation=cv2.INTER_AREA)

# From Yuval
def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple((torch.Tensor, torch.Tensor)):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2,indexing='ij')
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor,padding_size: int=0,downsampling_factor: int=1,
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    downsampling_offset = (downsampling_factor-1)/(2*downsampling_factor)

    ii, jj = meshgrid_xy(
        (torch.arange(width+2*padding_size)+downsampling_offset).to(tform_cam2world),
        (torch.arange(height+2*padding_size)+downsampling_offset).to(tform_cam2world),
    )
    if padding_size>0:
        ii = ii-padding_size
        jj = jj-padding_size
    directions = torch.stack(
        [
            (ii - width * 0.5) / get_focal(focal_length,'H'),
            -(jj - height * 0.5) / get_focal(focal_length,'W'),
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def load_blender_data(basedir, res=None, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if res is not None:
        scale_factor = int(H / res)
        H = W = res
        focal = focal/scale_factor
        # H = H//2
        # W = W//2
        # focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs_half_res[i] = im_resize(img, scale_factor)

        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


