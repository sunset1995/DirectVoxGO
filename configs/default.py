from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # llff | blender | nsvf | blendedmvs | tankstemple
    inverse_y=False,              # intrinsict mode
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background [blender|tankstemple|nsvf]
    half_res=False,               # downsample by a factor of 2 [blender]
    factor=4,                     # downsample factor in [llff]
    ndc=False,                    # use ndc coordinate (only for forward-facing)
    spherify=False,               # inward-facing in [llff]
    llffhold=8,                   # testsplit in [llff]
    load_depths=False,            # load depth [llff]
)

''' Template of training options
'''
coarse_train = dict(
    N_iters=10000,                # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per gradient step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_from=0,                    # count total variation loss from tv_from step
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    pg_scale=[],                  # checkpoints for progressive scaling
)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=20000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000],
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1024000,
    num_voxels_base=1024000,
    nearest=False,
    pre_act_density=False,
    in_act_density=False,
    bbox_thres=1e-3,
    mask_cache_thres=1e-3,
    rgbnet_dim=0,
    rgbnet_full_implicit=False,
    rgbnet_direct=True,
    rgbnet_depth=3,
    rgbnet_width=128,
    alpha_init=1e-6,
    fast_color_thres=0,
    maskout_near_cam_vox=True,
    world_bound_scale=1,
    stepsize=0.5,
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))

del deepcopy
