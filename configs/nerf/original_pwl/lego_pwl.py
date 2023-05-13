_base_ = '../default.py'

# expname = 'dvgo_pwl_lego_2'
# expname = 'dvgo_pwl_lego'
# expname = 'dvgo_pwl_lego_eps1e-6'
# expname = 'dvgo_pwl_lego_eps1e-6_pgscale_cache_coarse'
# expname = 'dvgo_pwl_lego_eps1e-6_pgscale_cache_coarse_cache'
expname = 'dvgo_lego_pwl'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

# No Coarse training
coarse_train = {
    # "N_iters": 0, 
    # "ray_sampler": "random"
    # "disable_cache": True
}
model_class="dvgo_pwl"

fine_train = {
    # "disable_cache": True,
    # "pg_scale": [], # no scaling
    # "lrate_density": 1e-1,           # lr of density voxel grid
    # "lrate_density": 1e-2,           # lr of density voxel grid
    # # "lrate_k0": 1e-1,                # lr of color/feature voxel grid
    # "lrate_k0": 1e-2,                # lr of color/feature voxel grid
    # "lrate_rgbnet":1e-3,            # lr of the mlp to preduct view-dependent color
}
