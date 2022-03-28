_base_ = '../default.py'

basedir = './logs/llff'

data = dict(
    dataset_type='llff',
    ndc=True,
    width=1008,
    height=756,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=30000,
    N_rand=4096,
    pg_scale=[2000,4000,6000,8000],
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
)

_mpi_depth = 256
_stepsize = 0.5

fine_model_and_render = dict(
    num_voxels=384*384*_mpi_depth,
    mpi_depth=_mpi_depth,
    stepsize=_stepsize,
    rgbnet_dim=16,
    rgbnet_width=128,
    world_bound_scale=1,
    fast_color_thres=_stepsize/_mpi_depth/5,
)

