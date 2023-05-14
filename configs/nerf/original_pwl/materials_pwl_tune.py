_base_ = '../../default.py'

expname = 'dvgo_materials_pwl'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/materials',
    dataset_type='blender',
    white_bkgd=True,
)

model_class="dvgo_pwl"


train_ratio = 2
fine_train = dict(
    N_iters=20000 * train_ratio,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[
        1000 * train_ratio,
        2000 * train_ratio,
        3000 * train_ratio,
        4000 * train_ratio
    ],
)

expname = '%s_iterx%d' % (expname, train_ratio)