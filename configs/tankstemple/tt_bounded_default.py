_base_ = '../default.py'

basedir = './logs/tanks_and_temple'

data = dict(
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=10000,
    pervoxel_lr_downrate=2,
)

