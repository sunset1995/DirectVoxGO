_base_ = '../default.py'

expname = 'dvgo_pwl_lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

model_class="dvgo_pwl"