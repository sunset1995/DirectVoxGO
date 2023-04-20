_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = './logs/nerf_synthetic_multicam2'

data = dict(
    datadir='./data/nerf_synthetic_multicam2/lego',
    dataset_type='blender',
    white_bkgd=True,
)

