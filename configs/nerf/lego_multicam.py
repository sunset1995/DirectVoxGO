_base_ = '../default.py'

expname = 'dvgo_lego_multicam2'
basedir = './logs/nerf_synthetic_multicam2'

data = dict(
    datadir='./data/nerf_synthetic_multicam2/lego_0.5_1.25',
    dataset_type='blender',
    white_bkgd=True,
)

