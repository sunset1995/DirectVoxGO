_base_ = '../default.py'

expname = 'dvgo_lego_0.5-1.0'
basedir = './logs/pwl_nerf_datasets/'

data = dict(
    datadir='./data/pwl_nerf_dataset/lego_rgba_randdist_nv100_dist0.5-1.0_depth_sfn/',
    dataset_type='blender',
    white_bkgd=True,
)

