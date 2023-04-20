_base_ = '../default.py'

expname = 'dvgo_lego_200'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender_supres',
    white_bkgd=True,
    res=200,
)

