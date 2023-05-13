_base_ = '../default.py'

expname = 'dvgo_chair_pwl'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
)

model_class="dvgo_pwl"
