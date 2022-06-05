_base_ = './tt_default.py'

fine_train = dict(
    pg_scale=[1000,2000,3000,4000,5000,6000,7000],
)

fine_model_and_render = dict(
    num_voxels=320**3,
)

