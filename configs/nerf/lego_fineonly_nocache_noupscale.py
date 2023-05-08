_base_ = '../default.py'

expname = 'dvgo_lego_fineonly_nocache_noupscale'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)

# No Coarse training
coarse_train = {
    "N_iters": 0,
    "disable_cache": True
}

fine_train = {
    "disable_cache": True,
    # "pg_scale":[1000, 2000, 3000, 4000],
    "pg_scale": [], # no scaling
}
