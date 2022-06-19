_base_ = './default_forward_facing.py'

expname = 'Otobai'

data = dict(
    datadir='./data/custom/Otobai/dense',
    factor=2,
    movie_render_kwargs={
        'scale_r': 0.8,
        'scale_f': 10.0,
        'zrate': 6.0,
        'zdelta': 0.5,
    }
)

