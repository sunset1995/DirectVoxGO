# Improving log

### Custome CUDA implementation for efficiency
Some intermediate steps are reimplemented in cuda (`lib/cuda/`), which improves training speed by 
**1.6\~3.1**. Below show the results with dense grid under `256^3` voxels and `160^3` voxels and times are measured on a Telsa V100 GPU.


| **num_voxels=256^3**    | lego  |       | mic   |       | ship  |       |
|--------------|-------|-------|-------|-------|-------|-------|
|              | psnr  | `mm:ss` | psnr  | `mm:ss` | psnr  | `mm:ss` |
| native pytorch<br>[b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 35.51 | `15:10`        | 34.39 | `14:11`        | 30.05 | `17:04` |
| cuda re-impl. Adam optimizer<br>[d3783f4](https://github.com/sunset1995/DirectVoxGO/tree/d3783f4) | 35.47 | `08:54` (1.7x) | 34.34 | `06:41` (2.1x) | 30.05 | `10:23` (1.6x) |
| cuda re-impl.  rendering<br>[3de7a6d](https://github.com/sunset1995/DirectVoxGO/tree/3de7a6d) | 35.63 | `06:31` (2.3x) | 34.48 | `04:31` (3.1x) | 30.30 | `08:20` (2.0x) |

```python
# The model&training config for the results above
coarse_train = dict(N_iters=5000)
fine_train = dict(pg_scale=[1000,2000,3000,4000,5000,6000])
fine_model_and_render = dict(num_voxels=256**3)
```


| **num_voxels=160^3**    | lego  |       | mic   |       | ship  |       |
|--------------|-------|-------|-------|-------|-------|-------|
|              | psnr  | `mm:ss` | psnr  | `mm:ss` | psnr  | `mm:ss` |
| native pytorch<br>[b076912](https://github.com/sunset1995/DirectVoxGO/tree/b076912) | 34.65 | `08:29`        | 33.19 | `07:04`        | 29.08 | `10:38`        |
| cuda re-impl.  Adam optimizer<br>[d3783f4](https://github.com/sunset1995/DirectVoxGO/tree/d3783f4) | 34.66 | `06:01` (1.4x) | 33.14 | `04:38` (1.5x) | 29.04 | `08:06` (1.3x) |
| cuda re-impl.  rendering<br>[3de7a6d](https://github.com/sunset1995/DirectVoxGO/tree/3de7a6d) | 34.56 | `04:50` (1.8x) | 33.10 | `03:22` (2.1x) | 29.19 | `06:31` (1.6x) |

```python
# The model&training config for the results above
coarse_train = dict(N_iters=5000)
fine_train = dict(pg_scale=[1000,2000,3000,4000])
fine_model_and_render = dict(num_voxels=160**3)
```

