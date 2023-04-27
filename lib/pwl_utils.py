####################
### PWL Integration
####################
import torch
import torch.nn.functional as F


## PWL Modified to match the Alpha2Wight function
def raw_to_alphas_weights(raw, ray_id, ray_pts, ray_o):
    """
    Args:
        M is stacked and masked number, N is the number of rays
        \cross{[alphas] (M,)}
        [raw] (M,)
        [ray_id] (M,)
        [ray_pts] (M, 3) 
        [ray_o] (N, 3)
    Returns:
        [alphas]
        [alphainv_last]
        [weights]
    """
    # compute accumulated transmittance
    M = raw.shape[0]
    N = ray_o.shape[0]
    lens = torch.stack([(ray_id == i).sum() for i in range(N)])
    clen = torch.cat([
        torch.zeros(1).to(lens), torch.cumsum(lens, 0)
    ])
    lmax = lens.max().item()
    # Will batch this alphas
    raw_padded = torch.ones((N, lmax)) * torch.nan
    dists_padded = torch.zeros((N, lmax))
    # TODO: vectorize this!
    for i in range(N):
        raw_padded[i, :lens[i]] = raw[clen[i]:clen[i+1]]
        depth_curr = torch.linalg.norm(
            ray_pts[clen[i]:clen[i+1]] - ray_o[i, :].reshape(-1, 3),
            dim=-1)
        dists_curr = torch.abs(depth_curr[:-1] - depth_curr[1:])
        dists_padded[i, :lens[i]-1] = dists_curr

    noise = 0 # TODO:guandao
    tau = torch.cat([
        torch.ones((N, 1)).to(raw) * 1e-10,
        # alpha[...,3] + noise,
        raw_padded + noise,
    ], -1) ### tau(near) = very small, (N, L + 1)

    # NOTE: tau and alphas are all positive
    # tau = F.relu(tau) ## Make positive from proof of DS-NeRF
    interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])  # (N, L)
   
    ### Evaluating exp(-0.5 (tau_{i+1}+tau_i) (s_{i+1}-s_i) )
    ### [N, L]
    alphas_padded = torch.exp( - interval_ave_tau * dists_padded)  

    ### Transmittance until s_n
    ### (i.e.) probability of not hitting before [i]
    T = torch.cumprod(
        torch.cat([
            torch.ones((alphas_padded.shape[0], 1)).to(alphas_padded), 
            alphas_padded], 
        -1), 
    -1) # [N_rays, N_samples+1], T(near)=1, starts off at 1

    ### Factor to multiply transmittance with = (1-expr)
    ### (i.e.) probability of hitting at [i]
    ### The weight will be directly multiplied into the color
    weights_padded = (1 - alphas_padded) * T[:, 1:] # [N_rays, N_samples]
    alphainv_last = T[torch.arange(N).long(), lens.long()]  # N_rays

    ### Now flatten it according to ray_id
    weights = torch.cat(
        [weights_padded[i, :lens[i].long()] for i in range(N)], 0)
    alphas = torch.cat(
        [alphas_padded[i, :lens[i].long()] for i in range(N)], 0)
    return alphas, alphainv_last, weights



# ## PWL Modified to match the Alpha2Wight function
# def raw2alphas(raw, ray_id, ray_pts, ray_o):
    # """
    # Args:
        # M is stacked and masked number, N is the number of rays
        # \cross{[alphas] (M,)}
        # [raw] (M,)
        # [ray_id] (M,)
        # [ray_pts] (M, 3) 
        # [ray_o] (N, 3)
    # Returns:
        # [weights]
        # [alphainv_last]
    # """
    # # compute accumulated transmittance
    # M = raw.shape[0]
    # N = ray_o.shape[0]
    # lens = torch.stack([(ray_id == i).sum() for i in range(N)])
    # clen = torch.cat([
        # torch.zeros(1).to(lens), torch.cumsum(lens, 0)
    # ])
    # lmax = lens.max().item()
    # # Will batch this alphas
    # raw_padded = torch.ones((N, lmax)) * torch.nan
    # dists_padded = torch.zeros((N, lmax))
    # # TODO: vectorize this!
    # for i in range(N):
        # raw_padded[i, :lens[i]] = raw[clen[i]:clen[i+1]]
        # depth_curr = torch.linalg.norm(
            # ray_pts[clen[i]:clen[i+1]] - ray_o[i, :].reshape(-1, 3),
            # dim=-1)
        # dists_curr = torch.abs(depth_curr[:-1] - depth_curr[1:])
        # dists_padded[i, :lens[i]-1] = dists_curr

    # noise = 0 # TODO:guandao
    # tau = torch.cat([
        # torch.ones((N, 1)).to(raw) * 1e-10,
        # # alpha[...,3] + noise,
        # raw_padded + noise,
    # ], -1) ### tau(near) = very small, (N, L + 1)

    # # NOTE: tau and alphas are all positive
    # # tau = F.relu(tau) ## Make positive from proof of DS-NeRF
    # interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])  # (N, L)
   
    # ### Evaluating exp(-0.5 (tau_{i+1}+tau_i) (s_{i+1}-s_i) )
    # ### [N, L]
    # expr = torch.exp( - interval_ave_tau * dists_padded)  

    # ### Transmittance until s_n
    # ### (i.e.) probability of not hitting before [i]
    # T = torch.cumprod(
        # torch.cat([torch.ones((expr.shape[0], 1)).to(expr), expr], -1), 
        # -1) # [N_rays, N_samples+1], T(near)=1, starts off at 1

    # ### Factor to multiply transmittance with = (1-expr)
    # ### (i.e.) probability of hitting at [i]
    # ### The weight will be directly multiplied into the color
    # weights_padded = (1 - expr) * T[:, 1:] # [N_rays, N_samples]
    # alphainv_last = T[torch.arange(N), lens.int()]  # N_rays

    # ### Now flatten it according to ray_id
    # weights = torch.cat([weights_padded[i, lens[i]] for i in range(N)], 0)
    # import pdb; pdb.set_trace()
    # return weights, alphainv_last


### Our reformulation to piecewise linear
def compute_weights_piecewise_linear(
        raw, z_vals, near, far, rays_d, noise=0., return_tau=False):
    raw2expr = lambda raw, dists: torch.exp(-raw*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]

    ### Original code
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    tau = torch.cat([torch.ones((raw.shape[0], 1), device=device)*1e-10, raw[...,3] + noise, torch.ones((raw.shape[0], 1), device=device)*1e10], -1) ### tau(near) = 0, tau(far) = very big (will hit an opaque surface)

    tau = F.relu(tau) ## Make positive from proof of DS-NeRF

    interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])
   
    '''
    Evaluating exp(-0.5 (tau_{i+1}+tau_i) (s_{i+1}-s_i) )
    '''
    expr = raw2expr(interval_ave_tau, dists)  # [N_rays, N_samples+1]

    ### Transmittance until s_n
    T = torch.cumprod(torch.cat([torch.ones((expr.shape[0], 1), device=device), expr], -1), -1) # [N_rays, N_samples+2], T(near)=1, starts off at 1

    ### Factor to multiply transmittance with
    factor = (1 - expr)

    weights = factor * T[:, :-1] # [N_rays, N_samples+1]

    '''
    We will need to return tau and T for backprop later
    '''
    ### Remember to remove the last value of T(far) is not used
    ### tau(far) is also not used

    if return_tau:
        return weights, tau, T
    else:
        return weights    

### Original piecewise constant assumption
def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights


### Integration aggregation function
def raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std=0, pytest=False, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if mode == "linear":
        weights, tau, T = compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise, return_tau=True)
    
        ### Skip the first bin weights [near, s_0]
        # weights_to_aggregate = weights[..., 1:]

        if color_mode == "midpoint":
            rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, rgb[: ,-1, :].unsqueeze(1)], 1)
            rgb_mid = .5 * (rgb_concat[:, 1:, :] + rgb_concat[:, :-1, :])

            rgb_map = torch.sum(weights[...,None] * rgb_mid, -2)  # [N_rays, 3]

        elif color_mode == "left":
            rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb], 1)
            rgb_map = torch.sum(weights[...,None] * rgb_concat, -2)

        else:
            print("ERROR: Color mode unimplemented, please select left or midpoint.")

        if DEBUG:
            print("Does nan exist in per point rgb_map")
            print(torch.isnan(rgb_map).any())

        ### Piecewise linear means take the midpoint
        z_vals = torch.cat([near, z_vals, far], -1)
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        depth_map = torch.sum(weights * z_vals_mid, -1)

    elif mode == "constant":
        weights = compute_weights(raw, z_vals, rays_d, noise)
        
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)    
        
        tau = None
        T = None    

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, tau, T

#########################
### Hierarchical sampling
#########################
## Eq 17
def pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )

    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_right - tau_left))

    ### clamp t to [0, s_right - s_left]
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)

    sample = s_left + t

    return sample

## Eq 21
def pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))

    ### clamp t to [0, s_right - s_left]
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)

    sample = s_left + t

    return sample

def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, quad_solution_v2=False, zero_threshold = 1e-4, epsilon_=1e-3):
    ### bins = z_vals, ie bin boundaries, input does not include near and far plane yet ## N_samples, with near and far it will become N_samples+2
    ### weights is the PMF of each bin ## N_samples + 1

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]

    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]


    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    ### PWL --> Eq 17 and 21 in the writeup, these two are actually equivalent and can be combined --> Eq 23
    samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)
    samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)

    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below


### This is the original piecewise constant case
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

###############################################################
#### Render rays function in pytorch where everything is called
###############################################################
def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                mode,
                color_mode,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                white_bkgd=False,
                quad_solution_v2=False,
                zero_tol = 1e-4,
                epsilon = 1e-3):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

    if perturb > 0.:
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(
        raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std,
        pytest=pytest, white_bkgd=white_bkgd)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if mode == "linear":
            z_samples, _, _, _ = sample_pdf_reformulation(
                z_vals, weights, tau, T, near, far, N_importance, 
                det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2, zero_threshold = zero_tol, epsilon_=epsilon)
        elif mode == "constant":
            z_samples = sample_pdf(
                z_vals_mid, weights[...,1:-1], N_importance, 
                det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        ######## Clamping in quad solution --> this should not be needed anymore
        z_samples = torch.clamp(z_samples, near, far)
        ########

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(
            raw, z_vals, near, far, rays_d, mode,
            color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd)
        

    if mode == "linear":
        weights = weights[..., 1:]

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map,
           'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        weights_0 = weights_0[..., 1:]

        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return ret

