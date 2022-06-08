#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
   helper function to skip oversampled points,
   especially near the foreground scene bbox boundary
   */
template <typename scalar_t>
__global__ void cumdist_thres_cuda_kernel(
        scalar_t* __restrict__ dist,
        const float thres,
        const int n_rays,
        const int n_pts,
        bool* __restrict__ mask) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    float cum_dist = 0;
    const int i_s = i_ray * n_pts;
    const int i_t = i_s + n_pts;
    int i;
    for(i=i_s; i<i_t; ++i) {
      cum_dist += dist[i];
      bool over = (cum_dist > thres);
      cum_dist *= float(!over);
      mask[i] = over;
    }
  }
}

torch::Tensor cumdist_thres_cuda(torch::Tensor dist, float thres) {
  const int n_rays = dist.size(0);
  const int n_pts = dist.size(1);
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  auto mask = torch::zeros({n_rays, n_pts}, torch::dtype(torch::kBool).device(torch::kCUDA));
  AT_DISPATCH_FLOATING_TYPES(dist.type(), "cumdist_thres_cuda", ([&] {
    cumdist_thres_cuda_kernel<scalar_t><<<blocks, threads>>>(
        dist.data<scalar_t>(), thres,
        n_rays, n_pts,
        mask.data<bool>());
  }));
  return mask;
}



__global__ void __set_i_for_segment_start_end(
        int64_t* __restrict__ ray_id,
        const int n_pts,
        int64_t* __restrict__ i_start_end) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<index && index<n_pts && ray_id[index]!=ray_id[index-1]) {
    i_start_end[ray_id[index]*2] = index;
    i_start_end[ray_id[index-1]*2+1] = index;
  }
}

template <typename scalar_t>
__global__ void segment_cumsum_cuda_kernel(
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ s,
    int64_t* __restrict__ i_start_end,
    const int n_rays,
    scalar_t* __restrict__ w_prefix,
    scalar_t* __restrict__ w_total,
    scalar_t* __restrict__ ws_prefix,
    scalar_t* __restrict__ ws_total) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start_end[i_ray*2];
    const int i_e = i_start_end[i_ray*2+1];

    float w_cumsum = 0;
    float ws_cumsum = 0;
    for(int i=i_s; i<i_e; ++i) {
      w_prefix[i] = w_cumsum;
      ws_prefix[i] = ws_cumsum;
      w_cumsum += w[i];
      ws_cumsum += w[i] * s[i];
    }
    w_total[i_ray] = w_cumsum;
    ws_total[i_ray] = ws_cumsum;
  }
}

std::vector<torch::Tensor> segment_cumsum_cuda(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id) {
  const int n_pts = ray_id.size(0);
  const int n_rays = ray_id[n_pts-1].item<int>() + 1;
  const int threads = 256;

  // Find the start and end index of a segment. For instance:
  // ray_id  = [0 0 0 1 1 2 4 4 4 4]
  // i_start_end = [[0,3] [3,5] [5,6] [0,0] [6,10]]
  auto i_start_end = torch::zeros({n_rays*2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_i_for_segment_start_end<<<(n_pts+threads-1)/threads, threads>>>(
          ray_id.data<int64_t>(), n_pts, i_start_end.data<int64_t>());
  i_start_end[ray_id[n_pts-1]*2+1] = n_pts;

  auto w_prefix = torch::zeros_like(w);
  auto w_total = torch::zeros({n_rays}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto ws_prefix = torch::zeros_like(w);
  auto ws_total = torch::zeros({n_rays}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(w.type(), "segment_cumsum_cuda", ([&] {
    segment_cumsum_cuda_kernel<scalar_t><<<blocks, threads>>>(
        w.data<scalar_t>(),
        s.data<scalar_t>(),
        i_start_end.data<int64_t>(),
        n_rays,
        w_prefix.data<scalar_t>(),
        w_total.data<scalar_t>(),
        ws_prefix.data<scalar_t>(),
        ws_total.data<scalar_t>());
  }));
  return {w_prefix, w_total, ws_prefix, ws_total};
}








