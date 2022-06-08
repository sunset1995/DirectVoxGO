#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cumdist_thres_cuda(torch::Tensor dist, float thres);

std::vector<torch::Tensor> segment_cumsum_cuda(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cumdist_thres(torch::Tensor dist, float thres) {
  CHECK_INPUT(dist);
  return cumdist_thres_cuda(dist, thres);
}

std::vector<torch::Tensor> segment_cumsum(torch::Tensor w, torch::Tensor s, torch::Tensor ray_id) {
  CHECK_INPUT(w);
  CHECK_INPUT(s);
  CHECK_INPUT(ray_id);
  return segment_cumsum_cuda(w, s, ray_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cumdist_thres", &cumdist_thres, "Generate mask for cumulative dist.");
  m.def("segment_cumsum", &segment_cumsum, "Compute segment prefix-sum (cumsum).");
}

