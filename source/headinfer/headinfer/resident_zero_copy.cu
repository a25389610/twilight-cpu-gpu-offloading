#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Exact K/V assembly with bounded GPU resident storage. Misses read the full
// pinned CPU slab through its mapped CUDA address; no full KV GPU mirror exists.
__global__ void assemble_zero_copy(
    const half2* host_k, const half2* host_v,
    const half2* resident_k, const half2* resident_v,
    const int32_t* hit_source, const int32_t* current_position,
    const int32_t* current_length, const int32_t* previous_length,
    half2* attention_k, half2* attention_v,
    int history_length, int host_capacity, int layer,
    int half2_per_row) {
  const int group = blockIdx.y;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int col = threadIdx.x;
  const int count = current_length[group];
  if (row >= count || col >= half2_per_row) return;
  int previous_base = group;
  int current_base = group;
  #pragma unroll
  for (int g = 0; g < group; ++g) {
    previous_base += previous_length[g];
    current_base += current_length[g];
  }
  const int source = hit_source[group * history_length + row];
  const int destination = current_base + row;
  if (source >= 0) {
    attention_k[destination * half2_per_row + col] =
        resident_k[(previous_base + source) * half2_per_row + col];
    attention_v[destination * half2_per_row + col] =
        resident_v[(previous_base + source) * half2_per_row + col];
  } else {
    const int position = current_position[group * history_length + row];
    const int host_row = (layer * 8 + group) * host_capacity + position;
    attention_k[destination * half2_per_row + col] =
        host_k[host_row * half2_per_row + col];
    attention_v[destination * half2_per_row + col] =
        host_v[host_row * half2_per_row + col];
  }
}

extern "C" int launch_resident_zero_copy(
    const void* host_k, const void* host_v,
    const void* resident_k, const void* resident_v,
    const void* hit_source, const void* current_position,
    const void* current_length, const void* previous_length,
    void* attention_k, void* attention_v,
    int history_length, int host_capacity, int layer,
    int head_dim, int max_group_rows, void* stream) {
  if (head_dim % 2 != 0) return -1;
  constexpr int rows_per_block = 8;
  const int half2_per_row = head_dim / 2;
  dim3 block(half2_per_row, rows_per_block);
  dim3 grid((max_group_rows + rows_per_block - 1) / rows_per_block, 8);
  assemble_zero_copy<<<grid, block, 0, (cudaStream_t)stream>>>(
      (const half2*)host_k, (const half2*)host_v,
      (const half2*)resident_k, (const half2*)resident_v,
      (const int32_t*)hit_source, (const int32_t*)current_position,
      (const int32_t*)current_length, (const int32_t*)previous_length,
      (half2*)attention_k, (half2*)attention_v,
      history_length, host_capacity, layer, half2_per_row);
  return (int)cudaGetLastError();
}
