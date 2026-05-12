#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                           \
  do {                                                             \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                            \
      exit(1);                                                     \
    }                                                              \
  } while (0)

__global__ void clearBucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) bucket[i] = 0;
}

__global__ void countBucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bucket[key[i]], 1);
}

__global__ void makeOffset(int *bucket, int *offset, int range) {
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i=0; i<range; i++) {
      offset[i] = sum;
      sum += bucket[i];
    }
  }
}

__global__ void writeSortedKey(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    for (int j=0; j<bucket[i]; j++) {
      key[offset[i]+j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset;
  CUDA_CHECK(cudaMallocManaged(&key, n*sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&bucket, range*sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&offset, range*sizeof(int)));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  const int M = 128;
  clearBucket<<<(range+M-1)/M,M>>>(bucket, range);
  countBucket<<<(n+M-1)/M,M>>>(key, bucket, n);
  makeOffset<<<1,1>>>(bucket, offset, range);
  writeSortedKey<<<(range+M-1)/M,M>>>(key, bucket, offset, range);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  CUDA_CHECK(cudaFree(key));
  CUDA_CHECK(cudaFree(bucket));
  CUDA_CHECK(cudaFree(offset));
}
