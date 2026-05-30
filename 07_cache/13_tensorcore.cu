// 提出版: FP16 WGMMA/TMA double buffering 版．
// FP32 入力は GPU 上で一度だけ half temporary に変換し，WGMMA f32.f16.f16 で計算する．
// TSUBAME H100 では sm_90a 向けにコンパイルする必要がある．
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <utility>
#include <chrono>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
namespace ptx = cuda::ptx;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#define STR2(x) #x
#define STR(x) STR2(x)
__constant__ CUtensorMap g_tma_map_a;
__constant__ CUtensorMap g_tma_map_b;

#ifndef TNSPA
#define TNSPA 1
#endif
#ifndef TNSPB
#define TNSPB 0
#endif

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult r=(call); if(r!=CUDA_SUCCESS){const char* n=nullptr; const char* msg=nullptr; cuGetErrorName(r,&n); cuGetErrorString(r,&msg); printf("CU %s:%d %s %s\n",__FILE__,__LINE__, n?n:"?", msg?msg:"?"); exit(1);} } while(0)

static void* checked_malloc(size_t bytes) {
  void* p = malloc(bytes);
  if (!p) { fprintf(stderr, "malloc failed: %zu bytes\n", bytes); exit(1); }
  return p;
}

// FP32 入力を half temporary に変換する．入力配列そのものは FP32 のまま保持する．
__global__ void convert_to_half_kernel(size_t n, const float* __restrict__ src, half* __restrict__ dst) {
  size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (base + 3 < n) {
    float4 v = *reinterpret_cast<const float4*>(&src[base]);
    reinterpret_cast<half2*>(&dst[base + 0])[0] = __float22half2_rn(make_float2(v.x, v.y));
    reinterpret_cast<half2*>(&dst[base + 2])[0] = __float22half2_rn(make_float2(v.z, v.w));
  } else {
    for (int i = 0; i < 4; i++) {
      size_t idx = base + i;
      if (idx < n) dst[idx] = __float2half(src[idx]);
    }
  }
}

__device__ __forceinline__ uint64_t make_gmma_desc(const void* ptr, int leading_bytes, int stride_bytes, int swizzle_mode) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0;
  desc |= (uint64_t)((addr >> 4) & 0x3fff);
  desc |= (uint64_t)(((leading_bytes >> 4) & 0x3fff)) << 16;
  desc |= (uint64_t)(((stride_bytes >> 4) & 0x3fff)) << 32;
  // WGMMA descriptor の bits 62-63 は swizzle mode．1 が 128B swizzle．
  desc |= (uint64_t(swizzle_mode) & 0x3ull) << 62;
  // 1024B 境界に揃えるので matrix base offset は 0 のままにする．
  return desc;
}

__device__ __forceinline__ int kmajor_index_64x16(int mn, int k) {
  return (mn & 7) * 8 + (k & 7) + (mn >> 3) * 128 + (k >> 3) * 64;
}
__device__ __forceinline__ int mnmajor_index_64x16(int mn, int k) {
  return (mn & 7) + (k & 7) * 8 + (mn >> 3) * 128 + (k >> 3) * 64;
}

__device__ __forceinline__ void wgmma64x128(uint64_t da, uint64_t db, float* d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,"
      " %8,  %9,  %10,  %11,  %12,  %13,  %14,  %15,"
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,"
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,"
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,"
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,"
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,"
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
    " %64, %65, p, 1, 1, " STR(TNSPA) ", " STR(TNSPB) ";\n"
    "}\n"
    : "+f"(d[0]),
      "+f"(d[1]),
      "+f"(d[2]),
      "+f"(d[3]),
      "+f"(d[4]),
      "+f"(d[5]),
      "+f"(d[6]),
      "+f"(d[7]),
      "+f"(d[8]),
      "+f"(d[9]),
      "+f"(d[10]),
      "+f"(d[11]),
      "+f"(d[12]),
      "+f"(d[13]),
      "+f"(d[14]),
      "+f"(d[15]),
      "+f"(d[16]),
      "+f"(d[17]),
      "+f"(d[18]),
      "+f"(d[19]),
      "+f"(d[20]),
      "+f"(d[21]),
      "+f"(d[22]),
      "+f"(d[23]),
      "+f"(d[24]),
      "+f"(d[25]),
      "+f"(d[26]),
      "+f"(d[27]),
      "+f"(d[28]),
      "+f"(d[29]),
      "+f"(d[30]),
      "+f"(d[31]),
      "+f"(d[32]),
      "+f"(d[33]),
      "+f"(d[34]),
      "+f"(d[35]),
      "+f"(d[36]),
      "+f"(d[37]),
      "+f"(d[38]),
      "+f"(d[39]),
      "+f"(d[40]),
      "+f"(d[41]),
      "+f"(d[42]),
      "+f"(d[43]),
      "+f"(d[44]),
      "+f"(d[45]),
      "+f"(d[46]),
      "+f"(d[47]),
      "+f"(d[48]),
      "+f"(d[49]),
      "+f"(d[50]),
      "+f"(d[51]),
      "+f"(d[52]),
      "+f"(d[53]),
      "+f"(d[54]),
      "+f"(d[55]),
      "+f"(d[56]),
      "+f"(d[57]),
      "+f"(d[58]),
      "+f"(d[59]),
      "+f"(d[60]),
      "+f"(d[61]),
      "+f"(d[62]),
      "+f"(d[63])
    : "l"(da), "l"(db), "r"(1));
}

// TMA で half temporary から shared memory に 128x16 の A と 16x128 の B を読み込む．
// その後，WGMMA が期待する簡易 swizzle へ shared memory 内で詰め替えてから wgmma を実行する．
// TMA 直結 layout ではないため高性能版ではなく，再挑戦用の正しさ優先プロトタイプである．
#ifndef TILE_K_LOAD
#define TILE_K_LOAD 64
#endif
#ifndef A_K_STEP_BYTES
#define A_K_STEP_BYTES 2048
#endif
#ifndef B_K_STEP_BYTES
#define B_K_STEP_BYTES 32
#endif
#ifndef DESCA_LBO
#define DESCA_LBO 16
#endif
#ifndef DESCA_SBO
#define DESCA_SBO 1024
#endif
#ifndef DESCB_LBO
#define DESCB_LBO 16
#endif
#ifndef DESCB_SBO
#define DESCB_SBO 1024
#endif
#ifndef GMMA_SWIZZLE_MODE
#define GMMA_SWIZZLE_MODE 1
#endif
#ifndef TMA_SWIZZLE_KIND
#define TMA_SWIZZLE_KIND CU_TENSOR_MAP_SWIZZLE_128B
#endif

#ifndef CONVERT_EACH_ITER
#define CONVERT_EACH_ITER 0
#endif

// 1 stage あたりの shared memory 使用量．K64 の A/B tile を 1024B 境界に載せる．
constexpr int CTA_M = 256;
constexpr int CTA_N = 128;
constexpr int WGMMA_M_TILE = 64;
constexpr int WGMMA_N_TILE = 128;
constexpr int NUM_WARPGROUPS = 4;
constexpr int STAGES = 2;
constexpr int A_STAGE_ELEMS = NUM_WARPGROUPS * WGMMA_M_TILE * TILE_K_LOAD;
constexpr int B_STAGE_ELEMS = TILE_K_LOAD * WGMMA_N_TILE;
constexpr int STAGE_ELEMS = A_STAGE_ELEMS + B_STAGE_ELEMS;

__device__ __forceinline__ barrier_t::arrival_token issue_tma_stage(barrier_t& bar,
                                                                    half* stage_base,
                                                                    int off_m_base,
                                                                    int off_n,
                                                                    int k0) {
  half* As_all = stage_base;
  half* Bs = stage_base + A_STAGE_ELEMS;
  if (threadIdx.x == 0) {
    int32_t coord_a0[2] = {off_m_base + 0,   k0};
    int32_t coord_a1[2] = {off_m_base + 64,  k0};
    int32_t coord_a2[2] = {off_m_base + 128, k0};
    int32_t coord_a3[2] = {off_m_base + 192, k0};
    int32_t coord_b[2]  = {k0, off_n};
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              As_all + 0 * WGMMA_M_TILE * TILE_K_LOAD, &g_tma_map_a, coord_a0,
                              cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              As_all + 1 * WGMMA_M_TILE * TILE_K_LOAD, &g_tma_map_a, coord_a1,
                              cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              As_all + 2 * WGMMA_M_TILE * TILE_K_LOAD, &g_tma_map_a, coord_a2,
                              cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              As_all + 3 * WGMMA_M_TILE * TILE_K_LOAD, &g_tma_map_a, coord_a3,
                              cuda::device::barrier_native_handle(bar));
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                              Bs, &g_tma_map_b, coord_b,
                              cuda::device::barrier_native_handle(bar));
    return cuda::device::barrier_arrive_tx(bar, 1,
                                           A_STAGE_ELEMS * int(sizeof(half)) +
                                           B_STAGE_ELEMS * int(sizeof(half)));
  }
  return bar.arrive();
}

__device__ __forceinline__ void wait_tma_stage(barrier_t& bar,
                                               barrier_t::arrival_token token) {
  bar.wait(std::move(token));
  // TMA は async proxy 経由で shared memory を書くため，WGMMA が読む前に proxy fence を入れる．
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void compute_wgmma_stage(half* stage_base, int wg, float* acc) {
  half* As_all = stage_base;
  half* Bs = stage_base + A_STAGE_ELEMS;
  half* As_wg = As_all + wg * WGMMA_M_TILE * TILE_K_LOAD;

#pragma unroll
  for (int ks = 0; ks < TILE_K_LOAD; ks += 16) {
    const char* ap = reinterpret_cast<const char*>(As_wg) + (ks / 16) * A_K_STEP_BYTES;
    const char* bp = reinterpret_cast<const char*>(Bs) + (ks / 16) * B_K_STEP_BYTES;
    uint64_t da = make_gmma_desc(ap, DESCA_LBO, DESCA_SBO, GMMA_SWIZZLE_MODE);
    uint64_t db = make_gmma_desc(bp, DESCB_LBO, DESCB_SBO, GMMA_SWIZZLE_MODE);
    wgmma64x128(da, db, acc);
  }
  // この実装では accumulator と shared memory stage を安全に再利用するため，K64 ごとに完了待ちする．
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  __syncthreads();
}

__global__ __launch_bounds__(512)
void kernel_wgmma_tma_direct(int dim_m, int dim_n, int dim_k,
                             float* __restrict__ C) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t bars[STAGES];
  if (threadIdx.x == 0) {
    init(&bars[0], blockDim.x);
    init(&bars[1], blockDim.x);
  }
  __syncthreads();

  int wg = threadIdx.x >> 7;
  int tid = threadIdx.x & 127;
  int off_m_base = blockIdx.x * CTA_M;
  int off_n = blockIdx.y * CTA_N;
  int off_m = off_m_base + wg * WGMMA_M_TILE;
  half* smem_half = reinterpret_cast<half*>(smem_raw);

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; i++) acc[i] = 0.0f;

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  // 最初の stage だけは計算前に読み込み完了を待つ．以降は計算中に次 stage を TMA で先読みする．
  auto token0 = issue_tma_stage(bars[0], smem_half + 0 * STAGE_ELEMS, off_m_base, off_n, 0);
  wait_tma_stage(bars[0], std::move(token0));

  for (int k0 = 0; k0 < dim_k; k0 += TILE_K_LOAD) {
    int stage = (k0 / TILE_K_LOAD) & 1;
    int next_k0 = k0 + TILE_K_LOAD;
    int next_stage = stage ^ 1;
    half* cur_base = smem_half + stage * STAGE_ELEMS;

    if (next_k0 < dim_k) {
      // 次の K64 tile を別 stage に非同期ロードしながら，現在の stage を WGMMA で計算する．
      auto token_next = issue_tma_stage(bars[next_stage],
                                        smem_half + next_stage * STAGE_ELEMS,
                                        off_m_base, off_n, next_k0);
      compute_wgmma_stage(cur_base, wg, acc);
      wait_tma_stage(bars[next_stage], std::move(token_next));
    } else {
      compute_wgmma_stage(cur_base, wg, acc);
    }
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  int t0 = tid & 3;
  int t1 = (tid >> 2) & 7;
  int t2 = (tid >> 5) & 3;
#pragma unroll
  for (int v = 0; v < 64; v++) {
    int v0 = v & 1;
    int v1 = (v >> 1) & 1;
    int v2 = (v >> 2) & 15;
    int flat = t0 * 128 + t1 + t2 * 16 + v0 * 64 + v1 * 8 + v2 * 512;
    int m = flat & 63;
    int n = flat >> 6;
    C[(off_n + n) * dim_m + off_m + m] = acc[v];
  }
  if (threadIdx.x == 0) {
    (&bars[0])->~barrier_t();
    (&bars[1])->~barrier_t();
  }
}


static PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled_ptr() {
  cudaDriverEntryPointQueryResult driver_status;
  void* ptr = nullptr;
  CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000,
                                              cudaEnableDefault, &driver_status));
  if (driver_status != cudaDriverEntryPointSuccess || ptr == nullptr) {
    fprintf(stderr, "cuTensorMapEncodeTiled entry point is not available\n");
    exit(1);
  }
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}

static CUtensorMap make_half_tma_map(void* base, uint64_t dim0, uint64_t dim1,
                                     uint64_t stride_dim1_bytes,
                                     uint32_t box0, uint32_t box1) {
  CUtensorMap map{};
  auto encode = get_cuTensorMapEncodeTiled_ptr();
  constexpr uint32_t rank = 2;
  uint64_t global_dim[rank] = {dim0, dim1};
  uint64_t global_stride[rank - 1] = {stride_dim1_bytes};
  uint32_t box_dim[rank] = {box0, box1};
  uint32_t elem_stride[rank] = {1, 1};
  CUresult res = encode(&map,
                        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                        rank,
                        base,
                        global_dim,
                        global_stride,
                        box_dim,
                        elem_stride,
                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                        TMA_SWIZZLE_KIND,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_CU(res);
  return map;
}

int main(int argc, char** argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  int Nt = 5;
  if (argc >= 4) { m = atoi(argv[1]); k = atoi(argv[2]); n = atoi(argv[3]); }
  if (argc >= 5) Nt = atoi(argv[4]);
  if ((m % 256) != 0 || (n % 128) != 0 || (k % TILE_K_LOAD) != 0) {
    fprintf(stderr, "matrix sizes must be multiples of 256,128,TILE_K_LOAD.\n");
    return 1;
  }

  size_t bytes_a = size_t(m) * k * sizeof(float);
  size_t bytes_b = size_t(k) * n * sizeof(float);
  size_t bytes_c = size_t(m) * n * sizeof(float);
  float* A = static_cast<float*>(checked_malloc(bytes_a));
  float* B = static_cast<float*>(checked_malloc(bytes_b));
  float* C = static_cast<float*>(checked_malloc(bytes_c));
  float* C2 = static_cast<float*>(checked_malloc(bytes_c));
  srand48(0);
  for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) A[(size_t)i * k + j] = drand48();
  for (int i = 0; i < k; i++) for (int j = 0; j < n; j++) B[(size_t)i * n + j] = drand48();

  float *dA, *dB, *dC, *dC2;
  half *dAh, *dBh;
  CHECK_CUDA(cudaMalloc(&dA, bytes_a));
  CHECK_CUDA(cudaMalloc(&dB, bytes_b));
  CHECK_CUDA(cudaMalloc(&dC, bytes_c));
  CHECK_CUDA(cudaMalloc(&dC2, bytes_c));
  CHECK_CUDA(cudaMalloc(&dAh, size_t(m) * k * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dBh, size_t(k) * n * sizeof(half)));
  CHECK_CUDA(cudaMemcpy(dA, A, bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, B, bytes_b, cudaMemcpyHostToDevice));

  auto convert_inputs = [&]() {
    constexpr int threads = 256;
    size_t num_a = size_t(m) * k;
    size_t num_b = size_t(k) * n;
    convert_to_half_kernel<<<(num_a + size_t(threads) * 4 - 1) / (size_t(threads) * 4), threads>>>(num_a, dA, dAh);
    CHECK_CUDA(cudaGetLastError());
    convert_to_half_kernel<<<(num_b + size_t(threads) * 4 - 1) / (size_t(threads) * 4), threads>>>(num_b, dB, dBh);
    CHECK_CUDA(cudaGetLastError());
  };
  convert_inputs();
  CHECK_CUDA(cudaDeviceSynchronize());

  CUtensorMap map_a = make_half_tma_map(dAh, m, k, uint64_t(m) * sizeof(half), 64, TILE_K_LOAD);
  CUtensorMap map_b = make_half_tma_map(dBh, k, n, uint64_t(k) * sizeof(half), TILE_K_LOAD, 128);
  CHECK_CUDA(cudaMemcpyToSymbol(g_tma_map_a, &map_a, sizeof(CUtensorMap)));
  CHECK_CUDA(cudaMemcpyToSymbol(g_tma_map_b, &map_b, sizeof(CUtensorMap)));

  cublasHandle_t h;
  CHECK_CUBLAS(cublasCreate(&h));
  CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
  float alpha = 1.0f, beta = 0.0f;
  int64_t flops = 2LL * m * n * k + 2LL * m * n;

  auto tic = chrono::steady_clock::now();
  for (int it = 0; it < Nt + 2; it++) {
    if (it == 2) tic = chrono::steady_clock::now();
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, dA, CUDA_R_32F, m,
                              dB, CUDA_R_32F, k,
                              &beta, dC, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  auto toc = chrono::steady_clock::now();
  double cublas_sec = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_gflops = double(flops) / cublas_sec / 1e9;

  dim3 grid(m / 256, n / 128);
  dim3 block(512);
  size_t dynamic_smem_bytes = size_t(STAGES) * STAGE_ELEMS * sizeof(half);
  CHECK_CUDA(cudaFuncSetAttribute(kernel_wgmma_tma_direct, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_smem_bytes)));
  tic = chrono::steady_clock::now();
  for (int it = 0; it < Nt + 2; it++) {
    if (it == 2) tic = chrono::steady_clock::now();
#if CONVERT_EACH_ITER
    convert_inputs();
#endif
    kernel_wgmma_tma_direct<<<grid, block, dynamic_smem_bytes>>>(m, n, k, dC2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  toc = chrono::steady_clock::now();
  double total_sec = chrono::duration<double>(toc - tic).count() / Nt;
  double total_gflops = double(flops) / total_sec / 1e9;


  // printf("config: WGMMA_TMA_DIRECT_K64_DB TILE_M=256 TILE_N=128 TILE_K=%d STAGES=%d BLOCK_THREADS=512 TNSPA=%d TNSPB=%d GMMA_SWIZZLE=%d A_STEP=%d B_STEP=%d CONVERT_EACH_ITER=%d\n", TILE_K_LOAD, STAGES, TNSPA, TNSPB, GMMA_SWIZZLE_MODE, A_K_STEP_BYTES, B_K_STEP_BYTES, CONVERT_EACH_ITER);
  printf("CUBLAS: %.2f Gflops, MY_KERNEL: %.2f Gflops, ratio: %.2f%%\n",
         cublas_gflops, total_gflops, 100.0 * total_gflops / cublas_gflops);

  CHECK_CUDA(cudaMemcpy(C, dC, bytes_c, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(C2, dC2, bytes_c, cudaMemcpyDeviceToHost));
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      int64_t idx = int64_t(m) * i + j;
      err += fabs(double(C[idx]) - double(C2[idx]));
    }
  }
  printf("error: %lf\n", err / n / m);

  free(A); free(B); free(C); free(C2);
  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(dC2));
  CHECK_CUDA(cudaFree(dAh)); CHECK_CUDA(cudaFree(dBh));
  CHECK_CUBLAS(cublasDestroy(h));
}
