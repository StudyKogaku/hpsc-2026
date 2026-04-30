#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0.0f;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 fx_vec = _mm512_setzero_ps();
    __m512 fy_vec = _mm512_setzero_ps();

    for(int j=0; j<N; j+=16) {
      __m512 xj = _mm512_load_ps(&x[j]);
      __m512 yj = _mm512_load_ps(&y[j]);
      __m512 mj = _mm512_load_ps(&m[j]);

      // rx = x[i] - x[j], ry = y[i] - y[j]
      __m512 rx = _mm512_sub_ps(xi, xj);
      __m512 ry = _mm512_sub_ps(yi, yj);

      // r^2 = rx^2 + ry^2
      __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx),
                                _mm512_mul_ps(ry, ry));

      // i != j に対応する mask
      __m512i idx = _mm512_setr_epi32(
          j+0, j+1, j+2, j+3,
          j+4, j+5, j+6, j+7,
          j+8, j+9, j+10, j+11,
          j+12, j+13, j+14, j+15
      );
      __m512i ivec = _mm512_set1_epi32(i);
      __mmask16 mask = _mm512_cmpneq_epi32_mask(idx, ivec);

      // inv_r = 1 / r
      // r = 0 の成分は mask で後から無効化する
      __m512 inv_r = _mm512_rsqrt14_ps(r2);

      // inv_r3 = 1 / r^3
      __m512 inv_r2 = _mm512_mul_ps(inv_r, inv_r);
      __m512 inv_r3 = _mm512_mul_ps(inv_r2, inv_r);

      // coeff = m[j] / r^3
      __m512 coeff = _mm512_mul_ps(mj, inv_r3);

      // fx[i] -= rx * m[j] / r^3
      // fy[i] -= ry * m[j] / r^3
      __m512 dfx = _mm512_mul_ps(rx, coeff);
      __m512 dfy = _mm512_mul_ps(ry, coeff);

      fx_vec = _mm512_mask_sub_ps(fx_vec, mask, fx_vec, dfx);
      fy_vec = _mm512_mask_sub_ps(fy_vec, mask, fy_vec, dfy);
    }

    fx[i] = _mm512_reduce_add_ps(fx_vec);
    fy[i] = _mm512_reduce_add_ps(fy_vec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
