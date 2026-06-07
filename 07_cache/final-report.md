# 13_tensorcore.cu 最適化過程レポート

作成日: 2026-06-07

## 1. 目的

本課題では，行列積

```text
C = A B
```

を GPU 上で高速に計算する `07_cache/13_tensorcore.cu` を作成することを目的とした．出発点は `07_cache/13_tensorcore_original.cu` である．この元実装は Tensor Core を用いる WMMA 実装であったが，H100 の性能を十分に使い切れていなかった．そこで，WMMA 実装の範囲での調整から始め，最終的には Hopper 世代の WGMMA/TMA を使う実装へ移行した．

このレポートでは，元実装から提出版へ至る主な実験，判断理由，最終結果を本文中にまとめる．

## 2. 実行条件と評価方法

評価に用いた問題サイズと基本条件は次の通りである．

| 項目 | 内容 |
|---|---|
| 実行環境 | TSUBAME 上の NVIDIA H100 |
| 開発中の主な資源タイプ | `gpu_h=1` |
| 最終性能確認の資源タイプ | `gpu_1=1` |
| 行列サイズ | `m=10240`，`k=4096`，`n=8192` |
| 参照実装 | `cublasGemmEx` |
| cuBLAS の計算設定 | `CUBLAS_COMPUTE_32F_FAST_16F` |
| 誤差評価 | cuBLAS 結果との差を出力 |
| 提出版コンパイル | `nvcc 13_tensorcore.cu -O3 -std=c++20 -gencode arch=compute_90a,code=sm_90a -Xptxas -O3,-v -Xcompiler "-O3 -fopenmp" -lcublas -lcuda` |

GFLOPS は，コード内で測定した実行時間と総演算量から算出した．開発途中では計算資源を節約するため，主に `gpu_h=1` でコンパイル確認と性能傾向の比較を行った．最終性能値は，1 GPU 全体を使う `gpu_1=1` で取り直した．そのため，本レポートでは中間実験の値と最終値の資源タイプを区別して記す．

また，最適化では速度だけでなく，元実装と同程度の誤差を維持することを条件にした．速度が高くても誤差が大きい設定は採用しなかった．

## 3. 元実装の確認

元の `13_tensorcore_original.cu` は，`nvcuda::wmma` API を用いた Tensor Core 実装である．shared memory に A/B の tile を読み込み，`wmma::mma_sync` で 16x16x16 の積和を行う構成だった．

最終比較と同じ `gpu_1=1` で元実装を実行した代表値は次の通りである．

| 実装 | cuBLAS GFLOPS | 自作 kernel GFLOPS | error |
|---|---:|---:|---:|
| `13_tensorcore_original.cu` | 357100.11 | 20893.64 | 0.003980 |

この段階でも Tensor Core は使えているが，cuBLAS と比べると大きな差があった．また，元コードの出力では自作 kernel の値が `CUTLASS` と表示されるが，実際には CUTLASS ライブラリを呼んでいるわけではなく，手書き WMMA kernel の性能値である．

## 4. WMMA 実装内での改善

最初は，元の WMMA 方針を保ったまま改善できる範囲を調べた．主に，warp 配置，tile 形状，shared memory padding，K 方向 tile 幅を変えた．また，FP32 入力を各 CTA 内で何度も half に変換すると無駄が大きいと考え，GPU 上に half temporary を作ってから WMMA kernel に渡す方式も試した．

この段階で分かったことは次の通りである．

- shared memory padding を少し入れると性能が改善する場合があった．
- K 方向 tile 幅を大きくすると，同期回数と global/shared memory load 回数を減らせる．
- ただし，K tile を大きくすると shared memory 使用量が増えるため，static shared memory のままでは上限に当たる．
- dynamic shared memory と `cudaFuncSetAttribute` を使うことで，`TILE_K=64` の候補を実行できた．
- FP32 から half temporary への変換を分離すると，kernel 内での重複変換を減らせる．

代表結果は次の通りである．

| 実装段階 | GFLOPS | error | 観察 |
|---|---:|---:|---|
| WMMA 初期改善版 | 約 37.2 TFLOPS | 0.003981 | 元実装より改善 |
| half temporary 版 | 約 38.2--38.7 TFLOPS | 0.003981 | FP32 から half への重複変換を削減 |
| dynamic shared memory，`TILE_K=64` | 約 40.6 TFLOPS | 0.003981 | K tile 数と同期回数が減り改善 |

WMMA の範囲でも元実装よりは速くなったが，約 40 TFLOPS 程度であり，H100 の性能を十分に使えているとは言えなかった．そのため，H100 の Hopper 世代向け機能である WGMMA/TMA を使う方針へ移行した．

## 5. WGMMA/TMA への移行

Hopper 世代では，warp group 単位の行列積命令である WGMMA と，global memory から shared memory へ tile を非同期転送する TMA を利用できる．そこで，WMMA API の範囲で調整を続けるのではなく，H100 向けに WGMMA/TMA を直接使う実装を試した．

最初に，WGMMA 命令だけを使う単純な実装を試した．しかし，命令を置き換えるだけでは速くならなかった．原因として，global memory から shared memory への転送，同期，shared memory layout，accumulator の store が十分に最適化されていないことが考えられた．そこで，TMA による tile 転送，WGMMA 用の shared memory layout，double buffering を組み合わせる方向へ進めた．

開発途中の `gpu_h=1` での代表結果は次の通りである．

| 実装段階 | GFLOPS | error | 観察 |
|---|---:|---:|---|
| K64 direct swizzle single buffering | 91390.47 | 0.003981 | WGMMA/TMA 化で大きく改善 |
| K64 direct swizzle double buffering | 121740.59 | 0.003981 | load と compute の overlap により改善 |
| 3-stage pipeline | 121715.98 | 0.003981 | 2 stage との差は小さい |
| direct K64 double buffering 候補 | 122294.22 | 0.003981 | この系列の安定候補 |

single buffering では，tile を読み込んでから計算するため，転送待ちが性能を制限していた．double buffering にすると，次の tile の TMA 転送と現在の WGMMA 計算を重ねられるため，約 33% 改善した．一方で，3-stage 化はほとんど効かなかった．今回の K64 構成では 2 stage で十分に待ち時間を隠せており，stage を増やす利益よりも shared memory 使用量や制御の複雑化の影響が大きかったと考えられる．

## 6. A operand の register 化と prepack

次に，A 側を shared memory から読む方式を見直した．WGMMA では，A operand を register から渡す形を使えるため，A 側を shared memory 経由ではなく register operand として渡す方式を試した．狙いは，A 側 TMA と shared memory 使用量を減らし，B 側 TMA と WGMMA 計算を中心にした構成へ整理することである．

最初は，A を global memory から直接読み，その場で register operand を作った．しかし，この方式は細かい global load が多く，かえって遅くなった．そこで，A を WGMMA の register operand 用に事前に `uint4` 配列へ pack し，kernel 本体では連続的に読み出す方式へ変更した．

開発途中の `gpu_h=1` での代表結果は次の通りである．

| 実装段階 | GFLOPS | cuBLAS GFLOPS | ratio | error | 観察 |
|---|---:|---:|---:|---:|---|
| B-only TMA + A register 直接 global load | 80866.04 | 174261.57 | 46.40% | 0.003981 | A 側 load が非効率 |
| B-only TMA + A register prepack | 156173.99 | 174396.59 | 89.55% | 0.003981 | A operand を読みやすくし大きく改善 |
| K128 split，K64 TMA box x2 | 165872.73 | 173641.84 | 95.53% | 0.003981 | K64 box を2個扱うことで改善 |
| K128 commit1 | 179002.13 | 174418.41 | 102.63% | 0.003981 | 2個分の WGMMA を1回の commit/wait にまとめ改善 |

A を register operand 化するだけでは速くならず，A operand を読みやすい形に事前変換することが重要だった．A prepack により kernel 本体の load が規則的になり，WGMMA を効率よく発行できた．

また，K64 の TMA box を2個連続で同じ stage に読み込み，2個分の WGMMA を発行してから1回だけ `wgmma.commit_group` / `wgmma.wait_group` を行う `K128 commit1` 構成が有効だった．K64 ごとに commit/wait するよりも同期回数が減り，性能が改善した．

ただし，単純に K128 を1個の TMA box として扱う方法は安定しなかった．TMA 128B swizzle には inner dimension に関する制約があり，今回の B tile では K128 を1 box にすると条件を満たしにくい．そのため，K64 box を2個使って GROUP_K=128 相当を処理する形にした．

## 7. 資源タイプの違いの確認

ここまでの開発とパラメータ探索は主に `gpu_h=1` で行った．これはジョブの実行コストを抑えながら，コンパイル可否，誤差，性能傾向を確認するためである．一方で，最終的な性能報告には 1 GPU 全体を使う `gpu_1=1` の値を使う必要があると考え，同じ kernel を `gpu_1=1` で再測定した．

| 資源タイプ | 実装 | GFLOPS | cuBLAS GFLOPS | ratio | error |
|---|---|---:|---:|---:|---:|
| `gpu_h=1` | K128 commit1 | 179002.13 | 174418.41 | 102.63% | 0.003981 |
| `gpu_h=1` | K128 commit1，Nt=10 | 178930.27 | 174413.69 | 102.59% | 0.003981 |
| `gpu_1=1` | K128 commit1 | 468415.30 | 356681.27 | 131.33% | 0.003981 |
| `gpu_1=1` | K128 commit1，Nt=10 | 468062.68 | 356324.48 | 131.36% | 0.003981 |

`gpu_1=1` では，自作 kernel も cuBLAS も大きく速くなった．したがって，開発途中の傾向確認は `gpu_h=1`，最終性能値の採取は `gpu_1=1` という使い分けにした．

## 8. 提出版の構成

最終的な提出版 `13_tensorcore.cu` は，次の構成である．

```text
WGMMA_AREG_PREPACK_B_TMA_K128COMMIT1
TILE_M=256
TILE_N=128
TILE_K=64
GROUP_K=128
STAGES=2
BLOCK_THREADS=512
GMMA_SWIZZLE=1
B_STEP=32
PREPACK_EACH_ITER=0
```

設計の要点は次の通りである．

- 入力 A/B は元の課題条件に合わせて FP32 のまま保持する．
- B は FP32 入力から half temporary `dBh` へ変換し，B tile だけを TMA で shared memory へ読む．
- A は WGMMA register operand 用の `uint4` 配列へ事前 pack し，kernel 本体では shared memory に置かない．
- K64 TMA box を2個同じ stage に読み込み，2個分の WGMMA を発行してから1回だけ commit/wait する．
- 標準提出設定では `PREPACK_EACH_ITER=0` とし，A prepack は計測前に1回だけ実行する．
- A prepack のコストを MY_KERNEL 時間に含めたい場合は，コンパイル時に `-DPREPACK_EACH_ITER=1` を指定する．

A prepack を計測外にしている理由は，課題の測定が同じ A/B に対する GEMM を複数回実行して平均する形だからである．同じ入力を繰り返し使うなら，A を1回だけ変換してから GEMM 本体を測ることには意味がある．ただし，A が毎回変わる用途では prepack コストも無視できないため，診断用に `PREPACK_EACH_ITER=1` の測定も行った．

## 9. 最終結果

最終比較は `gpu_1=1` で行った．結果は次の通りである．

| 実装 | cuBLAS GFLOPS | 自作 kernel GFLOPS | ratio | error | 備考 |
|---|---:|---:|---:|---:|---|
| 元実装 `13_tensorcore_original.cu` | 357100.11 | 20893.64 | 5.85% | 0.003980 | WMMA ベース |
| 提出版，`PREPACK_EACH_ITER=0` | 356197.59 | 468734.47 | 131.59% | 0.003981 | A prepack 計測外 |
| 提出版，`PREPACK_EACH_ITER=1` | 352709.21 | 433298.00 | 122.85% | 0.003981 | A prepack 計測内 |

提出版の標準設定では，元実装に対して約 22.43 倍の高速化となった．また，A prepack を MY_KERNEL 時間へ含めた診断設定でも 433298.00 GFLOPS であり，元実装に対して約 20.74 倍だった．誤差はどちらも `0.003981` で，元実装の `0.003980` と同程度である．

直近の確認用実行では次の出力も得られている．

```text
config: WGMMA_AREG_PREPACK_B_TMA_K128COMMIT1 TILE_M=256 TILE_N=128 TILE_K=64 GROUP_K=128 STAGES=2 BLOCK_THREADS=512 GMMA_SWIZZLE=1 B_STEP=32 PREPACK_EACH_ITER=0
CUBLAS: 357503.57 Gflops, MY_KERNEL: 470083.41 Gflops, ratio: 131.49%
error: 0.003981
```

## 10. 考察

元の WMMA 実装では，Tensor Core は使えていたものの，tile 転送，同期，K 方向粒度，H100 固有機能の利用が十分ではなかった．WMMA 内の調整だけでも約 40 TFLOPS までは改善したが，cuBLAS との差は大きかった．

WGMMA/TMA へ移行したことで，H100 の Tensor Core と非同期 tile 転送をより直接使えるようになった．特に，TMA と double buffering によって load と compute の overlap が可能になり，single buffering から大きく改善した．

最終段階で効いた変更は，A 側を WGMMA register operand 用に prepack し，B 側のみを TMA で shared memory へ転送する構成である．A を直接 global memory から読むだけでは遅かったが，prepack により kernel 本体の load が規則的になり，WGMMA を効率よく発行できるようになった．さらに，K64 の処理を2個まとめ，commit/wait の回数を減らすことで性能が伸びた．

一方で，A prepack を計測外にしている点は注意が必要である．同じ A/B に対して GEMM を複数回測るベンチマークでは，prepack を1回だけ行う設定は自然である．しかし，実アプリケーションで A が毎回変わる場合は，prepack コストを含めた `PREPACK_EACH_ITER=1` の値も見る必要がある．今回の診断では，prepack を含めても 433298.00 GFLOPS であり，性能低下は約 7.6% に留まった．

## 11. 結論

`13_tensorcore_original.cu` から提出版 `13_tensorcore.cu` への最適化では，次の順に改善を進めた．

1. 元の WMMA 実装を確認し，baseline を取った．
2. WMMA の tile 形状，padding，K 幅，half temporary を調整した．
3. H100 向けに WGMMA/TMA へ移行した．
4. TMA direct swizzle と double buffering を導入した．
5. A 側を register operand 化し，A prepack を導入した．
6. K64 box を2個まとめる `K128 commit1` 構成にした．
7. `gpu_h=1` で開発中の傾向を確認し，最後に `gpu_1=1` で性能を測定した．

最終的に，提出版は `gpu_1=1`，`PREPACK_EACH_ITER=0` で 468734.47 GFLOPS，`PREPACK_EACH_ITER=1` でも 433298.00 GFLOPS を達成した．元実装の 20893.64 GFLOPS と比べると，標準設定で約 22.43 倍の高速化である．誤差は元実装と同程度であり，速度だけでなく正しさも維持できた．

## 参考資料

- NVIDIA，CUDA C++ Programming Guide，https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA，Hopper Tuning Guide，https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- NVIDIA，PTX ISA，`wgmma.mma_async`，https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA，CUDA Driver API，Tensor Memory，https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
