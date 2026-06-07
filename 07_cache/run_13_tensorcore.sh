#!/bin/sh
#$ -cwd
#$ -l gpu_1=1            # GPUリソース(H100)を1枚要求
#$ -l h_rt=0:10:00       # 実行時間を指定（例: 10分）
#$ -N tensorcore_h100    # ジョブ名を指定（任意）

# 必要なモジュールをロード
# TSUBAME の環境では qsub から実行すると module が未初期化の場合があるため明示する．
. /etc/profile.d/modules.sh
module purge
module load cuda

# 提出版は Hopper WGMMA/TMA を使うため sm_90a 向けにコンパイルする．
# CONVERT_EACH_ITER=0 は，ベンチマーク中に A/B が変わらないため half temporary を一度だけ生成して再利用する設定である．
nvcc 13_tensorcore.cu -O3 -std=c++20 -gencode arch=compute_90a,code=sm_90a -Xptxas -O3,-v -Xcompiler "-O3 -fopenmp" -lcublas -lcuda

./a.out

# 実行例:
# cd /gs/fs/tga-hpc-lecture/uk07026/hpsc-2026/07_cache
# qsub -g tga-hpc-lecture run_13_tensorcore.sh
