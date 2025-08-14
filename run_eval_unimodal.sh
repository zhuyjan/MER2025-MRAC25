#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 定义feat和seed的取值范围
feats=("InternVL_2_5_HiCo_R16-UTT")
seeds=(0 1 2 3 4)

# 外层循环遍历feat
for feat in "${feats[@]}"; do
    # 内层循环遍历seed
    for seed in "${seeds[@]}"; do
        echo "Running with seed=$seed, feat=$feat"
        
        # 执行命令
        python train.py --seed $seed \
                        --dataset "seed${seed}" \
                        --emo_rule "MER" \
                        --save_model \
                        --save_root "./saved/${feat}_3" \
                        --feat "['${feat}', '${feat}', '${feat}']" \
                        --lr 0.0001 \
                        --batch_size 512 \
                        --num_workers 4 \
                        --epochs 80 \
        
        # 检查命令是否执行成功
        if [ $? -eq 0 ]; then
            echo "Command executed successfully for seed=$seed, feat=$feat"
        else
            echo "Error executing command for seed=$seed, feat=$feat"
            exit 1
        fi
    done
done