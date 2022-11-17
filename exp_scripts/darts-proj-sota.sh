#!/bin/bash
script_name=`basename "$0"` # 反引号(也可以用"$()")命令替换：执行反引号中的命令，basename返回文件的名字（去掉目录名)
id=${script_name%.*}    # %表明对script_name变量截取左边的，#号是截取右边的，这里是截取从右往左数第一个碰到点左面的
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}

## dev mode
space=${space:-s5}
resume_epoch=${resume_epoch:-50}
resume_expid=${resume_expid:-'search-darts-sota-s5-2'}
crit=${crit:-'acc'}

# 这里读取 --resume_expid search-darts-sota-s5-2，对第一个变量去掉--，申明一个 名为resume_expid的变量，幅值为search-darts-sota-s5-2
while [ $# -gt 0 ]; do  # 大于greater than >
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"    # / / 中间是删除的变量
        declare $param="$2"     # 申明变量， 变量名为param
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'resume_epoch:' $resume_epoch 'resume_expid' $resume_expid
echo 'proj crit:' $crit
echo 'gpu:' $gpu

cd ../sota/cnn
python train_search.py \
    --method darts-proj \
    --search_space $space --dataset $dataset \
    --seed $seed --save $id --gpu $gpu \
    --resume_epoch $resume_epoch --resume_expid $resume_expid --dev proj \
    --edge_decision random \
    --proj_crit_normal $crit --proj_crit_reduce $crit --proj_crit_edge $crit --proj_intv 5 \
    # --log_tag debug --fast \

## bash darts-proj-sota.sh --resume_expid search-darts-sota-debug-s5-2