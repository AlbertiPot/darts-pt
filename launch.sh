# # search
# SEED=5555
# TASK='cls_mask'
# RATIO=0.1
# PATCH_SIZE=8
# TARGET_DATA='cifar10'

# WDIR=/data/gbc/Workspace/darts-pt
# PYTHONHOME=/root/miniconda3/envs/rookie/bin
# CUDA_VISIBLE_DEVICES=0
# cd ${WDIR}

# ${PYTHONHOME}/python ${WDIR}/sota/cnn/train_search.py \
# --method darts \
# --search_space s5 \
# --seed ${SEED} \
# --dataset ${TARGET_DATA} \
# --save ${TARGET_DATA}'_pt_search_v1'${TASK}'r'${RATIO}'p'${PATCH_SIZE}${TARGET_DATA} \
# --data '/data/gbc/Datasets/cifar/'${TARGET_DATA}'/' \
# --learning_rate_min 0 \
# --task ${TASK} \
# --mask_ratio ${RATIO} \
# --patch_size ${PATCH_SIZE}


# pt
SEED=9999
TASK='cls_mask'
RATIO=0.1
PATCH_SIZE=8
TARGET_DATA='cifar10'
RESUME_DIR='search-'${TARGET_DATA}'_pt_search_v1cls_maskr'${RATIO}'p'${PATCH_SIZE}${TARGET_DATA}'-s5-seed'${SEED}

WDIR=/data/gbc/Workspace/darts-pt
PYTHONHOME=/root/miniconda3/envs/rookie/bin

cd ${WDIR}
${PYTHONHOME}/python ${WDIR}/sota/cnn/train_search.py \
--method darts-proj \
--search_space s5 \
--dataset ${TARGET_DATA} \
--seed ${SEED} \
--save pt \
--resume_epoch 50 \
--resume_expid ${RESUME_DIR} \
--dev proj \
--edge_decision random \
--proj_crit_normal acc \
--proj_crit_reduce acc \
--proj_crit_edge acc \
--proj_intv 5 \
--data "/data/gbc/Datasets/cifar/"${TARGET_DATA} \
--learning_rate_min 0 \
--task ${TASK} \
--mask_ratio ${RATIO} \
--patch_size ${PATCH_SIZE} \
--log_tag "2losses_pt"