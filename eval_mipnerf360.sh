#!/bin/bash
# 실험 설정
n_clusters=4
# n_turns=20
# 환경 변수 설정
diskpath="/home/cvnar"
exp_path="${diskpath}/gaussian-splatting/experiments"
# exp_name="360_scheduler_cluster${n_clusters}_turn${n_turns}"
exp_name="360_partialscheduler_cluster${n_clusters}"
colmap_path="${diskpath}/360"
colmap_path_augmented="${diskpath}/360_augmented"

for scene in $(ls "$colmap_path"); do
    echo "Processing scene: $scene"
    if [ "$scene" == "bicycle" ] || [ "$scene" == "flowers" ] || [ "$scene" == "garden" ] || [ "$scene" == "stump" ] || [ "$scene" == "treehill" ]; then
        images_folder="images_4"
    else
        images_folder="images_2"
    fi
    python train_partialscheduler.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/${exp_name}/${scene} \
    -i ${images_folder} \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters ${n_clusters} \
    # --n_turns ${n_turns}
    python render.py -m ${exp_path}/${exp_name}/${scene} --skip_train
    python metrics.py -m ${exp_path}/${exp_name}/${scene}
    python utils/experiment_utils.py ${scene} ${exp_name}
done
