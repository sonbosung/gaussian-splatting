#!/bin/bash
# 실험 설정
n_clusters=4
n_turns=20
# 환경 변수 설정
diskpath="/mnt/disk2"
exp_path="${diskpath}/auggs/experiments"
exp_name="360_scheduler_cluster${n_clusters}_turn${n_turns}"
colmap_path="${diskpath}/360"
colmap_path_augmented="${diskpath}/360_augmented"

for scene in $(ls "$colmap_path"); do
    echo "Processing scene: $scene"
    if [ "$scene" == "bicycle" ] || [ "$scene" == "flowers" ] || [ "$scene" == "garden" ] || [ "$scene" == "stump" ] || [ "$scene" == "treehill" ]; then
        images_folder="images_4"
    else
        images_folder="images_2"
    fi
    python train_scheduler.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/${exp_name}/${scene} \
    -i ${images_folder} \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters ${n_clusters} \
    --n_turns ${n_turns}
    python render.py -m ${exp_path}/${exp_name}/${scene} --skip_train
    python metrics.py -m ${exp_path}/${exp_name}/${scene}
    python experiment_utils.py ${scene} ${exp_name}
done
