diskpath=/mnt/disk2
exp_path=${diskpath}/experiments
colmap_path=${diskpath}/360
colmap_path_augmented=${diskpath}/360_augmented
exp_name=360_augmented_scheduler_n4
# for scene in bicycle flowers garden stump treehill
for scene in bicycle flowers garden stump treehill
do
    python train_scheduler.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/${exp_name}/${scene} \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters 4

    python render.py -m ${exp_path}/${exp_name}/${scene} --skip_train

    python metrics.py -m ${exp_path}/${exp_name}/${scene}
done

for scene in bonsai counter kitchen room
do
    python train_scheduler.py -s ${colmap_path_augmented}/${scene} -m ${exp_path}/${exp_name}/${scene} \
    -i images_2 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters 4

    python render.py -m ${exp_path}/${exp_name}/${scene} --skip_train

    python metrics.py -m ${exp_path}/${exp_name}/${scene}
done

