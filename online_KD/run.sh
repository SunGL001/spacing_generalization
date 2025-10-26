for i in {0}
do
python ./train_space.py \
    -gpu_ids  '0,1' \
    -wandb_entity [your_wandb_entity] \
    -wandb_project [your_wandb_project] \
    -net_teacher 'resnet18' \
    -net_student 'resnet18' \
    -downsample_factor 1.0 \
    -alpha 0.3 \
    -temp 3.0 \
    -lr 0.01 \
    -interval_rate 0.0 \
    -b 128 \
    -gpu \
    -feature_loss 'dist' \
    -dataset 'cifar100' &\

python ./train_space.py \
    -gpu_ids  '0,1' \
    -wandb_entity [your_wandb_entity] \
    -wandb_project [your_wandb_project] \
    -net_teacher 'resnet18' \
    -net_student 'resnet18' \
    -downsample_factor 1.0 \
    -alpha 0.3 \
    -temp 3.0 \
    -lr 0.01 \
    -interval_rate 1.5 \
    -b 128 \
    -gpu \
    -feature_loss 'dist' \
    -dataset 'cifar100'
done
