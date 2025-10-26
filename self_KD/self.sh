for i in {0}
do
python ./train.py \
    --gpu_ids  '0,1' \
    --wandb_entity [your_wandb_entity] \
    --wandb_project SKD \
    --model 'resnet18' \
    --interval_rate 0.0 \
    --dataset 'cifar100' &\

python ./train.py \
    --gpu_ids  '0,1' \
    --wandb_entity [your_wandb_entity] \
    --wandb_project SKD \
    --model 'resnet18' \
    --interval_rate 4.0 \
    --dataset 'cifar100' 
done
