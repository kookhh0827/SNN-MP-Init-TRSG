python train.py --workers 2 --epochs 200 --batch-size 16 --learning-rate 0.02 --weight_decay 5e-4 --momentum 0.9 \
                --T 4 --init_tau 2.0 --init_thr 1.0 --print-freq 100 \
                --model spiking --dataset imagenet --dataset_folder /SSD \
                --save_names resnet19_200epoch_128batch_0.10lr_5e-4wd_0.9mom_4t_2.0tau_1.0thr_cifar100
