CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tda_cifar10 \
--adaptation tda\
--distributed \
--dataset cifar10 \
--shift_type original \
--steps 10 \
--seeds 42 \
--batch_size 1 \
--closed_set \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tda_cifar10c \
--adaptation tda \
--distributed \
--dataset cifar10c \
--shift_type all \
--steps 10 \
--seeds 42 \
--batch_size 1 \
--closed_set \
