CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name clip_cifar10 \
--adaptation source \
--distributed \
--dataset cifar10 \
--shift_type original \
--steps 10 \
--seeds 42 \
--closed_set \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name clip_cifar10c \
--adaptation source \
--distributed \
--dataset cifar10c \
--shift_type all \
--steps 10 \
--seeds 42 \
--closed_set \
