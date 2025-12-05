CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_cifar100 \
--adaptation tent \
--distributed \
--dataset cifar100 \
--shift_type original \
--score_type max_prob \
--beta_tta 1.0 \
--beta_reg 0.0 \
--steps 10 \
--seeds 42 \
--closed_set \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_cifar100c \
--adaptation tent \
--distributed \
--dataset cifar100c \
--shift_type all \
--score_type max_prob \
--beta_tta 1.0 \
--beta_reg 0.0 \
--steps 10 \
--seeds 42 \
--closed_set \
