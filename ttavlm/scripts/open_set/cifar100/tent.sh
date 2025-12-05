CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_open_cifar100 \
--adaptation tent \
--distributed \
--dataset cifar100 \
--ood_dataset svhn \
--shift_type original \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_open_cifar100c \
--adaptation tent \
--distributed \
--dataset cifar100c \
--ood_dataset svhnc \
--shift_type all \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \
