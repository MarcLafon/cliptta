CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name unient_open_imagenet \
--adaptation unient \
--distributed \
--dataset imagenet \
--ood_dataset svhn \
--shift_type original \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name unient_open_imagenetc \
--adaptation unient \
--distributed \
--dataset imagenetc \
--ood_dataset svhnc \
--shift_type all \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \
