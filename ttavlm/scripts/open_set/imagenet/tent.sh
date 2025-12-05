CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_open_imagenet \
--adaptation tent \
--distributed \
--dataset imagenet \
--ood_dataset places \
--shift_type original \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tent_open_imagenetc \
--adaptation tent \
--distributed \
--dataset imagenetc \
--ood_dataset placesc \
--shift_type all \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \
