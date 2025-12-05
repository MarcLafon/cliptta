CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tda_imagenet \
--adaptation tda\
--distributed \
--dataset imagenet \
--shift_type original \
--steps 10 \
--seeds 42 \
--batch_size 1 \
--closed_set \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name tda_imagenetc \
--adaptation tda \
--distributed \
--dataset imagenetc \
--shift_type all \
--steps 10 \
--seeds 42 \
--batch_size 1 \
--closed_set \
