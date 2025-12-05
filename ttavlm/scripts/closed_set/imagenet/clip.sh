CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name clip_imagenet \
--adaptation source \
--distributed \
--dataset imagenet \
--shift_type original \
--steps 10 \
--seeds 42 \
--closed_set \

CUDA_VISIBLE_DEVICES=0,1 python ttavlm/main.py \
--exp_name clip_imagenetc \
--adaptation source \
--distributed \
--dataset imagenet \
--shift_type all \
--steps 10 \
--seeds 42 \
--closed_set \
