export PYTHONPATH=./:$PYTHONPATH
checkpoint_dir=~/models/adv_interp_model/
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_dir=$checkpoint_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --batch_size_test=80 \
    --resume