source setup.sh

python src/multi_choice_tasks/training.py \
    --model_class_name "roberta-large" \
    -s 3 \
    -n 1 \
    -g 4 \
    -nr 0 \
    --epochs 3 \
    --max_length 128 \
    --warmup_steps -1 \
    --gradient_accumulation_steps 4 \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 32 \
    --save_prediction \
    --save_checkpoint \
    --learning_rate 5e-6 \
    --eval_frequency 1000 \
    --experiment_name "roberta-large|alphaNLIs3" \
    --max_grad_norm 0.0 \
    --number_of_choices 2
