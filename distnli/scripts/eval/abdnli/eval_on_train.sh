source setup.sh

s=0

python src/multi_choice_tasks/eval_ontest.py \
    --model_class_name "roberta-large" \
    --load_model_path "[model path]/model.pt" \
    -s $s \
    -n 1 \
    -g 4 \
    -nr 0 \
    --total_step 3000 \
    --max_length 128 \
    --warmup_steps -1 \
    --gradient_accumulation_steps 4 \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 32 \
    --save_prediction \
    --save_checkpoint \
    --learning_rate 5e-6 \
    --eval_frequency 300 \
    --experiment_name "roberta-large|alphaNLI|evalTrains${s}" \
    --max_grad_norm 0.0 \
    --eval_dataset_name train \
    --number_of_choices 2
