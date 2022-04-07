source setup.sh

i=0

python src/multi_choice_tasks/training_soft.py \
    --model_class_name "roberta-large" \
    -s $((i+10)) \
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
    --experiment_name "roberta-large|alphaNLI|new_relabel_s${i}" \
    --max_grad_norm 0.0 \
    --train_dataset_name relabel_s${i} \
    --number_of_choices 2
