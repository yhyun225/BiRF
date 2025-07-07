accelerate launch \
    --num_processes 3 \
    --multi_gpu \
    preprocess_dataset.py \
    --batch_size_per_gpu 4 \
    --mixed_precision 'fp16'