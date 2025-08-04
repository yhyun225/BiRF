
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    main.py \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 1 \
    --i_print 100 \
    --i_log 10000 \
    --i_save 10000 \
    # --wandb

# accelerate launch \
#     --num_processes 1 \
#     main.py \
#     --batch_size_per_gpu 2 \
#     --gradient_accumulation_steps 8 \
#     --i_print 1 \
#     --i_log 10 \
#     --i_save 1000
