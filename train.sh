
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    main.py \
    --batch_size_per_gpu 12 \
    --gradient_accumulation_steps 2 \
    --i_print 10 \
    --i_log 100 \
    --i_save 5000 \
    # --wandb

# accelerate launch \
#     --num_processes 1 \
#     main.py \
#     --batch_size_per_gpu 2 \
#     --gradient_accumulation_steps 8 \
#     --i_print 1 \
#     --i_log 10 \
#     --i_save 1000
