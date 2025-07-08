accelerate launch \
    --num_processes 3 \
    --multi_gpu \
    main.py \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --i_print 1 \
    --i_log 10 \
    --i_save 1000

# accelerate launch \
#     --num_processes 1 \
#     main.py \
#     --batch_size_per_gpu 2 \
#     --gradient_accumulation_steps 8 \
#     --i_print 1 \
#     --i_log 10 \
#     --i_save 1000
