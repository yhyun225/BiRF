accelerate launch \
    --num_processes 3 \
    --multi_gpu \
    main.py \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --i_print 1 \
    --i_log 1 \

    # --multi-gpu