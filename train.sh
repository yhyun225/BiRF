accelerate launch \
    --num_processes 1 \
    main.py \
    --batch_size 2 \
    --gradient_accumulation_steps 2

    # --multi-gpu