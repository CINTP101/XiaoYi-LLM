CUDA_VISIBLE_DEVICES=0 python -m legacy.test_pt \
    --model_name_or_path /home/gyl/project/Medical_Image_Analysis/R2GenCSR/Qwen/Qwen1.5-1.8B-Chat \
    --test_file_dir /home/gyl/DataSets/medical/pretrain/test_encyclopedia.json \
    --load_in_4bit True \
    --batch_size 8 \
    --max_new_tokens 256 \
    --num_generate_examples 5 \
    --use_chinese_char_split True