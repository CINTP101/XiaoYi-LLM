# 指定GPU运行
CUDA_VISIBLE_DEVICES=0 python -m legacy.test_sft \
    --base_model_path /home/gyl/project/Medical_Image_Analysis/R2GenCSR/Qwen/Qwen1.5-1.8B-Chat \
    --test_data_path data/finetune/sharegpt_zh_1K_format.jsonl \
    --load_in_4bit \
    --max_samples 100 \
    --max_new_tokens 256 \
    --batch_size 50 \
    --output_dir ./qwen_medical_test_results