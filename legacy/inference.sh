CUDA_VISIBLE_DEVICES=1 python -m legacy.inference \
    --base_model /home/gyl/project/Medical_Image_Analysis/R2GenCSR/Qwen/Qwen1.5-1.8B-Chat \
    --lora_model /home/gyl/project/MedicalGPT/outputs-sft-qwen-v1/checkpoint-32 \
    --interactive