subjects="cat"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/shyam/test"
export INSTANCE_DIR="/home/shyam/dataset_svdiff/${subjects}"


# unet parameters
factor=-1
lora_rank=4
lr=5e-4 
steps=2
decompose_both=False

accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks${subjects}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --seed="0" \
    --diffusion_model="sdxl" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --factor=$factor \
    --lora_rank=$lora_rank \
    --decompose_both=$decompose_both \
    --unet_tune_mlp \

python3 generator.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks${subjects}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --seed="0" \
    --diffusion_model="sdxl" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --factor=$factor \
    --lora_rank=$lora_rank \
    --decompose_both=$decompose_both \
    --unet_tune_mlp \