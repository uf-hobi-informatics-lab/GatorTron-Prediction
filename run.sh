#!/bin/bash
#SBATCH --job-name=prediction    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xxxxxx@ufl.edu     # Where to send mail  
#SBATCH --ntasks=16                   # Run on a single CPU
#SBATCH --mem=80gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=prediction_%j.out   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1


module load python
module load conda
cd /orange/yonghui.wu/chenziyi/prediction/GatorTron-Clinical-Transformer-Prediction

#env
conda activate textclass

export CUDA_VISIBLE_DEVICES=2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
data_dir=/orange/yonghui.wu/chenziyi/prediction/GatorTron-Clinical-Transformer-Prediction/sample_data
nmd=./new_model_gatortron_test
pof=./gatortron_syn_sample_test.txt
log_highlight=./log_highlight_gatortron_sample_test.html
log=./log_gatortron_sample_test.txt


# NOTE: we have more options available, you can check our wiki for more information
python ./src/relation_extraction.py \
		--model_type megatron \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model UFNLP/gatortron-medium\
		--data_dir $data_dir \
		--new_model_dir $new_model_dir \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 512 \
		--cache_data \
		--do_train \
		--do_eval \
		--do_predict \
		--do_predict_highlight \
		--highlight_index 8\
		--highlight_output_file $log_highlight\
		--do_lower_case \
		--train_batch_size 1 \
		--eval_batch_size 1 \
		--learning_rate 1e-5 \
		--num_train_epochs 20 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file  $log\
