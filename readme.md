# GatorTron Clinical Transformer Prediction

This package has been developed so researchers can easily use state-of-the-art transformer models and large language models for disease onset prediction from electronic health records (EHR).

A visualization technique is also implemented in this package, providing insight into important narrative features driving the prediction. 

This package provides end to end process from training to prediction.

<img src="https://github.com/user-attachments/assets/e9ac2dca-b89a-47ee-86d6-31fa2b4c6160" width="500" height="500">

## Dependency
The package is built on top of the Transformers developed by the HuggingFace.
We have the requirement.txt to specify the packages required to run the project.

## Available Models
- GatorTron-base (345m parameters) https://huggingface.co/UFNLP/gatortron-base
- GatorTron-medium (3.9b parameters) https://huggingface.co/UFNLP/gatortron-medium
- GatorTron-large (8.9b parameters) https://huggingface.co/UFNLP/gatortron-large
- BERT
- XLNet
- RoBERTa
- ALBERT
- DeBERTa
- Longformer
> We will keep adding new models.

## Before You Start
- Prerequisite
> To use the package for text classification, you need to provide unstructured text data with labels.

> Preprocessing is required to truncate the text under the required maximum token limitation for different models and ensure it is in the correct format.
Once the preprocessing is complete, you can run the package to obtain end-to-end classification prediction results.

- Data Format
> See sample_data dir for the train, dev, and test data format.

> The sample data is a small subset of the data prepared from the 2018 umass made1.0 challenge corpus.

> We did not provide a script for data preprocessing. You can follow our example data format to generate your own dataset. 

```
# Data Format: tsv file with 4 columns:
1. target_class: True
2. sentence: [s] Penicillin [e] .
3. entity_type: Drug
4. entity_id: id_1

Note: 
1) the entity between [s][e] is the paragraph we used for text classification
2) in the test.tsv, you can set all labels to neg or False or whatever, because we will not use the label anyway
3) We recommend to evaluate the test performance in a separate process based on prediction. (see **post-processing**)
4) We recommend using official evaluation scripts to do evaluation to make sure the results reported are reliable.
```
- Special Tags
> we use 2 special tags to identify entities
```
# The defaults tags we defined in the repo are
EN_START = "[s]"
EN_END = "[e]"

If you need to customize these tags, you can change them in config.py
```
- Visualization 
> LIME (Local Interpretable Model-agnostic Explanations) package is used in our package to visualize the important narrative features. For more detailed information, please refer to: https://github.com/marcotcr/lime

## Use Instrucation

- Training/ Evaluation/ Prediction
> Please refer to the wiki page for all details of the parameters
> [flag details](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction/wiki/all-parameters)

> highlight_index: accessing rows by index in test dataset; highlight_output_file: visualization output file path.

```shell script
export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=./new_modelzw
pof=./predictions.txt
log=./log.txt
log_highlight =./log_highlight.html

# NOTE: we have more options available, you can check our wiki for more information

python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model UFNLP/gatortron-medium \
		--data_dir $data_dir \
		--new_model_dir $nmd \
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
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--learning_rate 1e-5 \
		--num_train_epochs 3 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \
```

## Citation
Please cite our paper: https://arxiv.org/abs/2403.11425
```
@article{chen2024narrative,
  title={Narrative Feature or Structured Feature? A Study of Large Language Models to Identify Cancer Patients at Risk of Heart Failure},
  author={Chen, Ziyi and Zhang, Mengyuan and Ahmed, Mustafa Mohammed and Guo, Yi and George, Thomas J and Bian, Jiang and Wu, Yonghui},
  journal={arXiv preprint arXiv:2403.11425},
  year={2024}
}
```

## Contact
Please contact us or post an issue if you have any questions.
* Ziyi Chen (chenziyi@ufl.edu)
* Yonghui Wu (yonghui.wu@ufl.edu)

## Other Clinical Pre-trained Transformer Models
We have a series transformer models pre-trained on MIMIC-III.
You can find them here:
- https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz
- https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip
