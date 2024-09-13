"""
This script is used for training and test
"""

import shutil
from pathlib import Path

import numpy as np
import torch

from config import CONFIG_VERSION_NAME, MODEL_DICT, NEW_ARGS, SPEC_TAGS, VERSION
from data_processing.io_utils import pkl_load, pkl_save, save_json
import matplotlib.pyplot as plt

import os
import argparse
from lime.lime_text import LimeTextExplainer


# from data_utils import convert_examples_to_relation_extraction_features
from data_utils import (
    RelationDataFormatSepProcessor,
    RelationDataFormatUniProcessor,
    batch_to_model_input,
    features2tensors,
    relation_extraction_data_loader,
)
from packaging import version
from tqdm import tqdm, trange
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_relation_extraction_features,
)
from utils import acc_and_f1


class TaskRunner(object):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model_dict = MODEL_DICT
        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None
        self.test_data_loader_highlight = None
        self.data_processor = None
        self.new_model_dir_path = Path(self.args.new_model_dir)
        self.new_model_dir_path.mkdir(parents=True, exist_ok=True)
        self._use_amp_for_fp16_from = 0

    def task_runner_default_init(self):
        # set up data processor
        if self.data_processor is None:
            if self.args.data_format_mode == 0:
                self.data_processor = RelationDataFormatSepProcessor(
                    max_seq_len=self.args.max_seq_length, num_core=self.args.num_core
                )
            elif self.args.data_format_mode == 1:
                self.data_processor = RelationDataFormatUniProcessor(
                    max_seq_len=self.args.max_seq_length, num_core=self.args.num_core
                )
            else:
                raise NotImplementedError(
                    "Only support 0, 1 but get data_format_mode as {}".format(
                        self.args.data_format_mode
                    )
                )
        else:
            self.args.logger.warning(
                "Use user defined data processor: {}".format(self.data_processor)
            )

        self.data_processor.set_data_dir(self.args.data_dir)
        self.data_processor.set_header(self.args.data_file_header)

        # init or reload model
        if self.args.do_train:
            # init amp for fp16 (mix precision training)
            # _use_amp_for_fp16_from: 0 for no fp16; 1 for naive PyTorch amp; 2 for apex amp
            if self.args.fp16:
                self._load_amp_for_fp16()
            self._init_new_model()
        else:
            self._init_trained_model()

        # load data
        self.data_processor.set_tokenizer(self.tokenizer)
        self.data_processor.set_tokenizer_type(self.args.model_type)
        self.args.logger.info("data loader info: {}".format(self.data_processor))
        self._init_dataloader()

        if self.args.do_train:
            self._init_optimizer()

        self.args.logger.info("Model Config:\n{}".format(self.config))
        self.args.logger.info("All parameters:\n{}".format(self.args))

    def train(self):
        # create data loader
        self.args.logger.info("start training...")
        tr_loss = 0.0
        t_step = 1
        latest_best_score = 0.0
        # training loop
        epoch_loss_history = []
        eval_epoch_loss_history = []
        iteration_loss_history = []
        acc_history = []
        f1_history = []
        recall_history = []
        precison_history = []

        epoch_iter = trange(
            self.args.num_train_epochs, desc="Epoch", disable=not self.args.progress_bar
        )
        for epoch in epoch_iter:
            epoch_loss = 0
            epoch_num = 0

            batch_iter = tqdm(
                self.train_data_loader, desc="Batch", disable=not self.args.progress_bar
            )

            batch_total_step = len(self.train_data_loader)
            for step, batch in enumerate(batch_iter):
                self.model.train()
                self.model.zero_grad()

                batch_input = batch_to_model_input(
                    batch, model_type=self.args.model_type, device=self.args.device
                )

                if self.args.fp16 and self._use_amp_for_fp16_from == 1:
                    with self.amp.autocast():
                        batch_output = self.model(**batch_input)
                        loss = batch_output[0]
                else:
                    batch_output = self.model(**batch_input)
                    loss = batch_output[0]

                loss = loss / self.args.gradient_accumulation_steps # iteration average sample loss
                iteration_loss_history.append(loss.item()) #loss.item() iteration average loss
                epoch_loss += loss.item() * len(batch) #Sum_loss in each iteration,  len(feature)=batch size
                epoch_num += len(batch) # sum of all the batch size from every iteration, Total number of patients
                tr_loss += loss.item()

                if self.args.fp16:
                    if self._use_amp_for_fp16_from == 1:
                        self.amp_scaler.scale(loss).backward()
                    elif self._use_amp_for_fp16_from == 2:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss.backward()

                # update gradient
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == batch_total_step:
                    if self.args.fp16:
                        if self._use_amp_for_fp16_from == 1:
                            self.amp_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm
                            )
                            self.amp_scaler.step(self.optimizer)
                            self.amp_scaler.update()
                        elif self._use_amp_for_fp16_from == 2:
                            torch.nn.utils.clip_grad_norm_(
                                self.amp.master_params(self.optimizer),
                                self.args.max_grad_norm,
                            )
                            self.optimizer.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )
                        self.optimizer.step()
                    if self.args.do_warmup:
                        self.scheduler.step()
                    # batch_iter.set_postfix({"loss": loss.item(), "tloss": tr_loss/step})
                if self.args.log_step > 0 and (step + 1) % self.args.log_step == 0:
                    self.args.logger.info(
                        "epoch: {}; global step: {}; total loss: {}; average loss: {}".format(
                            epoch + 1, t_step, tr_loss, tr_loss / t_step
                        )
                    )

                t_step += 1
            batch_iter.close()

            # at each epoch end, we do eval on dev
            if self.args.do_eval:
                acc, p, r, f1, eval_loss = self.eval(self.args.non_relation_label)
                self.args.logger.info(
                    """
                ******************************
                Epoch: {}
                evaluation on dev set
                vaild_loss:{} 
                acc: {}
                precision:{}; recall:{}
                f1:{}
                ******************************
                """.format(
                        epoch + 1, eval_loss, acc, p, r, f1
                    )
                )
                acc_history.append(acc)
                #auc_history.append(total_auc)
                f1_history.append(f1)
                recall_history.append(r)
                precison_history.append(p)
                eval_epoch_loss_history.append(eval_loss)
                #specificity_history.append(total_specificity)

                # max_num_checkpoints > 0, save based on eval
                # save model
                if self.args.max_num_checkpoints > 0 and latest_best_score < f1:
                    self._save_model(epoch + 1)
                    latest_best_score = f1

            epoch_loss = epoch_loss / epoch_num  # Got epoch average loss
            epoch_loss_history.append(epoch_loss)
        epoch_iter.close()

        #Get epoch_loss_metrics
        plt.plot(epoch_loss_history, label = 'training_loss')
        plt.plot(acc_history, label = 'validation_accuracy')
        plt.plot(f1_history, label = 'validation_f1')
        plt.plot(recall_history, label = 'validation_recall')
        plt.plot(precison_history, label = 'validation_precison')
        plt.title('Average loss of each epoch')
        plt.legend()
        plt.savefig(os.path.join(self.new_model_dir_path, 'epoch_loss_metrics.png'))
        plt.close()

        #Get epoch_loss
        plt.plot(epoch_loss_history, label = 'training_loss')
        plt.plot(eval_epoch_loss_history, label = 'validation_loss')
        plt.title('Average loss of each epoch')
        plt.legend()
        plt.savefig(os.path.join(self.new_model_dir_path, 'epoch_loss.png'))
        plt.close()

        #Get iteration_loss
        plt.plot(iteration_loss_history)
        plt.title('Average loss of each iteration')
        plt.savefig(os.path.join(self.new_model_dir_path, 'iteration_loss.png'))
        plt.close()

        # max_num_checkpoints=0 then save at the end of training
        if self.args.max_num_checkpoints <= 0:
            self._save_model(0)
            self.args.logger.info("training finish and the trained model is saved.")

    def eval(self, non_rel_label=""):
        self.args.logger.info("start evaluation...")

        # this is done on dev
        true_labels = np.array([dev_fea.label for dev_fea in self.dev_features])
        preds, eval_loss, _, _ = self._run_eval(self.dev_data_loader)
        acc, p, r, f1 = acc_and_f1(
            labels=true_labels,
            preds=preds,
            label2idx=self.label2idx,
            non_rel_label=non_rel_label,
        )

        return acc, p, r, f1, eval_loss

    def predict_highlight(self, index):
        self.args.logger.info("start prediction with highlight...")
        # this is for prediction
        text_a = self._check_cache(task="test")[index].text_a  #0-index
        #print('text_a in predict_highlight:',text_a)
        class_names = ['0', '1']
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text_a, self.predict_highlight_helper, num_features=15)
        return exp

    def predict_highlight_helper(self, inputs):
        outputs = []
        #print('inputs in predict_highlight_helper:', inputs)
        for input in inputs:
            #print('input under forloop:',input)
            examples = self._load_examples_by_task(task="test_highlight",input=input)
            test_features = convert_examples_to_relation_extraction_features(
                    examples,
                    tokenizer=self.tokenizer,
                    max_length=self.args.max_seq_length,
                    label_list=self.label2idx,
                    output_mode="classification",
                )
            test_data_loader_highlight = relation_extraction_data_loader(
                    test_features,
                    batch_size=1,
                    task="test",
                    logger=self.args.logger,
                    binary_mode=self.args.use_binary_classification_mode,
                )
            _, _, _, pred_probs = self._run_eval(test_data_loader_highlight)
            
            outputs.append(pred_probs[0])
        outputs = np.array(outputs)
        return outputs

    def predict(self):
        self.args.logger.info("start prediction...")
        # this is for prediction
        preds, _, pred_probs, _ = self._run_eval(self.test_data_loader)
        # convert predicted label idx to real label
        self.args.logger.info(
            "label to index for prediction:\n{}".format(self.label2idx)
        )
        preds = [self.idx2label[pred] for pred in preds]

        return preds, pred_probs

    def _init_new_model(self):
        """initialize a new model for fine-tuning"""
        self.args.logger.info("Init new model...")

        model, config, tokenizer = self.model_dict[self.args.model_type]

        # init tokenizer and add special tags
        self.tokenizer = tokenizer.from_pretrained(
            self.args.pretrained_model, do_lower_case=self.args.do_lower_case
        )
        last_token_idx = len(self.tokenizer)
        self.tokenizer.add_tokens(SPEC_TAGS)
        spec_token_new_ids = tuple(
            [
                (last_token_idx + idx)
                for idx in range(len(self.tokenizer) - last_token_idx)
            ]
        )
        total_token_num = len(self.tokenizer)

        # init config
        unique_labels, label2idx, idx2label = self.data_processor.get_labels()
        self.args.logger.info("label to index:\n{}".format(label2idx))
        save_json(label2idx, self.new_model_dir_path / "label2idx.json")
        num_labels = len(unique_labels)
        self.label2idx = label2idx
        self.idx2label = idx2label

        self.config = config.from_pretrained(
            self.args.pretrained_model, num_labels=num_labels
        )
        self.config.update({CONFIG_VERSION_NAME: VERSION})
        # The number of tokens to cache.
        # The key/value pairs that have already been pre-computed in a previous forward pass won’t be re-computed.
        if self.args.model_type == "xlnet":
            self.config.mem_len = self.config.d_model
            # change dropout name
            self.config.hidden_dropout_prob = self.config.dropout
        self.config.tags = spec_token_new_ids
        self.config.scheme = self.args.classification_scheme
        # binary mode
        self.config.binary_mode = self.args.use_binary_classification_mode
        # focal loss config
        self.config.use_focal_loss = self.args.use_focal_loss
        self.config.focal_loss_gamma = self.args.focal_loss_gamma
        # sample weights in loss functions
        self.config.balance_sample_weights = self.args.balance_sample_weights
        if self.args.balance_sample_weights:
            label2freq = self.data_processor.get_sample_distribution()
            label_id2freq = {label2idx[k]: v for k, v in label2freq.items()}
            self.config.sample_weights = np.zeros(len(label2freq))
            for k, v in label_id2freq.items():
                self.config.sample_weights[k] = v
            self.args.logger.info(
                f"using sample weights: {label_id2freq} and converted weight matrix is {self.config.sample_weights}"
            )
        # init model
        self.model = model.from_pretrained(
            self.args.pretrained_model, config=self.config
        )
        self.config.vocab_size = total_token_num
        self.model.resize_token_embeddings(total_token_num)

        # load model to device
        self.model.to(self.args.device)

    def _init_optimizer(self):
        # set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        self.args.logger.info("The optimizer detail:\n {}".format(self.optimizer))

        # set up optimizer warm up scheduler (you can set warmup_ratio=0 to deactivated this function)
        if self.args.do_warmup:
            t_total = (
                len(self.train_data_loader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            warmup_steps = np.dtype("int64").type(self.args.warmup_ratio * t_total)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
            )

        # mix precision training
        if self.args.fp16 and self._use_amp_for_fp16_from == 2:
            self.model, self.optimizer = self.amp.initialize(
                self.model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

    def _init_trained_model(self):
        """initialize a fine-tuned model for prediction"""
        dir_list = [d for d in self.new_model_dir_path.iterdir() if d.is_dir()]
        latest_ckpt_dir = sorted(dir_list, key=lambda x: int(x.stem.split("_")[-1]))[-1]

        self.args.logger.info(
            "Init model from {} for prediction".format(latest_ckpt_dir)
        )

        model, config, tokenizer = self.model_dict[self.args.model_type]

        self.config = config.from_pretrained(latest_ckpt_dir)
        # compatibility check for config arguments
        if not (self.config.to_dict().get(CONFIG_VERSION_NAME, None) == VERSION):
            self.config.update(NEW_ARGS)

        self.tokenizer = tokenizer.from_pretrained(
            latest_ckpt_dir, do_lower_case=self.args.do_lower_case
        )
        self.model = model.from_pretrained(latest_ckpt_dir, config=self.config)

        # load label2idx
        self.label2idx, self.idx2label = pkl_load(latest_ckpt_dir / "label_index.pkl")
        # load model to device
        self.model.to(self.args.device)

    def _load_amp_for_fp16(self):
        # first try to load PyTorch naive amp; if fail, try apex; if fail again, throw a RuntimeError
        if version.parse(torch.__version__) >= version.parse("1.6.0"):
            self.amp = torch.cuda.amp
            self._use_amp_for_fp16_from = 1
            self.amp_scaler = torch.cuda.amp.GradScaler()
        else:
            try:
                from apex import amp

                self.amp = amp
                self._use_amp_for_fp16_from = 2
            except ImportError:
                self.args.logger.error(
                    "apex (https://www.github.com/nvidia/apex) for fp16 training is not installed."
                )
            finally:
                self.args.fp16 = False

    def _save_model(self, epoch=0):
        dir_to_save = self.new_model_dir_path / f"ckpt_{epoch}"

        self.tokenizer.save_pretrained(dir_to_save)
        self.config.save_pretrained(dir_to_save)
        self.model.save_pretrained(dir_to_save)
        # save label2idx
        pkl_save((self.label2idx, self.idx2label), dir_to_save / "label_index.pkl")
        # remove extra checkpoints
        dir_list = [d for d in self.new_model_dir_path.iterdir() if d.is_dir()]
        if len(dir_list) > self.args.max_num_checkpoints > 0:
            oldest_ckpt_dir = sorted(
                dir_list, key=lambda x: int(x.stem.split("_")[-1])
            )[0]
            shutil.rmtree(oldest_ckpt_dir)

    def _run_eval(self, data_loader):
        temp_loss = 0.0
        # set model to evaluate mode
        self.model.eval()

        # create dev data batch iteration
        batch_iter = tqdm(data_loader, desc="Batch", disable=not self.args.progress_bar)
        total_sample_num = len(batch_iter)
        preds = None
        preds_prob = None

        for batch in batch_iter:
            batch_input = batch_to_model_input(
                batch, model_type=self.args.model_type, device=self.args.device
            )
            with torch.no_grad():
                batch_output = self.model(**batch_input)
                loss, logits = batch_output[:2]
                #print('logits:', logits)
                temp_loss += loss.item()
                logits_prob = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                # print('logits_prob:',logits_prob)
                if preds is None:
                    preds = logits
                else:
                    preds = np.append(preds, logits, axis=0)

                if preds_prob is None:
                    preds_prob = logits_prob
                else:
                    preds_prob = np.append(preds_prob, logits_prob, axis=0)

        batch_iter.close()
        temp_loss = temp_loss / total_sample_num
        #print('preds:',preds)
        #use softmax to get the prob for the positive class (class 1)
        #pred_prob = np.max(preds, axis=-1)
        pred_prob = np.max(preds_prob, axis=-1)
        #print('preds_prob after softmax:',preds_prob)
        #pred_prob = preds_prob[:, 1]
        print('pred_prob:',pred_prob)
        preds = np.argmax(preds, axis=-1)
        print('preds after argmax:',preds)
        return preds, temp_loss, pred_prob, preds_prob

    def _load_examples_by_task(self, task="train", input=None):
        examples = None

        if task == "train":
            examples = self.data_processor.get_train_examples()
        elif task == "dev":
            examples = self.data_processor.get_dev_examples()
        elif task == "test":
            examples = self.data_processor.get_test_examples()
        elif task == "test_highlight":
            examples = self.data_processor.get_test_examples_highlight(input)
        else:
            raise RuntimeError(
                "expect task to be train, dev or test but get {}".format(task)
            )

        return examples

    def _check_cache(self, task="train"):
        cached_examples_file = Path(
            self.args.data_dir
        ) / "cached_{}_{}_{}_{}_{}.pkl".format(
            self.args.model_type,
            self.args.data_format_mode,
            self.args.max_seq_length,
            self.tokenizer.name_or_path.split("/")[-1],
            task,
        )
        # load examples from files or cache
        if self.args.cache_data and cached_examples_file.exists():
            examples = pkl_load(cached_examples_file)
            self.args.logger.info(
                "load {} data from cached file: {}".format(task, cached_examples_file)
            )
        elif self.args.cache_data and not cached_examples_file.exists():
            self.args.logger.info(
                "create {} examples...and will cache the processed data at {}".format(
                    task, cached_examples_file
                )
            )
            examples = self._load_examples_by_task(task)
            pkl_save(examples, cached_examples_file)
        else:
            self.args.logger.info(
                "create training examples..." "the processed data will not be cached"
            )
            examples = self._load_examples_by_task(task)
        return examples

    def reset_dataloader(self, data_dir, has_file_header=None, max_len=None):
        """
        allow reset data dir and data file header and max seq len
        """
        self.data_processor.set_data_dir(data_dir)
        if has_file_header:
            self.data_processor.set_header(has_file_header)
        if max_len and isinstance(max_len, int) and 0 < max_len <= 512:
            self.data_processor.set_max_seq_len(max_len)
        self.args.logger.warning("reset data loader information")
        self.args.logger.warning("new data loader info: {}".format(self.data_processor))
        self._init_dataloader()

    def _init_dataloader(self):
        if self.args.do_train and self.train_data_loader is None:
            train_examples = self._check_cache(task="train")
            # convert examples to tensor
            train_features = convert_examples_to_relation_extraction_features(
                train_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification",
            )

            self.train_data_loader = relation_extraction_data_loader(
                train_features,
                batch_size=self.args.train_batch_size,
                task="train",
                logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode,
            )

        if self.args.do_eval and self.dev_data_loader is None:
            dev_examples = self._check_cache(task="dev")
            # example2feature
            dev_features = convert_examples_to_relation_extraction_features(
                dev_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification",
            )
            self.dev_features = dev_features

            self.dev_data_loader = relation_extraction_data_loader(
                dev_features,
                batch_size=self.args.train_batch_size,
                task="test",
                logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode,
            )

        if self.args.do_predict and self.test_data_loader is None:
            test_examples = self._check_cache(task="test")
            # example2feature
            test_features = convert_examples_to_relation_extraction_features(
                test_examples,
                tokenizer=self.tokenizer,
                max_length=self.args.max_seq_length,
                label_list=self.label2idx,
                output_mode="classification",
            )

            self.test_data_loader = relation_extraction_data_loader(
                test_features,
                batch_size=self.args.eval_batch_size,
                task="test",
                logger=self.args.logger,
                binary_mode=self.args.use_binary_classification_mode,
            )
