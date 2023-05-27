from transformers import AutoTokenizer,get_constant_schedule, get_constant_schedule_with_warmup, AutoModel, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoConfig
from tqdm import trange
from nltk.stem import PorterStemmer
from copy import deepcopy
from collections import Counter
from zsl.data_args import DataArgs
from zsl.training_args import TrainingArgs
from zsl.eda import eda
from zsl.utils import *
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
import shutil
import transformers
from torch.utils.data import DataLoader
import pandas as pd


class WeaklyZSLTrainer():
    def __init__(self, data_args: DataArgs, training_args: TrainingArgs):
        transformers.logging.set_verbosity_info()
        self.logger = transformers.logging.get_logger()
        self.data_args = data_args
        self.training_args = training_args
        self.raw_classes = read_class_names(os.path.join(self.data_args.data_path, "class_names.txt"))
        add_filehandler_for_logger(self.data_args.data_path, self.logger)
        self.logger.info("General Data Args: " + json.dumps(self.data_args.__dict__, indent=2))
        self.logger.info("General Training Args: " + json.dumps(self.training_args.__dict__, indent=2))
        set_seed(self.training_args.seed)

    def aug_with_eda(self, sequence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=5):
        aug_sentences = eda(sequence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        return aug_sentences

    def aug_batch_with_eda(self, examples, class_dist, aug_target=100):
        self.logger.info(f"start EDA to ensure each class has at least {aug_target} examples")
        aug_labels2num_aug = {}
        for key, value in class_dist.items():
            if value <= aug_target:
                aug_labels2num_aug[key] = math.ceil((aug_target - value) / value)
        aug_examples = []
        label_key = "predicted_label" if "predicted_label" in examples[0] else "label"
        for example in examples:
            if example[label_key] in aug_labels2num_aug:  # this makes it stricter for multi-label dataset augmentation
                # num_aug_list = [aug_labels2num_aug[j] for j in example[label_key].split(",") if j in aug_labels2num_aug]
                # num_aug = int(sum(num_aug_list)/len(num_aug_list))
                text_ = deepcopy(example["text"])
                aug_sentences = self.aug_with_eda(text_, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, alpha_rd=0.2, num_aug=aug_labels2num_aug[example[label_key]])
                for each in aug_sentences:
                    # example["text_"] = text_
                    new_example = deepcopy(example)
                    new_example.update({"text": each})
                    aug_examples.append(new_example)
        examples.extend(aug_examples)
        preds = []
        for i in examples:
            preds.extend(i[label_key].split(","))
        return examples, Counter(preds)

    def encode_data(self, tokenizer, examples, with_label=False, with_td=False):
        encoded_examples = {}
        for i in trange(0, len(examples), self.data_args.bs):
            batch_examples = examples[i:i + self.data_args.bs]
            input_texts = [be["text"] for be in batch_examples]
            inputs = tokenizer.batch_encode_plus(input_texts, truncation=True, max_length=self.data_args.max_seq_length, return_tensors='pt', padding="max_length")
            inputs.update({"input_texts": input_texts})
            if with_label:
                # for testing purpose
                if self.data_args.multi_label:
                    inputs.update({"labels": [be[self.data_args.label_field_name].split(",") for be in batch_examples]})
                    one_hots = []
                    for be in batch_examples:
                        labels_this = be[self.data_args.label_field_name].split(",")
                        one_hot = [1 if i in labels_this else 0 for i in self.raw_classes]
                        one_hots.append(one_hot)
                    inputs.update({"label_indices": torch.tensor(one_hots).float()})

                else:
                    inputs.update({"labels": [be[self.data_args.label_field_name] for be in batch_examples]})
                    inputs.update({"label_indices": [self.raw_classes.index(be[self.data_args.label_field_name]) for be in batch_examples]})

            if with_td:
                if self.data_args.multi_label:
                    inputs.update({"target_dist": torch.tensor([be["target_dist"] for be in batch_examples]).float()})
                else:
                    inputs.update({"target_dist": torch.tensor([be["target_dist"] for be in batch_examples])})

            for k, v in inputs.items():
                if k not in encoded_examples:
                    encoded_examples[k] = v
                else:
                    if torch.is_tensor(v):
                        encoded_examples[k] = torch.cat([encoded_examples[k], v], dim=0)
                    else:
                        encoded_examples[k].extend(v)

        if self.data_args.multi_label and "labels" in encoded_examples:
            # for multi_label each element in list of batch should be of equal size before batching, so remove labels but keep label_indices
            del encoded_examples["labels"]

        return encoded_examples

    def pretrain_self_train(self, only_pretrain=False, pretrain_set_name="train", eval_set="val"):
        if not pretrain_set_name.endswith("_td"):
            assert hasattr(self, "label2vocab"), "label2vocab has not generated yet, consider to run self.setup_labelvocab first"

        model_short_name = self.data_args.model_for_preselftrain.split('/')[-1]
        pretrain_model_path = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "pretrain", "final_model")
        return_dict = {}
        if os.path.exists(pretrain_model_path) and not self.training_args.override:
            # if it (pretrain_model_path) exists, then load it directly
            model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_path)
            tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)

        else:
            pretrain_model_folder = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "pretrain")
            if os.path.isdir(pretrain_model_folder) and self.training_args.override:
                shutil.rmtree(pretrain_model_folder)
            model, tokenizer, pretrain_results = self.pre_train( set_name=pretrain_set_name, eval_set=eval_set)
            return_dict.update({"pre_train": pretrain_results})

        # start self training
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        set_name = "train"
        train_data_load_path = os.path.join(self.data_args.data_path, f"{model_short_name}-st-{set_name}-data.pt")
        if os.path.isfile(train_data_load_path) and not self.training_args.override:
            encoded_train_examples = torch.load(train_data_load_path)
        else:
            data_path = os.path.join(self.data_args.data_path, f"{set_name}.json")
            examples = read_jsonl(data_path)
            encoded_train_examples = self.encode_data(tokenizer, examples)
            torch.save(encoded_train_examples, train_data_load_path)

        data_path = os.path.join(self.data_args.data_path, f"{eval_set}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=True)
        eval_dataset = MyDataset(encoded_eval_examples)
        if return_dict == {}:
            pretrain_results = self.inference(model, eval_dataset)
            return_dict.update({"pre_train": pretrain_results})
        if only_pretrain:
            return return_dict

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.training_args.self_train_training_lr, eps=1e-8)
        total_steps = int(len(encoded_train_examples["input_ids"]) * self.training_args.self_train_epochs / (1 * self.training_args.selftrain_batch_size * self.training_args.selftrain_accumulation_steps))

        if self.training_args.selftrain_lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.training_args.warmup_ratio * total_steps, num_training_steps=total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        agree_count = 0
        idx = 0
        selftrain_results = {}
        selftrain_model_folder = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "selftrain_interval")
        if os.path.isdir(selftrain_model_folder) and self.training_args.override:
            shutil.rmtree(selftrain_model_folder)

        interval_count = math.ceil(total_steps / self.training_args.self_train_update_interval)
        # assert interval_count == 0, "too few training examples, consider decrease self_train_update_interval"
        for i in trange(interval_count):
            target_num = min(1 * self.training_args.selftrain_batch_size * self.training_args.self_train_update_interval * self.training_args.accumulation_steps,
                             len(encoded_train_examples["input_ids"]))
            if idx + target_num >= len(encoded_train_examples["input_ids"]):
                select_idx = torch.cat((torch.arange(idx, len(encoded_train_examples["input_ids"])),
                                        torch.arange(idx + target_num - len(encoded_train_examples["input_ids"]))))
            else:
                select_idx = torch.arange(idx, idx + target_num)
            assert len(select_idx) == target_num
            selected_train_examples = {}
            for k, v in encoded_train_examples.items():
                if torch.is_tensor(v):
                    selected_train_examples[k] = v[select_idx]
                else:
                    if idx + target_num >= len(encoded_train_examples["input_ids"]):
                        selected_train_examples[k] = v[idx:len(encoded_train_examples["input_ids"])] + v[0:idx + target_num - len(encoded_train_examples["input_ids"])]
                    else:
                        selected_train_examples[k] = v[select_idx[0]:select_idx[-1] + 1]

            idx = (idx + len(select_idx)) % len(encoded_train_examples["input_ids"])
            predicted_encoded_examples, _, agree = self.select_via_inference(model, selected_train_examples, confidence_threshold=0.00)
            self.logger.info(f"agree: {agree}")
            if self.training_args.self_train_early_stop:
                if 1 - agree < 1e-3:
                    agree_count += 1
                else:
                    agree_count = 0
                if agree_count >= 3:
                    break
            train_dataset = MyDataset(predicted_encoded_examples)
            train_loader = DataLoader(train_dataset, batch_size=self.training_args.selftrain_batch_size, num_workers=1, shuffle=True)
            model.train().to(device)
            total_interval_loss = 0
            interval_global_step = 0
            wrap_train_dataset_loader = tqdm(train_loader)
            model.zero_grad()
            for j, batch in enumerate(wrap_train_dataset_loader):
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["attention_mask"].to(device)
                target_dist = batch["target_dist"].to(device)
                outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                logits = outputs.logits
                preds = torch.nn.LogSoftmax(dim=-1)(logits)
                loss = loss_fn(preds, target_dist)
                total_interval_loss += loss.item()
                loss.backward()
                if (j + 1) % self.training_args.accumulation_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                interval_global_step += 1
                wrap_train_dataset_loader.update(1)
                wrap_train_dataset_loader.set_description(
                    f"Continual Self Training - interval {i + 1}/{interval_count} iter {j}/{len(wrap_train_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.training_args.self_train_eval_steps > 0 and interval_global_step % self.training_args.self_train_eval_steps == 0:
                    self.logger.info(f"evaluation during training on {eval_set} set: ")
                    selftrain_results = self.inference(model, eval_dataset)
                    model.train()
            self.logger.info(f"Average training loss at interval {i + 1}: {total_interval_loss / len(wrap_train_dataset_loader)}")
            # evaluate at the end of interval if eval_steps is smaller than or equal to 0
            if self.training_args.self_train_eval_steps <= 0:
                self.logger.info(f"evaluation during training on {eval_set} set: ")
                selftrain_results = self.inference(model, eval_dataset)
            # save up at end of interval!, this will cost much memory if there are many intervals
            # save_path = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "selftrain_interval", f"interval_{i + 1}")  # i+1 refers to the interval id
            # model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)
        # save up after the self training terminates
        save_path = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "selftrain_interval", "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        return_dict.update({f"self_train": selftrain_results})
        self.logger.info(json.dumps(return_dict, indent=2))
        return return_dict

    def pre_train(self, eval_set="val", set_name="train"):
        if not set_name.endswith("_td"):
            assert hasattr(self, "label2vocab"), "label2vocab has not generated yet, consider to run self.setup_labelvocab first"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_short_name = self.data_args.model_for_preselftrain.split('/')[-1]
        load_path = os.path.join(self.data_args.data_path, f"{model_short_name}-pt-{set_name}-data.pt")
        tokenizer = AutoTokenizer.from_pretrained(self.data_args.model_for_preselftrain)
        if os.path.isfile(load_path) and not self.training_args.override:
            predicted_encoded_examples = torch.load(load_path)
        else:
            data_path = os.path.join(self.data_args.data_path, f"{set_name}.json")
            examples = read_jsonl(data_path)

            if set_name.endswith("_td"):
                predicted_encoded_examples = self.encode_data(tokenizer, examples, with_td=True)
            else:
                encoded_train_examples = self.encode_data(tokenizer, examples)
                predicted_encoded_examples, _ = self.make_bow_predictions(encoded_train_examples)
            torch.save(predicted_encoded_examples, load_path)

        id2label = {id: label for id, label in enumerate(self.raw_classes)}
        label2id = {label: id for id, label in id2label.items()}
        config = AutoConfig.from_pretrained(self.data_args.model_for_preselftrain)
        config.id2label = id2label
        config.label2id = label2id
        model = AutoModelForSequenceClassification.from_pretrained(self.data_args.model_for_preselftrain, **config.__dict__)

        train_dataset = MyDataset(predicted_encoded_examples)
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.pretrain_batch_size, num_workers=1, shuffle=True)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.training_args.pre_train_training_lr, eps=1e-8)
        total_steps = len(train_loader) * self.training_args.pre_train_epochs / self.training_args.accumulation_steps
        if self.training_args.pretrain_lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.training_args.warmup_ratio * total_steps, num_training_steps=total_steps)
        elif self.training_args.pretrain_lr_scheduler == "linearconstant":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)

        if self.training_args.pre_train_one_hot:
            if self.data_args.multi_label:
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
            else:
                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

        data_path = os.path.join(self.data_args.data_path, f"{eval_set}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=True)
        eval_dataset = MyDataset(encoded_eval_examples)
        return_dict = {}
        model.train().to(device)
        global_step = 0
        for i in range(self.training_args.pre_train_epochs):
            self.logger.info(f"Epoch {i + 1}:")
            wrap_dataset_loader = tqdm(train_loader)
            model.zero_grad()
            total_epoch_loss = 0
            for j, batch in enumerate(wrap_dataset_loader):
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["attention_mask"].to(device)
                target_dist = batch["target_dist"].to(device)
                outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                logits = outputs.logits
                # logits = logits[:, 0, :]
                if self.training_args.pre_train_one_hot:
                    if self.data_args.multi_label:
                        loss = loss_fn(logits, target_dist.float())
                    else:
                        label_indices = target_dist.max(-1).indices
                        loss = loss_fn(logits, label_indices)
                else:
                    preds = torch.nn.LogSoftmax(dim=-1)(logits)
                    loss = loss_fn(preds, target_dist)

                total_epoch_loss += loss.item()
                loss.backward()
                if (j + 1) % self.training_args.accumulation_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                global_step += 1
                wrap_dataset_loader.update(1)
                wrap_dataset_loader.set_description(
                    f"Initial Pre-Training on {set_name} - epoch {i + 1}/{self.training_args.pre_train_epochs} iter {j}/{len(wrap_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.training_args.pre_train_eval_steps > 0 and global_step % self.training_args.pre_train_eval_steps == 0:
                    self.logger.info(f"evaluation during training for {eval_set} set: ")
                    return_dict = self.inference(model, eval_dataset)
                    model.train()
            self.logger.info(f"Average training loss for epoch {i + 1}: {total_epoch_loss / len(train_loader)}")
            # evaluate at the end of epoch if eval_steps is smaller than or equal to 0
            if self.training_args.pre_train_eval_steps <= 0:
                self.logger.info(f"evaluation during training for {eval_set} set: ")
                return_dict = self.inference(model, eval_dataset)
                model.train()
            # save up at end of each epoch!
            # model.save_pretrained(os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "pretrain", f"epoch_{i + 1}"))
            # tokenizer.save_pretrained(os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "pretrain", f"epoch_{i + 1}"))

        # save up at end of training!
        pretrain_model_path = os.path.join(self.training_args.output_path, "pre_self_train", model_short_name, "pretrain", "final_model")
        model.save_pretrained(pretrain_model_path)
        tokenizer.save_pretrained(pretrain_model_path)
        self.logger.info(json.dumps(return_dict, indent=2))
        return model, tokenizer, return_dict

    def full_train(self, set_name=None, eval_set="val", with_eda=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        set_name = "train" if set_name is None else set_name
        model_short_name = self.data_args.model_for_fulltrain.split('/')[-1]
        load_path = os.path.join(self.data_args.data_path, f"{model_short_name}-ft-{set_name}-data.pt")
        tokenizer = AutoTokenizer.from_pretrained(self.data_args.model_for_fulltrain)

        if os.path.isfile(load_path) and not self.training_args.override:
            encoded_train_examples = torch.load(load_path)
        else:
            data_path = os.path.join(self.data_args.data_path, f"{set_name}.json")
            examples = read_jsonl(data_path)
            if with_eda:
                class_dist = dict(Counter([i["label"] for i in examples]))
                examples, class_dist = self.aug_batch_with_eda(examples, class_dist, aug_target=500)

            encoded_train_examples = self.encode_data(tokenizer, examples, with_label=True)
            torch.save(encoded_train_examples, load_path)

        id2label = {id_: label for id_, label in enumerate(self.raw_classes)}
        label2id = {label: id_ for id_, label in id2label.items()}
        config = AutoConfig.from_pretrained(self.data_args.model_for_fulltrain)
        config.id2label = id2label
        config.label2id = label2id
        model = AutoModelForSequenceClassification.from_pretrained(self.data_args.model_for_fulltrain, **config.__dict__)

        train_dataset = MyDataset(encoded_train_examples)
        train_loader = DataLoader(train_dataset, batch_size=self.training_args.fulltrain_batch_size, num_workers=1, shuffle=True)
        total_steps = len(train_loader) * self.training_args.full_train_epochs / self.training_args.accumulation_steps

        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.training_args.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.training_args.full_train_training_lr, eps=1e-8)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.training_args.pre_train_training_lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.training_args.warmup_ratio * total_steps, num_training_steps=total_steps)
        if self.data_args.multi_label:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        data_path = os.path.join(self.data_args.data_path, f"{eval_set}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=True)

        eval_dataset = MyDataset(encoded_eval_examples)
        model.train().to(device)
        global_step = 0
        eval_loss = 0
        for i in range(self.training_args.full_train_epochs):
            self.logger.info(f"Epoch {i + 1}:")
            wrap_dataset_loader = tqdm(train_loader)
            model.zero_grad()
            total_epoch_loss = 0
            for j, batch in enumerate(wrap_dataset_loader):
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["attention_mask"].to(device)
                label_indices = batch["label_indices"].to(device)
                outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                logits = outputs.logits
                loss = loss_fn(logits, label_indices)
                total_epoch_loss += loss.item()
                eval_loss += loss.item()
                loss.backward()
                if (j + 1) % self.training_args.accumulation_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                global_step += 1
                wrap_dataset_loader.update(1)
                wrap_dataset_loader.set_description(
                    f"Full-Training - epoch {i + 1}/{self.training_args.full_train_epochs} iter {j}/{len(wrap_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.training_args.full_train_eval_steps > 0 and global_step % self.training_args.full_train_eval_steps == 0:
                    # self.logger.info(f"evaluation during training on {eval_set} set: ")
                    self.logger.info(f"\naverage training loss at global_step={global_step}: {eval_loss / self.training_args.full_train_eval_steps}")
                    eval_loss = 0
                    self.inference(model, eval_dataset)
                    model.train()

            self.logger.info(f"Average training loss for epoch {i + 1}: {total_epoch_loss / len(train_loader)}")
            # evaluate at the end of epoch if eval_steps is smaller than or equal to 0
            if self.training_args.full_train_eval_steps <= 0:
                self.logger.info(f"evaluation during training on {eval_set} set: ")
                self.inference(model, eval_dataset)
                model.train()

            # save up at end of each epoch!
            # model.save_pretrained(os.path.join(self.training_args.output_path, "full_train", model_short_name, f"epoch_{i + 1}"))
            # tokenizer.save_pretrained(os.path.join(self.training_args.output_path, "full_train", model_short_name, f"epoch_{i + 1}"))
        # save up at end of training!
        fulltrain_model_path = os.path.join(self.training_args.output_path, "full_train", model_short_name, "final_model")
        model.save_pretrained(fulltrain_model_path)
        tokenizer.save_pretrained(fulltrain_model_path)

        self.logger.info(f"evaluation on test set with full-trained model: {fulltrain_model_path}")
        return_dict = {f"full_train(eval_set={eval_set})": self.predict(load_path=fulltrain_model_path, set_name=eval_set)}
        return return_dict

    def predict(self, load_path=None, set_name="test", pred2file=False):
        # load up
        # model_short_name = self.data_args.model_for_preselftrain.split('/')[-1]
        # model_path = os.path.join(self.training_args.output_path, "self_train", model_short_name, f"epoch_1")
        # load_path = model_path if load_path is None else load_path
        if load_path is None:
            model_short_name = self.data_args.model_for_fulltrain.split('/')[-1]
            load_path = os.path.join(self.training_args.output_path, "full_train", model_short_name, "final_model")
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        data_path = os.path.join(self.data_args.data_path, f"{set_name}.json")
        examples = read_jsonl(data_path)
        encoded_eval_examples = self.encode_data(tokenizer, examples, with_label=True)
        eval_dataset = MyDataset(encoded_eval_examples)
        if pred2file:
            scores_dict, preds = self.inference(model, eval_dataset, return_preds=True)
            predfile = os.path.join(self.training_args.output_path, f"{set_name}_pred.csv")
            df = pd.DataFrame(columns=["text", "label"])
            df["text"] = [example["text"] for example in examples]
            df["label"] = [example["label"] for example in examples]
            df["prediction"] = preds
            df.to_csv(predfile, index=False)
            self.logger.info(f"predictions are written to: {predfile}")
        else:
            scores_dict = self.inference(model, eval_dataset)
        return scores_dict

    def select_via_inference(self, model, unpredicted_encoded_examples, confidence_threshold=0.0):
        new_predicted_examples, new_unpredicted_examples = {}, {}
        unpredicted_dataset = MyDataset(unpredicted_encoded_examples)
        unpredicted_loader = DataLoader(unpredicted_dataset, batch_size=self.training_args.eval_batch_size, num_workers=1, shuffle=False)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # try:
        model.eval().to(device)
        preds = []
        with torch.no_grad():
            wrap_unpredicted_loader = tqdm(unpredicted_loader, desc="select via inference")
            for batch in wrap_unpredicted_loader:
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                logits = outputs.logits
                probs = torch.nn.Softmax(dim=-1)(logits)
                selected_indices, unselected_indices = [], []
                top2_values, _ = probs.topk(2, 1)
                for i, v in enumerate(top2_values):
                    if v[0] - v[1] > confidence_threshold:
                        selected_indices.append(i)
                    else:
                        unselected_indices.append(i)
                selected_probs_dist = probs[selected_indices, :].cpu()
                pred_indices = selected_probs_dist.argmax(-1).tolist()
                preds.extend([self.raw_classes[i] for i in pred_indices])
                if "target_dist" not in new_predicted_examples:
                    new_predicted_examples["target_dist"] = selected_probs_dist
                else:
                    new_predicted_examples["target_dist"] = torch.cat([new_predicted_examples["target_dist"], selected_probs_dist], dim=0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        v = v.cpu()  # free up gpu memory
                        selected_v = v[selected_indices, :]
                    else:
                        selected_v = [v[i] for i in selected_indices]
                    if k not in new_predicted_examples:
                        new_predicted_examples[k] = selected_v
                    else:
                        if torch.is_tensor(selected_v):
                            new_predicted_examples[k] = torch.cat([new_predicted_examples[k], selected_v], dim=0)
                        else:
                            new_predicted_examples[k].extend(selected_v)

                    if torch.is_tensor(v):
                        v = v.cpu()  # free up gpu memory
                        unselected_v = v[unselected_indices, :]
                    else:
                        unselected_v = [v[i] for i in unselected_indices]
                    if k not in new_unpredicted_examples:
                        new_unpredicted_examples[k] = unselected_v
                    else:
                        if torch.is_tensor(unselected_v):
                            new_unpredicted_examples[k] = torch.cat([new_unpredicted_examples[k], unselected_v], dim=0)
                        else:
                            new_unpredicted_examples[k].extend(unselected_v)

            # soft over label distribution
            all_preds = new_predicted_examples["target_dist"]
            new_predicted_examples["target_dist"] = soft_label(all_preds)
            # reference to implementation from: https://github.com/yumeng5/LOTClass
            agree = (all_preds.argmax(dim=-1) == new_predicted_examples["target_dist"].argmax(dim=-1)).int().sum().item() / len(all_preds.argmax(dim=-1))
            self.logger.info(f"the class distribution of the newly selected samples based on model predictions is (total = {sum(Counter(preds).values())}): {json.dumps(Counter(preds), indent=2)}")

        return new_predicted_examples, new_unpredicted_examples, agree

    def inference(self, model, dataset, return_preds=False):
        eval_loader = DataLoader(dataset, batch_size=self.training_args.eval_batch_size, num_workers=1, shuffle=False)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model.eval().to(device)
            gts, preds = [], []
            with torch.no_grad():
                wrap_loader = tqdm(eval_loader, desc="predicting")
                for j, batch in enumerate(wrap_loader):
                    input_ids = batch["input_ids"].to(device)
                    input_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
                    logits = outputs.logits
                    # IF MULTI-LABEL
                    if self.data_args.multi_label:
                        probs = logits.sigmoid()
                        preds.extend((probs > 0.5).int().tolist())
                        gts.extend(batch["label_indices"].int().tolist())
                    else:
                        probs = torch.nn.Softmax(dim=-1)(logits)
                        pred_indices = probs.argmax(-1).tolist()
                        preds.extend([self.raw_classes[i] for i in pred_indices])
                        if "labels" in batch:
                            gts.extend(batch["labels"])

                    # wrap_loader.update(1)
                    # wrap_loader.set_description("predicting")
                if len(gts) == len(preds):
                    self.logger.info(classification_report(gts, preds, digits=4))
                    self.logger.info(f"accuracy score: {accuracy_score(gts, preds)}")
                    if not self.data_args.multi_label and len(set(gts)) == 2:
                        preds = [self.raw_classes.index(i) for i in preds]
                        gts = [self.raw_classes.index(i) for i in gts]
                    return_dict = calculate_perf(preds, gts)
                    self.logger.info(json.dumps(return_dict, indent=2))
                    if return_preds:
                        return return_dict, preds
                    return return_dict
                return {}
        except RuntimeError as err:
            self.logger.info(f"GPU memory is not enough: {err}")

    def sent_embeddings(self, input_texts, tokenizer=None, model=None, pbar=True):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.data_args.model_for_labelvocab)
        if model is None:
            model = AutoModel.from_pretrained(self.data_args.model_for_labelvocab)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.eval().to(device)
        mean_outputs = []
        with torch.no_grad():
            if pbar:
                iter_bar = trange(0, len(input_texts), self.data_args.bs)
            else:
                iter_bar = range(0, len(input_texts), self.data_args.bs)
            for i in iter_bar:
                # run inputs through model and mean-pool over the sequence
                # dimension to get sequence-level representations
                inputs = tokenizer.batch_encode_plus(input_texts[i:i + self.data_args.bs], truncation=True, max_length=self.data_args.max_seq_length, return_tensors='pt', padding=True).to(device)
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                output = model(input_ids, attention_mask=attention_mask)
                last_hidden_state, _ = output.last_hidden_state, output.pooler_output
                mean_outputs.append(last_hidden_state.mean(dim=1))
        return torch.cat(mean_outputs, 0)

    def report_data_stats(self, examples):
        tmp_list = [len(example["text"].split(" ")) for example in examples]
        max_ex_len = max(tmp_list)
        avg_ex_len = np.average(tmp_list)
        self.logger.info('Example max length: {} (words)'.format(max_ex_len))
        self.logger.info('Example average length: {} (words)'.format(avg_ex_len))
        exceed_count = len([i for i in tmp_list if i > self.data_args.max_seq_length])
        self.logger.info(f'Examples with words beyond max_seq_length ({self.data_args.max_seq_length}): {exceed_count}/{len(examples)} (examples)')
        self.logger.info("##################################")

    def run_entail_baseline(self, eval_set=None, hypothesis_template="This text is about {}.", threshold=0.0, write_examples=False):
        # we need label field in test.json as the ground truth references since testing is undergoing here
        test_path = os.path.join(self.data_args.data_path, "test.json" if eval_set is None else eval_set + ".json")
        examples = read_jsonl(test_path)
        self.logger.info(f"\n### Dataset statistics of {test_path}: ###")
        self.report_data_stats(examples)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.data_args.entail_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.data_args.entail_model).to(device)
        label_idx = -1
        if "entailment" in model.config.label2id:
            label_idx = model.config.label2id["entailment"]
        elif "ENTAILMENT" in model.config.label2id:
            label_idx = model.config.label2id["ENTAILMENT"]
        else:
            label_idx = model.config.label2id["yes"]
        preds = []
        gts = []
        predicted_examples, unpredicted_examples = [], []
        model.eval()
        self.logger.info(f"number of model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        with torch.no_grad():
            for example in tqdm(examples):
                premise = example["text"]
                # preprocessing the text example (for tweets specifically)
                premise = pre.clean(premise)
                premises = []
                hypothesises = []
                # pose sequence as a NLI premise and label as a hypothesis
                for label in self.raw_classes:
                    hypothesis = hypothesis_template.format(label)
                    premises.append(premise)
                    hypothesises.append(hypothesis)

                labels_probs_map = {}
                # in case run out of memory, we have batch inference here
                for i in range(0, len(premises), self.data_args.bs):
                    batch_premises = premises[i:i + self.data_args.bs]
                    batch_hypothesises = hypothesises[i:i + self.data_args.bs]
                    # run through model pre-trained on MNLI
                    inputs = tokenizer(batch_premises, batch_hypothesises, truncation=True, padding=True, max_length=self.data_args.max_seq_length, return_tensors='pt').to(device)
                    logits = model(**inputs)[0]
                    logits = logits[:, [0, label_idx if label_idx > 1 else 1]]
                    idx_ = 1 if label_idx > 1 else label_idx
                    probs = logits.softmax(dim=1)
                    entail_probs = probs[:, idx_].tolist()
                    labels = self.raw_classes[i:i + self.data_args.bs]
                    for entail_prob, label in zip(entail_probs, labels):
                        labels_probs_map[label] = entail_prob
                if self.data_args.multi_label:
                    threshold = threshold if threshold >= 0.5 else 0.5

                    candidates = [k for k, v in labels_probs_map.items() if v > threshold]

                    if len(candidates) == 0 and threshold == 0.5:
                        candidates = [max(labels_probs_map, key=labels_probs_map.get)]

                    if len(candidates) != 0:
                        pred_one_hot = [1 if i in candidates else 0 for i in self.raw_classes]
                        preds.append(pred_one_hot)

                        if self.data_args.label_field_name in example:
                            gt_candidates = example[self.data_args.label_field_name].split(",")
                            gt_one_hot = [1 if i in gt_candidates else 0 for i in self.raw_classes]
                            gts.append(gt_one_hot)
                        example["raw_target_dist"] = [round(labels_probs_map[i], 4) for i in self.raw_classes]
                        example["target_dist"] = pred_one_hot
                        example["target"] = candidates
                        example["predicted_label"] = ",".join(candidates)
                        predicted_examples.append(example)
                    else:
                        unpredicted_examples.append(example)
                else:
                    if labels_probs_map[max(labels_probs_map, key=labels_probs_map.get)] > threshold:
                        target_label = max(labels_probs_map, key=labels_probs_map.get)
                        preds.append(target_label)
                        if self.data_args.label_field_name in example:
                            gts.append(example[self.data_args.label_field_name])
                        example["target_dist"] = [1 if i in [target_label] else 0 for i in self.raw_classes]
                        example["raw_target_dist"] = [round(labels_probs_map[i], 4) for i in self.raw_classes]
                        example["target"] = [target_label]
                        example["predicted_label"] = target_label
                        predicted_examples.append(example)
                    else:
                        unpredicted_examples.append(example)
        if gts != []:
            self.logger.info(f"run entailment baseline results (from pre-trained model: {self.data_args.entail_model}): ")
            self.logger.info(f"hypothesis template: {hypothesis_template}")
            self.logger.info(f"eval on data path: {test_path}")
            self.logger.info(f"threshold: {threshold}")
            if self.data_args.multi_label:
                self.logger.info(classification_report(gts, preds, digits=4, target_names=self.raw_classes))
                self.logger.info(f"accuracy score: {accuracy_score(gts, preds)}")
            else:
                self.logger.info(classification_report(gts, preds, digits=4))

        if write_examples:
            def write_examples(to_examples, file_tag="et_td"):
                # target_labels = []
                target_file = os.path.join(self.data_args.data_path, f"test_{file_tag}.json" if eval_set is None else eval_set + f"_{file_tag}.json")
                with open(target_file, "w+") as f:
                    for ex in to_examples:
                        # target_labels.extend(ex["target"])
                        f.write(json.dumps(ex) + "\n")
                self.logger.info(f"write {len(to_examples)} examples to: {target_file}")
                # self.logger.info(f"target class dist: {json.dumps(dict(Counter(target_labels)), indent=2)}")

            write_examples(predicted_examples)
            write_examples(unpredicted_examples, file_tag="et_up")

        return_dict = calculate_perf(preds, gts)
        self.logger.info(json.dumps(return_dict, indent=2))
        return return_dict

    def make_bow_predictions(self, examples, use_weighting=True, p=0.7, expand_with_top=100):
        # start_gain_threshold: a parameter used to control the threshold for BOW predictions, suggested to set to be 0.1/len(classes) for single-label dataset and 0.5 for multi-label dataset
        label2vocab_tokens_count = {}
        label2vocab_tokens_count2 = {}
        tokenlabelfreq = {}
        doc_freq = {}
        porter = PorterStemmer()
        for label, vocab in self.label2vocab.items():
            # similar to the top_k_top_p way to select vocabulary
            min_vocab_num = 2
            label2vocab_tokens_count[label] = {}
            for k, v in vocab:
                if len(label2vocab_tokens_count[label]) < expand_with_top and v >= p:
                    label2vocab_tokens_count[label][k] = v

            if len(label2vocab_tokens_count[label]) < min_vocab_num:
                label2vocab_tokens_count[label] = {k: v for k, v in vocab[:min_vocab_num]}

            if use_weighting:
                for k in label2vocab_tokens_count[label].keys():
                    if k not in doc_freq:
                        doc_freq[k] = 1
                    else:
                        doc_freq[k] += 1

            # this number is experimentally tuned and top 100 gives overall the best performance
            # treating all tokens in the selected label vocab as unigrams works empirically better than treating separate grams
            vocab_tokens = [porter.stem(each) for each in " ".join([v[0] for v in vocab][:expand_with_top]).split() if each not in stopwords]
            label2vocab_tokens_count2[label] = dict(Counter(vocab_tokens))
            for k in label2vocab_tokens_count2[label]:
                if k not in tokenlabelfreq:
                    tokenlabelfreq[k] = 1
                else:
                    tokenlabelfreq[k] += 1

        if use_weighting:
            for label, vocab_tokens_count in label2vocab_tokens_count.items():
                label2vocab_tokens_count[label] = {k: v * math.log(len(label2vocab_tokens_count) / doc_freq[k]) for k, v in vocab_tokens_count.items()}

        ##   ##    ##   ##   ##   ##
        # by default the threshold is 1*math.log(len(labels)) that is the normalized likely maximum relevance score
        gain_threshold = 1 * math.log(len(label2vocab_tokens_count))
        # we reduce the threshold to half when the number of the minimum label vocabulary is less than 10
        # this is based on such an important intuition: some insufficiently-expanded labels (since these classes usually do not contain too many semantic-related phrases in the corpus, i.e., the class-abstract and imbalance problem) are zero pseudo annotated if the threshold is too high.
        # Reducing it to half helps alleviate this problem
        if min([len(i) for i in label2vocab_tokens_count.values()]) < 10:
            gain_threshold *= 0.5
        # this is the intuition to come up with the scenario for setting up the threshold.
        # The scenario is fixed and implemented within the system without user involvement. It is not dependent on datasets and thus is generalizable to multiple domains.
        ##   ##   ##   ##   ##   ##

        self.logger.info("-----------label vocabulary cut at 10----------------")
        for k, v in label2vocab_tokens_count.items():
            self.logger.info(f"{k}\t{', '.join(list(v.keys())[:10])}")
        self.logger.info("-----------------------------------------------------")

        def select_by_gain_thresh():
            gts, preds = [], []
            unpredicted_examples = []
            predicted_examples = []
            for example in tqdm(examples):
                example_text = pre.clean(example["text"])
                ngram_range = [1, 3]
                ngram_range[-1] = len(example_text.split()) if len(example_text.split()) < ngram_range[-1] else ngram_range[-1]
                grams = get_grams([example_text], ngram_range=tuple(ngram_range))
                tokens = [each for each in grams if each not in stopwords]

                gain_scores = {}
                for label, vocab in self.label2vocab.items():
                    gain = 0
                    vocab_tokens_count = label2vocab_tokens_count[label]
                    for token in tokens:
                        if token in vocab_tokens_count:
                            gain += vocab_tokens_count[token]
                    gain_scores[label] = gain

                if self.data_args.multi_label:
                    preds_this = [k for k, v in gain_scores.items() if v > gain_threshold]
                    if len(preds_this) > 0:
                        preds.append(preds_this)
                        if self.data_args.label_field_name in example:
                            gt = example[self.data_args.label_field_name].split(",")
                            gts.append(gt)
                        example["predicted_label"] = ",".join(preds_this)
                        predicted_examples.append(example)
                    else:
                        unpredicted_examples.append(example)
                else:
                    candidate_label = max(gain_scores.items(), key=lambda k: k[1])[0]
                    if gain_scores[candidate_label] > gain_threshold:
                        if self.data_args.label_field_name in example:
                            gt = example[self.data_args.label_field_name]
                            gts.append(gt)
                        preds.append(candidate_label)
                        example["predicted_label"] = candidate_label
                        predicted_examples.append(example)
                    else:
                        unpredicted_examples.append(example)
            return preds, gts, predicted_examples, unpredicted_examples

        self.logger.info(f" start initial sample selection based on gain threshold, starting from {gain_threshold}")
        preds, gts, predicted_examples, unpredicted_examples = select_by_gain_thresh()
        if self.data_args.label_field_name not in examples[0]:
            return preds, gts, predicted_examples, unpredicted_examples, gain_threshold

        if self.data_args.multi_label:
            def post_process(preds, gts):
                preds_ = []
                for i in preds:
                    preds_.extend(i)
                class_dist_ = Counter(preds_)
                self.logger.info(f"class distribution on predicted examples ({sum(list(Counter(preds_).values()))}): {json.dumps(class_dist_, indent=2)}")
                preds_one_hots = []
                gts_one_hots = []
                for pred, gt in zip(preds, gts):
                    preds_one_hots.append([1 if i in pred else 0 for i in self.raw_classes])
                    gts_one_hots.append([1 if i in gt else 0 for i in self.raw_classes])
                preds, gts = preds_one_hots, gts_one_hots
                return preds_, preds, gts, class_dist_

            preds_, preds, gts, class_dist = post_process(preds, gts)
            self.logger.info(classification_report(gts, preds, digits=4))
            self.logger.info(f"accuracy score: {accuracy_score(gts, preds)}")
        else:
            class_dist = dict(Counter(preds))
            self.logger.info(f"class distribution on predicted examples ({sum(list(Counter(preds).values()))}): {json.dumps(class_dist, indent=2)}")
            self.logger.info(classification_report(gts, preds, digits=4))
        return preds, gts, predicted_examples, unpredicted_examples, class_dist

    def match_and_augment(self, target_set=None, p=0.7, expand_with_top=100, write_examples=False):
        assert hasattr(self, "label2vocab"), "label2vocab has not generated yet, consider to run self.setup_labelvocab first"
        # we need label field in test.json as the ground truth references since testing is undergoing here
        target_set = "test" if target_set is None else target_set
        target_path = os.path.join(self.data_args.data_path, target_set + ".json")
        examples = read_jsonl(target_path)
        self.logger.info(f"\n### Dataset statistics of {target_path}: ###")
        self.report_data_stats(examples)
        preds, gts, predicted_examples, unpredicted_examples, predicted_class_dist = self.make_bow_predictions(examples, p=p,expand_with_top=expand_with_top)
        # eda_expansion = True
        # tested, it did not work
        # if eda_expansion:
        #     aug_examples, aug_class_dist = self.aug_batch_with_eda(predicted_examples, predicted_class_dist, aug_target=aug_target)
        #     self.logger.info(f"class distribution on eda expanded predicted data ({sum(aug_class_dist.values())}): {json.dumps(dict(aug_class_dist), indent=2)}")
        #     predicted_examples = aug_examples
        if gts != []:
            self.logger.info(f"results report on set: {target_set}")
            if self.data_args.multi_label:
                self.logger.info(classification_report(gts, preds, digits=4, target_names=self.raw_classes))
                self.logger.info(f"accuracy score: {accuracy_score(gts, preds)}")
            else:
                self.logger.info(classification_report(gts, preds, digits=4))

        if write_examples:
            def write_examples(to_examples, file_tag="bow_td"):
                tfp = os.path.join(self.data_args.data_path, target_set + f"_{file_tag}.json")
                with open(tfp, "w+") as f:
                    for ex in to_examples:
                        if "predicted_label" in ex:
                            predicted_labels = ex["predicted_label"].split(",")
                            ex["target_dist"] = [1 if i in predicted_labels else 0 for i in self.raw_classes]
                            ex["target"] = predicted_labels
                        f.write(json.dumps(ex) + "\n")
                self.logger.info(f"write {len(to_examples)} examples to: {tfp}")

            write_examples(predicted_examples, "bow_eda_td")
            write_examples(unpredicted_examples, "bow_eda_up")
        return_dict = calculate_perf(preds, gts)
        self.logger.info(json.dumps(return_dict, indent=2))
        return return_dict

    def run_sent_embeddings_predictions(self, examples, classes):
        examples_texts, gts = [], []
        for example in examples:
            examples_texts.append(pre.clean(example["text"]))
            # examples_texts.append(example["text"])
            gts.append(example[self.data_args.label_field_name])

        input_texts = classes + examples_texts
        mean_outputs = self.sent_embeddings(input_texts)
        mean_label_reps = mean_outputs[:len(classes)]
        mean_examples_reps = mean_outputs[len(classes):]
        similarities = get_cos_sim(mean_label_reps, mean_examples_reps)
        values, indices = torch.topk(similarities, 1, 0)
        preds = []
        for _, (values_i, indices_i) in enumerate(zip(values.tolist(), indices.tolist())):
            for j, (value, indice) in enumerate(zip(values_i, indices_i)):
                preds.append(self.raw_classes[indice])
        return gts, preds

    def run_sent_embeddings_baseline(self, expand_label=False, threshold=0.0, grams_match=False, set_name="test"):
        raw_classes = self.raw_classes
        if expand_label:
            assert hasattr(self, "label2vocab"), "label2vocab has not generated yet, consider to run self.setup_labelvocab first"
            extended_classes = []
            for cls in raw_classes:
                cls = cls.replace("_", " ")
                # with top 10 extensions, basically perform at the same level as non-expansion
                extended = ", ".join([v for _, v in self.label2vocab[cls][:10]])
                extended_classes.append(extended)
            classes = extended_classes
        else:
            classes = raw_classes
        # we need label field in test.json as the ground truth references since testing is undergoing here
        test_path = os.path.join(self.data_args.data_path, f"{set_name}.json")
        examples = read_jsonl(test_path)
        if grams_match:
            gts, preds = [], []
            tokenizer = AutoTokenizer.from_pretrained(self.data_args.model_for_labelvocab)
            model = AutoModel.from_pretrained(self.data_args.model_for_labelvocab)
            mean_label_reps = self.sent_embeddings(classes, tokenizer=tokenizer, model=model)
            random.seed(2021)
            random.shuffle(examples)
            for example in tqdm(examples):
                grams = get_grams([example["text"]], ngram_range=(1, 3))
                grams = [example["text"]] if grams == [] else grams
                mean_grams_reps = self.sent_embeddings(grams, tokenizer=tokenizer, model=model, pbar=False)
                similarities = get_cos_sim(mean_grams_reps, mean_label_reps)
                gts.append(example[self.data_args.label_field_name])
                pred_idx = similarities.max(0)[0].argmax().item()
                max_candidate = self.raw_classes[pred_idx]
                if self.data_args.multi_label:
                    candidates = []
                    thre = 0.6  # for multi-label selection: situation
                    for idx, v in enumerate(similarities.max(0)[0]):
                        if v > thre:
                            candidates.append(self.raw_classes[idx])
                    if len(candidates) == 0:
                        preds.append(max_candidate)
                    else:
                        preds.append(",".join(candidates))
                else:
                    preds.append(max_candidate)

        else:
            gts, preds = self.run_sent_embeddings_predictions(examples, classes)

        if self.data_args.multi_label:
            new_gts, new_preds = [], []
            for gt, pred in zip(gts, preds):
                one_hot = [0] * len(self.raw_classes)
                # for multi-label, the label with the highest score is assigned
                for ip in pred.split(","):
                    one_hot[self.raw_classes.index(ip)] = 1
                new_preds.append(one_hot)
                one_hot = [0] * len(self.raw_classes)
                for i_gt in gt.split(","):
                    one_hot[self.raw_classes.index(i_gt)] = 1
                new_gts.append(one_hot)
            gts, preds = new_gts, new_preds
        self.logger.info(f"run sent embeddings baseline results on {set_name} set (expand_label_top_10 = {expand_label}, simple_grams_match = {grams_match}): ")
        self.logger.info(classification_report(gts, preds, digits=4))
        return_dict = calculate_perf(preds, gts)
        return_dict = {f"run_sent_embeddings_baseline(eval_set={set_name})": return_dict}
        self.logger.info(json.dumps(return_dict, indent=2))
        return return_dict

    def setup_labelvocab(self, num_features_to_extract=5000, ngram_range=(1, 3), topk_for_each_class=500, corpus_from=None, labelvocab_file=None, force=False):
        # empirically selected for ngram_range: (1,1), (1,2), (1,3), (2,3),(1,4),(2,4) => (1,3)
        # num_features_to_extract selected for num_features_to_extract: 2500, 5000,10000 => 5000
        # corpus_from: by default, the vocab is generated from the train.json file in self.data_args.data_path. If this is specified, then the corpus from this given corpus_from
        labelvocab_file = os.path.join(self.data_args.data_path, "label2vocab.json" if labelvocab_file is None else labelvocab_file)
        if os.path.isfile(labelvocab_file) and not force:
            self.logger.info(f"found existing label vocab at: {labelvocab_file}, hence load it directly")
            self.label2vocab = json.load(open(labelvocab_file, "r"))
        else:
            self.logger.info(f"not found label vocab, hence start generating it...")
            # it is expected to gain better performance when the given unlabeled corpus is larger
            # but here we use train.json for simplicity by default if corpus_from is not specified
            corpus_from = os.path.join(self.data_args.data_path, "train.json") if corpus_from is None else os.path.join(self.data_args.data_path, corpus_from+".json")
            # train_path = os.path.join(self.data_args.data_path, "original", "train.json")
            grams = get_grams_from_data(corpus_from, num_features_to_extract=num_features_to_extract, ngram_range=ngram_range)
            # we don't need the label field in train.json for generating label2vocab but only a list of classe names
            classes = self.raw_classes
            input_texts = classes + grams
            mean_outputs = self.sent_embeddings(input_texts)
            # mean_outputs
            mean_label_reps = mean_outputs[:len(classes)]
            mean_phrases_reps = mean_outputs[len(classes):]
            similarities = get_cos_sim(mean_label_reps, mean_phrases_reps)
            values, indices = torch.topk(similarities, topk_for_each_class, 1)
            self.label2vocab = {}
            for i, (values_i, indices_i) in enumerate(zip(values.tolist(), indices.tolist())):
                # print("---")
                for j, (value, indice) in enumerate(zip(values_i, indices_i)):
                    # print(f'label: {classes[i]} \t similarity: {value} \t phrase#{j}: {grams[indice]}')
                    if classes[i] not in self.label2vocab:
                        self.label2vocab[classes[i]] = [(grams[indice], value)]
                    else:
                        self.label2vocab[classes[i]].append((grams[indice], value))
            with open(labelvocab_file, "w") as f:
                json.dump(self.label2vocab, f, indent=2)
            self.logger.info(f"configuration: (num_features_to_extract={num_features_to_extract}, ngram_range={ngram_range}, tf_idf_threshold={topk_for_each_class}, corpus_from={corpus_from})")
            self.logger.info(f"label vocab is generated and saved at: {labelvocab_file}")
        return self.label2vocab
