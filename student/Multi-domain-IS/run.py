import glob
import logging
import os
import json
import shutil
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from models.bert_model import ResponseWarmupTraining
from processors.utils_ import get_entities
from processors.seq_ import convert_examples_to_features
from processors.seq_ import ResponseRank_processors as processors
from processors.seq_ import collate_fn_ranking, collate_fn_feedback
from metrics.metrics_ import ValidScore, TestScore
from tools.finetuning_argparse import get_argparse
from tensorboardX import SummaryWriter
from datetime import datetime

from accelerate import Accelerator



MODEL_CLASSES = {
    'bert': (BertConfig, ResponseWarmupTraining, BertTokenizer),
}
additional_special_tokens = ['[EOD]', '[FEEDBACK]',
                             '[RR]', '[RF]', '[AR]', '[AF]',
                             '[nonesentence]', '[url]', ':)', ':(', ':-(', ':-)',
                             'OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'OO']


def _loss_(args, batch, model, accelerator, global_step, writer, type):

    inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1], "feedback_idx": batch[3],
              "label": batch[4], "mode": type}
    if args.model_type != "distilbert":
        inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
    start_time = datetime.now()
    outputs = model(**inputs)
    loss, reward_logits = outputs[:2]

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    writer.add_scalar('loss_'+type, loss.item(), global_step)
    return loss

from zipfile import ZipFile
def inference(args, model, tokenizer, accelerator, mode='unlabeled_train'):
    file = ZipFile(os.path.join(args.unlabeled_data_dir, 'unlabeled_for_generation.zip'), 'r')
    for f in file.namelist():
        file.extract(f, args.unlabeled_data_dir)
    with open(os.path.join(args.unlabeled_data_dir, mode + '_len_list.tsv'), 'r', encoding='utf-8') as f:
        len_list = f.readlines()
    len_list = [int(i.strip()) for i in len_list]
    os.remove(os.path.join(args.unlabeled_data_dir, mode + '_len_list.tsv'))
    for i in range(max(len_list)):
        data_type = mode + '-' + str(i)
        results = {}
        len_mask = [0] * len(len_list)

        for idx, ll in enumerate(len_list):
            if i + 1 <= ll:
                len_mask[idx] = 1
        if args.do_predict2 and args.local_rank in [-1, 0]:
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

            test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                            data_type=data_type, data_mask=len_mask)
            test_intent_sampler = SequentialSampler(
                        test_intent_dataset) if args.local_rank == -1 else DistributedSampler(
                        test_intent_dataset)
            test_intent_dataloader = DataLoader(test_intent_dataset, sampler=test_intent_sampler,
                                                batch_size=args.eval_batch_size)
            test_intent_dataloader = accelerator.prepare(test_intent_dataloader)
            logits = predict_intent(args, model, test_intent_dataloader, type='intent')
            tags = ['OQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'O']
            pred_tags = []
            for m in logits:
                tmp = []
                for idx, p in enumerate(m):
                    if p > 0.5:
                        tmp.append(tags[idx])
                pred_tags.append(tmp)
            for cj in range(i+1, max(len_list) + 1):
                with open(os.path.join(args.unlabeled_data_dir, mode + '-' + str(cj), 'context.tsv'), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [i.strip('\n').split('\t') for i in lines]
                result_lines = []
                if cj == max(len_list):
                    t = 0
                    for k_idx, k in enumerate(lines):
                        if len(k) <= 2 * i:
                            result_lines.append('\t'.join(k))
                        else:
                            k[2 * i] = '\t'.join([k[2 * i]] + ['_'.join(pred_tags[t])])
                            result_lines.append('\t'.join(k))
                        if len_mask[k_idx] == 1:
                            t += 1
                else:
                    t = 0
                    for k_idx, k in enumerate(lines):
                        if len(k) <= i+1:
                            result_lines.append('\t'.join(k))
                        else:
                            if len(pred_tags[t]) == 0:
                                pred_tags[t] = ['']
                            k[i] = ' '.join([k[i]] + pred_tags[t])
                            result_lines.append('\t'.join(k))
                        if len_mask[k_idx] == 1:
                            t += 1
                with open(os.path.join(args.unlabeled_data_dir, mode + '-' + str(cj), 'context.tsv'), 'w', encoding='utf-8') as f:
                    for u in result_lines:
                        f.write(u + '\n')
            if os.path.exists(os.path.join(args.unlabeled_data_dir, 'cached-intent-' + mode + '-' + str(i) + '_512_' + args.task_name)):
                os.remove(os.path.join(args.unlabeled_data_dir, 'cached-intent-' + mode + '-' + str(i) + '_512_' + args.task_name))
            if os.path.exists(os.path.join(args.unlabeled_data_dir,
                                           'cached-answer-' + mode + '-' + str(i) + '_512_' + args.task_name)):
                os.remove(os.path.join(args.unlabeled_data_dir, 'cached-answer-' + mode + '-' + str(i) + '_512_' + args.task_name))
    if os.path.exists(os.path.join(args.unlabeled_data_dir, mode)):
        shutil.rmtree(os.path.join(args.unlabeled_data_dir, mode))
    os.rename(os.path.join(args.unlabeled_data_dir, mode + '-' + str(max(len_list))),
              os.path.join(args.unlabeled_data_dir, mode))
    if os.path.exists(os.path.join(args.unlabeled_data_dir, 'cached-answer-' + mode + '_512_' + args.task_name)):
        os.remove(os.path.join(args.unlabeled_data_dir, 'cached-answer-' + mode + '_512_' + args.task_name))
    dataset_intent, dataset_answer_intent, dataset_answer = load_and_cache_examples(args,
                                                                                    args.task_name,
                                                                                    tokenizer,
                                                                                    data_type=mode)
    return dataset_intent, dataset_answer_intent, dataset_answer


def shannon_entropy_2D(input_tensor):
    return -(input_tensor * torch.log(input_tensor + 1e-10)).sum(1).reshape(-1, 1)


def shannon_entropy_3D(input_tensor):
    return -(input_tensor * torch.log(input_tensor + 1e-10)).sum(2)


def information_gain_score(input_tensor):
    mean_logits = input_tensor.mean(dim=0)
    first_term = shannon_entropy_3D(input_tensor).mean(0).reshape(-1, 1)
    second_term = shannon_entropy_2D(input_tensor.mean(0))
    var = - first_term + second_term
    return var


def _predict_batch_certainty(args, model, test_answer_dataloader, prefix,
                   checkpoint_dir=None, T=10, type='answer'):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("***** Running %s *****", prefix)
    logger.info("  Num examples = %d", len(test_answer_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_answer_dataloader)
    pred_logits, pred_tags, targets, vars = [], [], [], []
    for step, batch in enumerate(test_answer_dataloader):
        if step % 20 == 0:
            logger.info(str(step))
        model.train()
        with torch.no_grad():
            inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                      "label": batch[4], 'feedback_idx': batch[3], 'mode': 'answer'}
            if args.model_type != "distilbert":
                inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            deter_loss, deter_logits = outputs[:2]
            logits = deter_logits
            tags = deter_logits > threshold
            tmp_test_loss = deter_loss
        if args.n_gpu > 1:
            tmp_test_loss = tmp_test_loss.mean()
        if step % 500 == 0:
            print("[test] [" + type + "] step {} / {} loss {}".format(step, test_data_len, tmp_test_loss))
        test_loss += tmp_test_loss.item()
        nb_test_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        pred_tags += tags
        targets += out_label_ids
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir
    with open(os.path.join(resultcheck_dir, args.task_name + '_MSDialog_' + type + '_predict_' + str(int(time.time())) + '.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k in zip(targets, pred_tags, pred_logits):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
    return targets, pred_tags, pred_logits, vars


def _predict_batch_uncertainty(args, model, test_answer_dataloader, prefix,
                   checkpoint_dir=None, T=10, type='answer'):
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("***** Running %s *****", prefix)
    logger.info("  Num examples = %d", len(test_answer_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_answer_dataloader)
    logits_list = []
    pred_logits, pred_tags, targets, vars = [], [], [], []
    for step, batch in enumerate(test_answer_dataloader):
        if step % 10 == 0:
            logger.info(str(step))
        model.train()
        with torch.no_grad():
            inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                      "label": batch[4], 'feedback_idx': batch[3], 'mode': 'answer'}
            if args.model_type != "distilbert":
                inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            logits_list = []
            model.train()
            tmp_test_loss = 0
            for t in range(T):
                outputs = model(**inputs)
                mc_loss, mc_reward_logits = outputs[:2]
                logits_list.append(torch.stack([1 - mc_reward_logits, mc_reward_logits], dim=1))
                tmp_test_loss += mc_loss
            mc_logits_ = torch.stack(logits_list, dim=0)
            var = information_gain_score(mc_logits_)
            mean_mc_logits = mc_logits_.mean(dim=0)
            logits = mean_mc_logits[:, 1]
            tags = logits > threshold
            tmp_test_loss = tmp_test_loss / T
        if args.n_gpu > 1:
            tmp_test_loss = tmp_test_loss.mean()
        if step % 500 == 0:
            print("[test] [" + type + "] step {} / {} loss {}".format(step, test_data_len, tmp_test_loss))
        test_loss += tmp_test_loss.item()
        nb_test_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        var = var.float().cpu().squeeze().tolist()
        pred_tags += tags
        targets += out_label_ids
        vars += var
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir
    with open(os.path.join(resultcheck_dir, args.task_name + '_MSDialog_' + type + '_predict_' + str(int(time.time())) + '.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k, m in zip(targets, pred_tags, pred_logits, vars):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\t' + str(m) + '\n')
    return targets, pred_tags, pred_logits, vars


def _predict_batch(args, model, test_answer_dataloader, prefix,
                   checkpoint_dir=None, T=10, type='answer'):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("***** Running %s *****", prefix)
    logger.info("  Num examples = %d", len(test_answer_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_answer_dataloader)
    logits_list = []
    pred_logits, pred_tags, targets, vars = [], [], [], []
    output_features = []
    for step, batch in enumerate(test_answer_dataloader):
        if step % 10 == 0:
            logger.info(str(step))
        model.train()
        with torch.no_grad():
            inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                      "label": batch[4], 'feedback_idx': batch[3], 'mode': 'answer'}
            if args.model_type != "distilbert":
                inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            logits_list = []
            if T == 1:
                model.eval()
            else:
                model.train()
            for t in range(T):
                logits_list.append(torch.stack([1 - mc_reward_logits, mc_reward_logits], dim=1))
            mc_logits_ = torch.stack(logits_list, dim=0)
            mean_mc_logits = mc_logits_.mean(dim=0)
            var = information_gain_score(mc_logits_)
            model.eval()
            deter_loss, deter_logits, deter_reward_output = model(**inputs)
            output_features += deter_reward_output.cpu().tolist()
            logits = mean_mc_logits[:, 1]
            tags = logits > 0.40
            # Thresholds for different experimental settings: [0.40, 0.75, 0.50]
            tmp_test_loss = deter_loss
        if args.n_gpu > 1:
            tmp_test_loss = tmp_test_loss.mean()
        if step % 500 == 0:
            print("[test] [" + type + "] step {} / {} loss {}".format(step, test_data_len, tmp_test_loss))
        test_loss += tmp_test_loss.item()
        nb_test_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        var = var.float().cpu().squeeze().tolist()
        pred_tags += tags
        targets += out_label_ids
        vars += var
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir
    with open(os.path.join(resultcheck_dir, args.task_name + '_MSDialog_' + type + '_predict.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k, m in zip(targets, pred_tags, pred_logits, vars):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\t' + str(m) + '\n')
    return targets, pred_tags, pred_logits, vars, output_features


def filter_intent_answer(args, pseudo_answer_intent_dataloader, pseudo_answer_dataloader, model, tokenizer, T=10, data_type='train_inference'):
    assert len(pseudo_answer_intent_dataloader) == len(pseudo_answer_dataloader)
    logger.info('use intent selection' + str(args.useintentselection))
    if args.useintentselection:
        _ai, pred_tags_ai, pred_logits_ai, vars_ai = _predict_batch_uncertainty(args, model,
                                                                    pseudo_answer_intent_dataloader,
                                                                    prefix='answer_i',
                                                                    T=T, type='answer_i')
        _a, pred_tags_a, pred_logits_a, vars_a = _predict_batch_uncertainty(args, model,
                                                                pseudo_answer_dataloader,
                                                                prefix='answer',
                                                                T=T, type='answer')
        assert len(pred_tags_ai) == len(pred_tags_a) == len(pred_logits_ai) == len(pred_logits_a) == len(vars_ai) == len(vars_a)
        unlabeled_data_mask = [0] * len(pred_tags_ai)
        pseudo_labels = [0] * len(pred_logits_ai)
        pred_logits = []
        vars_logits = []
        for idx, (ig_ai, ig_a, p_ai, p_a) in enumerate(zip(vars_ai, vars_a, pred_logits_ai, pred_logits_a)):
            if ig_ai - ig_a > 0.02:
                pred_logits.append(p_ai)
                pseudo_labels[idx] = p_ai
                vars_logits.append(ig_ai)
                unlabeled_data_mask[idx] = 2
            else:
                pred_logits.append(p_a)
                pseudo_labels[idx] = p_a
                vars_logits.append(ig_a)
                unlabeled_data_mask[idx] = 1
        assert len(pred_logits) == len(pred_logits_ai) == len(vars_logits) == len(vars_ai)
        pred_logits = torch.tensor(pred_logits)
        vars_logits = torch.tensor(vars_logits)
        pool_len = 10
        upper = 0.8
        down = 0.1
        for i in range(0, pred_logits.shape[0], pool_len):
            pool = pred_logits[i: i + 10]
            var = vars_logits[i: i + 10]
            pos_num = (pool > upper).sum()
            var_num = (-var < -0.2).sum()
            if pos_num != 1:
                if var_num == 1 and pos_num == 0 and (pool > 0.5).sum() == 1:
                    continue
                else:
                    for j in range(pool_len):
                        idx = i + j
                        unlabeled_data_mask[idx] = 0
            else:
                for j in range(pool_len):
                    idx = i + j
                    if not (pred_logits[idx] < down or pred_logits[idx] > upper):
                        unlabeled_data_mask[idx] = 0
        assert len(pseudo_labels) % 10 == 0 and len(unlabeled_data_mask) % 10 == 0

        with open(os.path.join(args.unlabeled_data_dir, data_type, 'da_mask.tsv'), 'w') as f:
            for i in range(0, len(unlabeled_data_mask), 10):
                f.write('\t'.join([str(m) for m in unlabeled_data_mask[i: i + 10]]) + '\n')
        with open(os.path.join(args.unlabeled_data_dir, data_type, 'pseudo_labels.tsv'), 'w') as f:
            for i in range(0, len(pseudo_labels), 10):
                f.write('\t'.join([str(m) for m in pseudo_labels[i: i + 10]]) + '\n')

        dataset_intent, dataset_answer_intent, dataset_answer = load_and_cache_examples(args,
                                                                                            args.task_name,
                                                                                            tokenizer,
                                                                                            data_type='train_pseudo')



    else:
        _ai, pred_tags_ai, pred_logits_ai, vars_ai = _predict_batch_certainty(args, model,
                                                                              pseudo_answer_intent_dataloader,
                                                                              prefix='answer_i',
                                                                              T=5, type='answer_i')
        assert len(pred_tags_ai) == len(pred_logits_ai)
        unlabeled_data_mask = [0] * len(pred_tags_ai)
        pseudo_labels = [0] * len(pred_logits_ai)
        pred_logits = []
        for idx, p_ai in enumerate(pred_logits_ai):
            pred_logits.append(p_ai)
            pseudo_labels[idx] = p_ai
            unlabeled_data_mask[idx] = 2

        assert len(pred_logits) == len(pred_logits_ai)
        pred_logits = torch.tensor(pred_logits)
        pool_len = 10
        upper = 0.8
        down = 0.1
        vars_logits = torch.tensor(vars_ai)
        for i in range(0, pred_logits.shape[0], pool_len):
            pool = pred_logits[i: i + 10]
            var = vars_logits[i: i + 10]
            pos_num = (pool > upper).sum()
            var_num = (-var < -0.2).sum()
            if pos_num != 1:
                if var_num == 1 and pos_num == 0 and (pool > 0.5).sum() == 1:
                    continue
                else:
                    for j in range(pool_len):
                        idx = i + j
                        unlabeled_data_mask[idx] = 0
            else:
                for j in range(pool_len):
                    idx = i + j
                    if not (pred_logits[idx] < down or pred_logits[idx] > upper):
                        unlabeled_data_mask[idx] = 0
        assert len(pseudo_labels) % 10 == 0 and len(unlabeled_data_mask) % 10 == 0
        with open(os.path.join(args.unlabeled_data_dir, data_type, 'da_mask.tsv'), 'w') as f:
            for i in range(0, len(unlabeled_data_mask), 10):
                f.write('\t'.join([str(m) for m in unlabeled_data_mask[i: i + 10]]) + '\n')
        with open(os.path.join(args.unlabeled_data_dir, data_type, 'pseudo_labels.tsv'), 'w') as f:
            for i in range(0, len(pseudo_labels), 10):
                f.write('\t'.join([str(m) for m in pseudo_labels[i: i + 10]]) + '\n')

        dataset_intent, dataset_answer_intent, dataset_answer = load_and_cache_examples(args,
                                                                                        args.task_name,
                                                                                        tokenizer,
                                                                                        data_type='train_pseudo')
    return dataset_intent, dataset_answer_intent, dataset_answer


def use_unlabeled_data_for_training(args,
                                    train_intent_dataset,
                                    train_answer_intent_dataset,
                                    train_answer_dataset,
                                    model, tokenizer):

    accelerator = Accelerator(fp16=args.fp16)
    if not os.path.exists(args.output_dir + '/tensorboard'):
        os.mkdir(args.output_dir + '/tensorboard')
    writer = SummaryWriter(args.output_dir + '/tensorboard')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_answer_intent_sampler = SequentialSampler(train_answer_intent_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_answer_intent_dataset)
    train_answer_intent_dataloader = DataLoader(train_answer_intent_dataset, sampler=train_answer_intent_sampler,
                                         batch_size=args.train_batch_size)

    train_answer_sampler = SequentialSampler(train_answer_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_answer_dataset)
    train_answer_dataloader = DataLoader(train_answer_dataset, sampler=train_answer_sampler,
                                            batch_size=args.train_batch_size)

    train_intent_sampler = SequentialSampler(train_intent_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_intent_dataset)
    train_intent_dataloader = DataLoader(train_intent_dataset, sampler=train_intent_sampler,
                                         batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_answer_sampler) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_answer_sampler) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier_reward_1.named_parameters())
    intent_param_optimizer = list(model.classifier_feedback.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in intent_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in intent_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running training *****")
    logger.info("  Num ranking examples = %d", len(train_answer_sampler))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_answer_sampler) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_answer_sampler) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_ranking_loss = 0.0
    tr_feedback_loss = 0.0
    model.zero_grad()
    optimizer.zero_grad()
    seed_everything(args.seed)

    if args.save_steps == -1 and args.logging_steps == -1:
        args.logging_steps = len(train_answer_dataloader)
        args.save_steps = len(train_answer_dataloader)
    global_step_intent = 0
    global_step_answer_intent = 0
    global_step_answer = 0
    answer_turns = 1
    logger.info("answer turns = %d", answer_turns)
    intent_turns = 1
    logger.info("intent turns = %d", intent_turns)
    model, optimizer, train_answer_dataloader, \
    train_answer_intent_dataloader = accelerator.prepare(model,
                                                         optimizer,
                                                         train_answer_dataloader,
                                                         train_answer_intent_dataloader)
    using_unlabeled_turns = 4  # Set 4 for 1% or 5% labeled data and 8 for 10% labeled data
    for ut in range(using_unlabeled_turns):
        data_dir = 'train_inference'
        pseudo_intent_dataset_, pseudo_answer_intent_dataset_, pseudo_answer_dataset_ = inference(args, model, tokenizer,
                                                                                                accelerator,
                                                                                               mode=data_dir)
        pseudo_answer_intent_sampler_ = SequentialSampler(pseudo_answer_intent_dataset_) if args.local_rank == -1 \
            else DistributedSampler(pseudo_answer_intent_dataset_)
        pseudo_answer_intent_dataloader_ = DataLoader(pseudo_answer_intent_dataset_, sampler=pseudo_answer_intent_sampler_,
                                             batch_size=args.eval_batch_size)
        pseudo_answer_sampler_ = SequentialSampler(pseudo_answer_dataset_) if args.local_rank == -1 \
            else DistributedSampler(pseudo_answer_dataset_)
        pseudo_answer_dataloader_ = DataLoader(pseudo_answer_dataset_,
                                                      sampler=pseudo_answer_sampler_,
                                                      batch_size=args.eval_batch_size)
        pseudo_answer_intent_dataloader_ = accelerator.prepare(pseudo_answer_intent_dataloader_)
        pseudo_answer_dataloader_ = accelerator.prepare(pseudo_answer_dataloader_)
        pseudo_intent_dataset, pseudo_answer_intent_dataset, pseudo_answer_dataset = filter_intent_answer(args,
                                                                                                          pseudo_answer_intent_dataloader_,
                                                                                                          pseudo_answer_dataloader_,
                                                                                                          model,
                                                                                                          tokenizer, T=5,
                                                                                                          data_type=data_dir)
        pseudo_answer_intent_sampler = SequentialSampler(pseudo_answer_intent_dataset) if args.local_rank == -1 \
            else DistributedSampler(pseudo_answer_intent_dataset)
        pseudo_answer_intent_dataloader = DataLoader(pseudo_answer_intent_dataset, sampler=pseudo_answer_intent_sampler,
                                                    batch_size=args.train_batch_size)

        pseudo_answer_sampler = SequentialSampler(pseudo_answer_dataset) if args.local_rank == -1 \
            else DistributedSampler(pseudo_answer_dataset)
        pseudo_answer_dataloader = DataLoader(pseudo_answer_dataset, sampler=pseudo_answer_sampler,
                                             batch_size=args.train_batch_size)

        pseudo_intent_sampler = SequentialSampler(pseudo_intent_dataset) if args.local_rank == -1 \
            else DistributedSampler(pseudo_intent_dataset)
        pseudo_intent_dataloader = DataLoader(pseudo_intent_dataset, sampler=pseudo_intent_sampler,
                                             batch_size=args.train_batch_size)

        pseudo_intent_dataloader, pseudo_answer_intent_dataloader, pseudo_answer_dataloader = accelerator.prepare(pseudo_intent_dataloader, pseudo_answer_intent_dataloader, pseudo_answer_dataloader)
        for epoch in range(5):
            logger.info("Starting training epoch %d", epoch)
            type = 'ranking'
            start_time = datetime.now()
            type = 'answer'
            for answer_turn in range(answer_turns):
                for step, (batch_answer_intent, batch_answer) in enumerate(zip(train_answer_intent_dataloader, train_answer_dataloader)):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss_ = _loss_(args, batch_answer_intent, model, accelerator, global_step_answer_intent, writer, type + '_i')
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info("epoch {} step {} / {}    answer_i loss {}    time_consumed {}".format(epoch, step, len(train_answer_intent_dataloader), loss_.item(), datetime.now() - start_time))
                    global_step_answer_intent += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_answer_intent += 1
                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step_answer_intent % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir, "checkpoint-answer_intent-{}".format(global_step_answer_intent))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    loss_ = _loss_(args, batch_answer, model, accelerator, global_step_answer, writer, type)
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info(
                            "epoch {} step {} / {}    answer loss {}    time_consumed {}".format(epoch, step,
                                                                                                   len(train_answer_dataloader),
                                                                                                   loss_.item(),
                                                                                                   datetime.now() - start_time))
                    global_step_answer += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_answer += 1
                        if args.local_rank in [-1,
                                               0] and args.save_steps > 0 and global_step_answer % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir,
                                                      "checkpoint-answer-{}".format(global_step_answer))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            type = 'intent'
            train_intent_dataloader = accelerator.prepare(train_intent_dataloader)
            for intent_turn in range(intent_turns):
                for step, batch in enumerate(train_intent_dataloader):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss_ = _loss_(args, batch, model, accelerator, global_step_intent, writer, type)
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info("epoch {} step {} / {}    intent generation loss {}    time_consumed {}".format(epoch, step,
                                                                                                           len(train_intent_dataloader),
                                                                                                           loss_.item(),
                                                                                                           datetime.now() - start_time))
                    global_step_intent += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_intent += 1
                        if args.local_rank in [-1,
                                               0] and args.save_steps > 0 and global_step_intent % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir,
                                                      "checkpoint-intent-{}".format(global_step_intent))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            logger.info("\n")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()

            type = 'answer'
            for answer_turn in range(answer_turns):
                for step, batch_answer_intent in enumerate(pseudo_answer_intent_dataloader):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss_ = _loss_(args, batch_answer_intent, model, accelerator, global_step_answer_intent, writer, type + '_i')
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info("epoch {} step {} / {}  pseudo answer_i loss {}    time_consumed {}".format(epoch, step, len(pseudo_answer_intent_dataloader), loss_.item(), datetime.now() - start_time))
                    global_step_answer_intent += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_answer_intent += 1
                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step_answer_intent % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir, "checkpoint-answer_intent-{}".format(global_step_answer_intent))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                for step, batch_answer in enumerate(pseudo_answer_dataloader):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss_ = _loss_(args, batch_answer, model, accelerator, global_step_answer, writer, type)
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info(
                            "epoch {} step {} / {}  pseudo answer loss {}    time_consumed {}".format(epoch, step,
                                                                                                   len(pseudo_answer_dataloader),
                                                                                                   loss_.item(),
                                                                                                   datetime.now() - start_time))
                    global_step_answer += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_answer += 1
                        if args.local_rank in [-1,
                                               0] and args.save_steps > 0 and global_step_answer % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir,
                                                      "checkpoint-answer-{}".format(global_step_answer))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            type = 'intent'
            pseudo_intent_dataloader = accelerator.prepare(pseudo_intent_dataloader)
            for intent_turn in range(intent_turns):
                for step, batch in enumerate(pseudo_intent_dataloader):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()

                    loss_ = _loss_(args, batch, model, accelerator, global_step_intent, writer, type)
                    optimizer.step()
                    scheduler.step()
                    if step % 1000 == 0:
                        logger.info("epoch {} step {} / {}   pseudo intent generation loss {}    time_consumed {}".format(epoch, step,
                                                                                                           len(pseudo_intent_dataloader),
                                                                                                           loss_.item(),
                                                                                                           datetime.now() - start_time))
                    global_step_intent += 1
                    tr_feedback_loss += loss_.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        global_step_intent += 1
                        if args.local_rank in [-1,
                                               0] and args.save_steps > 0 and global_step_intent % args.save_steps == 0 and epoch > args.logging_epochs:
                            output_dir = os.path.join(args.output_dir,
                                                      "checkpoint-intent-{}".format(global_step_intent))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_vocabulary(output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            logger.info("\n")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()

    return global_step_intent, global_step_answer, tr_ranking_loss / global_step_intent, \
           tr_feedback_loss / global_step_answer



def evaluate(args, model, writer, global_step, eval_dataloader, prefix="", type=None, resultcheck_dir=None):
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    eval_loss = 0.0
    nb_eval_steps = 0
    pred_logits, pred_tags, targets = [], [], []
    eval_data_len = len(eval_dataloader)
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        with torch.no_grad():
            if type == 'ranking':
                inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                          "label": batch[3]}
                if args.model_type != "distilbert":
                    inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            else:
                inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                          "label": batch[4]}
                if args.model_type != "distilbert":
                    inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = logits.argmax(1)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()
        if step % 2000 == 0:
            print("[valid] [" + type + "] step {} / {} loss {}".format(step, eval_data_len, tmp_eval_loss))
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        out_label_ids = [0 if i[2] == 1 else 1 for i in out_label_ids]
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        tags = [0 if i == 2 else 1 for i in tags]
        pred_tags += tags
        targets += out_label_ids
    resultcheck_dir = resultcheck_dir if resultcheck_dir != None else args.output_dir
    with open(os.path.join(resultcheck_dir, args.task_name + '_MANtIS_apple_' + type + '_result.txt'), 'w', encoding='utf-8') as f:
        for i, j, k in zip(targets, pred_tags, pred_logits):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
    metric.update(targets, pred_tags)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    writer.add_scalar('eval_loss_' + type, eval_loss, global_step)
    eval_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval " + type + " results %s *****", prefix)
    logger.info(str(eval_info))
    return results


def evaluate2(args, model, eval_dataloader, prefix="", type=None, checkpoint_dir=None):
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    eval_loss = 0.0
    nb_eval_steps = 0
    eval_data_len = len(eval_dataloader)
    pred_logits, pred_tags, targets = [], [], []
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if type == 'ranking':
                # ranking
                inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                          "label": batch[3]}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            else:
                # feedback
                inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                          "label": batch[4]}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = logits > threshold
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()
        if step % 500 == 0:
            print("[valid] [" + type + "] step {} / {} loss {}".format(step, eval_data_len, tmp_eval_loss))
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        pred_tags += tags
        targets += out_label_ids
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir
    with open(os.path.join(resultcheck_dir, args.task_name + '_MANtIS_apple_is_answer_' + type + '_result.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k in zip(targets, pred_tags, pred_logits):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
    metric.update(targets, pred_tags, torch.tensor(pred_logits))
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval " + type + " results %s *****", prefix)
    logger.info(str(eval_info))

    return results


def predict(args, model, test_intent_dataloader, test_answer_dataloader, prefix="", type=None, checkpoint_dir=None):
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(test_answer_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_answer_dataloader)
    pred_logits, pred_tags, targets = [], [], []
    for step, batch in enumerate(test_answer_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                      "label": batch[4], 'feedback_idx': batch[3], 'mode': 'answer'}
            if args.model_type != "distilbert":
                inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_test_loss, logits = outputs[:2]
            tags = logits > threshold
        if args.n_gpu > 1:
            tmp_test_loss = tmp_test_loss.mean()
        if step % 500 == 0:
            print("[test] [" + type + "] step {} / {} loss {}".format(step, test_data_len, tmp_test_loss))
        test_loss += tmp_test_loss.item()
        nb_test_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits += logits.cpu().squeeze().tolist()
        tags = tags.long().cpu().squeeze().tolist()
        pred_tags += tags
        targets += out_label_ids
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir

    report_path = os.path.join(resultcheck_dir, 'answer_intent_report')
    if not os.path.exists(report_path):
        os.mkdir(report_path)
    with open(os.path.join(report_path, args.task_name + '_MSDialog_is_answer_' + type + '_result.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k in zip(targets, pred_tags, pred_logits):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
    metric.update(targets, pred_tags, torch.tensor(pred_logits))
    logger.info("\n")
    test_loss = test_loss / nb_test_steps
    test_info = metric.result()
    results = {f'{key}': value for key, value in test_info.items()}
    results['loss'] = test_loss
    logger.info("***** Test " + type + " results %s *****", prefix)
    logger.info(str(test_info))

    return results


def predict_my(args, model, test_intent_dataloader, test_answer_intent_dataloader, test_answer_dataloader, prefix="", type=None, checkpoint_dir=None, T=10):
    accelerator = Accelerator(fp16=args.fp16)
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_answer_dataloader)
    pred_logits, pred_tags, targets = [], [], []
    output_features = []
    model = accelerator.prepare(model)
    test_answer_intent_dataloader = accelerator.prepare(test_answer_intent_dataloader)
    test_answer_dataloader = accelerator.prepare(test_answer_dataloader)
    _ai, pred_tags_ai, pred_logits_ai, vars_ai, feat_m_ai = _predict_batch(args, model, test_answer_intent_dataloader, prefix='answer_i', T=T, type='answer_i')
    _a, pred_tags_a, pred_logits_a, vars_a, feat_m_a = _predict_batch(args, model, test_answer_dataloader, prefix='answer', T=T, type='answer')

    assert len(pred_tags_ai) % 10 == 0
    targets = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * (len(pred_tags_ai) // 10)
    assert len(targets) == len(pred_logits_ai)
    for v_ai, v_a, p_t_ai, p_t_a, p_l_ai, p_l_a, f_m_ai, f_m_a in zip(vars_ai, vars_a, pred_tags_ai, pred_tags_a, pred_logits_ai, pred_logits_a, feat_m_ai, feat_m_a):
        if v_ai - v_a > 0.02:
            pred_tags.append(p_t_ai)
            pred_logits.append(p_l_ai)
            output_features.append(f_m_ai)
        else:
            pred_tags.append(p_t_a)
            pred_logits.append(p_l_a)
            output_features.append(f_m_a)
    assert len(pred_tags) == len(pred_logits) == len(pred_logits_ai)
    resultcheck_dir = checkpoint_dir if checkpoint_dir != None else args.output_dir

    report_path = os.path.join(resultcheck_dir, 'answer_intent_report')
    if not os.path.exists(report_path):
        os.mkdir(report_path)
    with open(os.path.join(report_path, args.task_name + '_MSDialog_is_answer_' + type + '_result.txt'), 'w',
              encoding='utf-8') as f:
        for i, j, k in zip(targets, pred_tags, pred_logits):
            f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
    with open(os.path.join(resultcheck_dir, 'MANtIS_1_features.txt'), 'w',
              encoding='utf-8') as f:
        output_features_ = ['\t'.join([str(j) for j in i]) for i in output_features]
        for idx, (i, k) in enumerate(zip(output_features_, pred_tags)):
            if idx % 10 == 0:
                gg_l = '1'
            else:
                gg_l = '0'
            f.write(gg_l + '\t' + str(k) + '\t' + i + '\n')
    metric.update(targets, pred_tags, torch.tensor(pred_logits))
    logger.info("\n")
    test_info = metric.result()
    results = {f'{key}': value for key, value in test_info.items()}
    logger.info("***** Test " + type + " results %s *****", prefix)
    logger.info(str(test_info))


def predict_intent(args, model, test_intent_dataloader, prefix="", type=None, checkpoint_dir=None):
    metric = ValidScore(args.id2label, markup=args.markup)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(test_intent_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    threshold = 0.5
    test_loss = 0.0
    nb_test_steps = 0
    test_data_len = len(test_intent_dataloader)
    pred_logits, pred_tags, targets = [], [], []
    for step, batch in enumerate(test_intent_dataloader):
        model.eval()
        with torch.no_grad():
            inputs = {"sample_input_ids": batch[0], "sample_input_mask": batch[1],
                      "label": batch[4], 'feedback_idx': batch[3], 'mode': 'intent'}
            if args.model_type != "distilbert":
                inputs["segment_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_test_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_test_loss = tmp_test_loss.mean()
        if step % 1000 == 0:
            print("[test] [" + type + "] step {} / {} loss {}".format(step, test_data_len, tmp_test_loss))
        test_loss += tmp_test_loss.item()
        nb_test_steps += 1
        out_label_ids = inputs['label'].cpu().squeeze().tolist()
        pred_logits.append(logits)
        targets += out_label_ids

    return torch.cat(pred_logits, dim=0)


def load_and_cache_examples(args, task, tokenizer, data_type='train', data_mask=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    if data_type == 'train_inference' or data_type == 'train_pseudo' or 'train_inference' in data_type:
        data_dir = args.unlabeled_data_dir
    elif data_type == 'test_inference' or data_type == 'test_pseudo' or 'test_inference' in data_type:
        data_dir = args.unlabeled_data_dir
    else:
        data_dir = args.data_dir
    cached_features_intent_file = os.path.join(data_dir, 'cached-intent-{}_{}_{}'.format(
        data_type,
        str(args.train_max_seq_length if data_type in ['train', 'train_labeled',
                                                       'train_unlabeled'] else args.eval_max_seq_length),
        str(task)))
    cached_features_answer_file = os.path.join(data_dir, 'cached-answer-{}_{}_{}'.format(
        data_type,
        str(args.train_max_seq_length if data_type in ['train', 'train_labeled',
                                                       'train_unlabeled'] else args.eval_max_seq_length),
        str(task)))
    cached_features_answer_intent_file = os.path.join(data_dir, 'cached-answer_intent-{}_{}_{}'.format(
        data_type,
        str(args.train_max_seq_length if data_type in ['train', 'train_labeled',
                                                       'train_unlabeled'] else args.eval_max_seq_length),
        str(task)))

    if os.path.exists(cached_features_answer_file) and os.path.exists(cached_features_answer_intent_file) and os.path.exists(cached_features_intent_file) and not args.overwrite_cache:
        logger.info("Loading answer selection with intent features from cached file %s", cached_features_answer_intent_file)
        features_answer_intent = torch.load(cached_features_answer_intent_file)
        logger.info("Loading answer features from cached file %s", cached_features_answer_file)
        features_answer = torch.load(cached_features_answer_file)
        logger.info("Loading intent features from cached file %s", cached_features_intent_file)
        features_intent = torch.load(cached_features_intent_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        elif data_type == 'test':
            examples = processor.get_test_examples(data_dir)
        elif data_type == 'train_inference':
            examples = processor.get_test_examples(data_dir, 'train_inference')
        elif 'train_inference' in data_type:
            examples = processor.get_inference_examples(data_dir, data_type, data_mask)
        elif data_type == 'test_inference':
            examples = processor.get_test_examples(data_dir, 'test_inference')
        elif 'test_inference' in data_type:
            examples = processor.get_inference_examples(data_dir, data_type, data_mask)
        elif data_type == 'train_pseudo':
            examples = processor.get_pseudo_examples(data_dir, 'train_inference')
        elif 'inference' in data_type:
            examples = processor.get_test_examples(data_dir, data_type)
        elif data_type == 'unlabeled':
            examples = processor.get_unlabeled_examples(data_dir)
        else:
            examples = processor.get_inference_examples(data_dir, data_type)
        features_intent, features_answer_intent, features_answer = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train'
                                            else args.eval_max_seq_length,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
        if args.local_rank in [-1, 0] and data_dir != args.unlabeled_data_dir:
            logger.info("Saving answer selection with intent features into cached file %s", cached_features_answer_intent_file)
            torch.save(features_answer_intent, cached_features_answer_intent_file)
            logger.info("Saving answer features into cached file %s", cached_features_answer_file)
            torch.save(features_answer, cached_features_answer_file)
            logger.info("Saving intent features into cached file %s", cached_features_intent_file)
            torch.save(features_intent, cached_features_intent_file)



    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier() 
    all_input_ids_feedback = []
    all_input_mask_feedback = []
    all_input_len_feedback = []
    all_segment_ids_feedback = []
    all_feedback_pos_feedback = []
    all_label_feedback = []
    for ff_idx, ff in enumerate(features_answer_intent):
        # if f_idx % 10 != 0:
        #     continue
        input_ids_feedback = ff.input_ids_feedback
        input_mask_feedback = ff.input_mask_feedback
        segment_ids_ranking = ff.segment_ids_feedback
        input_len_feedback = ff.input_len_feedback
        feedback_pos_feedback = ff.feedback_idx_feedback
        label_feedback = ff.label

        all_input_ids_feedback.append(input_ids_feedback)
        all_input_mask_feedback.append(input_mask_feedback)
        all_input_len_feedback.append(input_len_feedback)
        all_segment_ids_feedback.append(segment_ids_ranking)
        all_feedback_pos_feedback.append(feedback_pos_feedback)
        all_label_feedback.append(label_feedback)

    all_input_ids_feedback = torch.tensor(all_input_ids_feedback, dtype=torch.long)
    all_input_mask_feedback = torch.tensor(all_input_mask_feedback, dtype=torch.long)
    all_input_len_feedback = torch.tensor(all_input_len_feedback, dtype=torch.long)
    all_segment_ids_feedback = torch.tensor(all_segment_ids_feedback, dtype=torch.long)
    all_feedback_pos_feedback = torch.tensor(all_feedback_pos_feedback, dtype=torch.long)
    all_label_feedback = torch.tensor(all_label_feedback, dtype=torch.long)

    dataset_answer_intent = TensorDataset(all_input_ids_feedback,
                                   all_input_mask_feedback,
                                   all_segment_ids_feedback,
                                   all_feedback_pos_feedback,
                                   all_label_feedback,
                                   all_input_len_feedback)

    all_input_ids_feedback = []
    all_input_mask_feedback = []
    all_input_len_feedback = []
    all_segment_ids_feedback = []
    all_feedback_pos_feedback = []
    all_label_feedback = []
    for ff_idx, ff in enumerate(features_answer):
        input_ids_feedback = ff.input_ids_feedback
        input_mask_feedback = ff.input_mask_feedback
        segment_ids_ranking = ff.segment_ids_feedback
        input_len_feedback = ff.input_len_feedback
        feedback_pos_feedback = ff.feedback_idx_feedback
        label_feedback = ff.label

        all_input_ids_feedback.append(input_ids_feedback)
        all_input_mask_feedback.append(input_mask_feedback)
        all_input_len_feedback.append(input_len_feedback)
        all_segment_ids_feedback.append(segment_ids_ranking)
        all_feedback_pos_feedback.append(feedback_pos_feedback)
        all_label_feedback.append(label_feedback)

    all_input_ids_feedback = torch.tensor(all_input_ids_feedback, dtype=torch.long)
    all_input_mask_feedback = torch.tensor(all_input_mask_feedback, dtype=torch.long)
    all_input_len_feedback = torch.tensor(all_input_len_feedback, dtype=torch.long)
    all_segment_ids_feedback = torch.tensor(all_segment_ids_feedback, dtype=torch.long)
    all_feedback_pos_feedback = torch.tensor(all_feedback_pos_feedback, dtype=torch.long)
    all_label_feedback = torch.tensor(all_label_feedback, dtype=torch.long)

    dataset_answer = TensorDataset(all_input_ids_feedback,
                                     all_input_mask_feedback,
                                     all_segment_ids_feedback,
                                     all_feedback_pos_feedback,
                                     all_label_feedback,
                                     all_input_len_feedback)

    all_input_ids_feedback = []
    all_input_mask_feedback = []
    all_input_len_feedback = []
    all_segment_ids_feedback = []
    all_feedback_pos_feedback = []
    all_label_feedback = []
    for ff_idx, ff in enumerate(features_intent):
        # if f_idx % 10 != 0:
        #     continue
        input_ids_feedback = ff.input_ids_feedback
        input_mask_feedback = ff.input_mask_feedback
        segment_ids_ranking = ff.segment_ids_feedback
        input_len_feedback = ff.input_len_feedback
        feedback_pos_feedback = ff.feedback_idx_feedback
        label_feedback = ff.label

        all_input_ids_feedback.append(input_ids_feedback)
        all_input_mask_feedback.append(input_mask_feedback)
        all_input_len_feedback.append(input_len_feedback)
        all_segment_ids_feedback.append(segment_ids_ranking)
        all_feedback_pos_feedback.append(feedback_pos_feedback)
        all_label_feedback.append(label_feedback)

    all_input_ids_feedback = torch.tensor(all_input_ids_feedback, dtype=torch.long)
    all_input_mask_feedback = torch.tensor(all_input_mask_feedback, dtype=torch.long)
    all_input_len_feedback = torch.tensor(all_input_len_feedback, dtype=torch.long)
    all_segment_ids_feedback = torch.tensor(all_segment_ids_feedback, dtype=torch.long)
    all_feedback_pos_feedback = torch.tensor(all_feedback_pos_feedback, dtype=torch.long)
    all_label_feedback = torch.tensor(all_label_feedback, dtype=torch.long)

    dataset_intent = TensorDataset(all_input_ids_feedback,
                                   all_input_mask_feedback,
                                   all_segment_ids_feedback,
                                   all_feedback_pos_feedback,
                                   all_label_feedback,
                                   all_input_len_feedback)

    return dataset_intent, dataset_answer_intent, dataset_answer


from zipfile import ZipFile
def test_inference(args, model, tokenizer, accelerator, mode='test'):
    data_dir = args.unlabeled_data_dir
    file = ZipFile(os.path.join(args.unlabeled_data_dir, 'unlabeled_for_generation.zip'), 'r')
    for f in file.namelist():
        file.extract(f, args.unlabeled_data_dir)
    with open(os.path.join(args.unlabeled_data_dir, mode + '_len_list.tsv'), 'r', encoding='utf-8') as f:
        len_list = f.readlines()
    len_list = [int(i.strip()) for i in len_list]
    os.remove(os.path.join(args.unlabeled_data_dir, mode + '_len_list.tsv'))
    for i in range(max(len_list)):
        data_type = mode + '-' + str(i)
        results = {}
        len_mask = [0] * len(len_list)
        for idx, ll in enumerate(len_list):
            if i + 1 <= ll:
                len_mask[idx] = 1
        if args.do_predict2 and args.local_rank in [-1, 0]:
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

            test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = load_and_cache_examples(args,
                                                                                                           args.task_name,
                                                                                                           tokenizer,
                                                                                                           data_type=data_type,
                                                                                                           data_mask=len_mask)
            test_intent_sampler = SequentialSampler(
                test_intent_dataset) if args.local_rank == -1 else DistributedSampler(
                test_intent_dataset)
            test_intent_dataloader = DataLoader(test_intent_dataset, sampler=test_intent_sampler,
                                                batch_size=args.eval_batch_size)
            test_intent_dataloader = accelerator.prepare(test_intent_dataloader)
            logits = predict_intent(args, model, test_intent_dataloader, type='intent')
            tags = ['OQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'O']
            pred_tags = []
            for m in logits:
                tmp = []
                for idx, p in enumerate(m):
                    if p > 0.5:
                        tmp.append(tags[idx])
                pred_tags.append(tmp)
            for cj in range(i + 1, max(len_list) + 1):
                with open(os.path.join(args.unlabeled_data_dir, mode + '-' + str(cj), 'context.tsv'), 'r',
                          encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [i.strip('\n').split('\t') for i in lines]
                result_lines = []
                if cj == max(len_list):
                    t = 0
                    for k_idx, k in enumerate(lines):
                        if len(k) <= 2 * i:
                            result_lines.append('\t'.join(k))
                        else:
                            k[2 * i] = '\t'.join([k[2 * i]] + ['_'.join(pred_tags[t])])
                            result_lines.append('\t'.join(k))
                        if len_mask[k_idx] == 1:
                            t += 1
                else:
                    t = 0
                    for k_idx, k in enumerate(lines):
                        if len(k) <= i + 1:
                            result_lines.append('\t'.join(k))
                        else:
                            if len(pred_tags[t]) == 0:
                                pred_tags[t] = ['']
                            k[i] = ' '.join([k[i]] + pred_tags[t])
                            result_lines.append('\t'.join(k))
                        if len_mask[k_idx] == 1:
                            t += 1
                with open(os.path.join(args.unlabeled_data_dir, mode + '-' + str(cj), 'context.tsv'), 'w',
                          encoding='utf-8') as f:
                    for u in result_lines:
                        f.write(u + '\n')
            if os.path.exists(os.path.join(data_dir, 'cached-intent-' + mode + '-' + str(i) + '_512_' + args.task_name)):
                os.remove(os.path.join(data_dir, 'cached-intent-' + mode + '-' + str(i) + '_512_' + args.task_name))
            if os.path.exists(
                    os.path.join(data_dir, 'cached-answer-' + mode + '-' + str(i) + '_512_' + args.task_name)):
                os.remove(os.path.join(data_dir, 'cached-answer-' + mode + '-' + str(i) + '_512_' + args.task_name))
    if os.path.exists(os.path.join(data_dir, mode)):
        shutil.rmtree(os.path.join(data_dir, mode))
    os.rename(os.path.join(data_dir, mode + '-' + str(max(len_list))),
              os.path.join(data_dir, mode))
    if os.path.exists(os.path.join(data_dir, 'cached-answer-' + mode + '_512_' + args.task_name)):
        os.remove(os.path.join(data_dir, 'cached-answer-' + mode + '_512_' + args.task_name))
    dataset_intent, dataset_answer_intent, dataset_answer = load_and_cache_examples(args, args.task_name, tokenizer, data_type=mode)

    return dataset_intent, dataset_answer_intent, dataset_answer


def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    seed_everything(args.seed)
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,)
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_intent_dataset, train_answer_intent_dataset, train_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                                                   data_type='train')
        global_step_re, \
        global_step_fe, \
        re_loss, \
        fe_loss = use_unlabeled_data_for_training(args, train_intent_dataset, train_answer_intent_dataset,
                                                   train_answer_dataset, model, tokenizer)
        logger.info(" global_step = %s, average ranking loss = %s", global_step_re, re_loss)
        logger.info(" global_step = %s, average feedback loss = %s", global_step_fe, fe_loss)
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # loading eval dataloader
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        eval_feedback_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                                              data_type='dev')
        eval_feedback_sampler = SequentialSampler(
            eval_feedback_dataset) if args.local_rank == -1 else DistributedSampler(
            eval_feedback_dataset)
        eval_feedback_dataloader = DataLoader(eval_feedback_dataset, sampler=eval_feedback_sampler,
                                              batch_size=args.eval_batch_size)

        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

        checkpoints = checkpoints_name
        if args.eval_all_checkpoints:
            checkpoints = checkpoints_name
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
            tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)
            result_feedback = evaluate2(args, model, eval_feedback_dataloader, prefix=prefix, type='feedback')
            if global_step:
                result_feedback = {"eval_answer_feedback_{}_{}".format(global_step, k): v for k, v in result_feedback.items()}
            results.update(result_feedback)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
    # testing
    results = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        # loading eval dataloader
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                                              data_type='test')
        # Note that DistributedSampler samples randomly
        test_answer_sampler = SequentialSampler(
            test_answer_dataset) if args.local_rank == -1 else DistributedSampler(
            test_answer_dataset)
        test_answer_dataloader = DataLoader(test_answer_dataset, sampler=test_answer_sampler,
                                              batch_size=args.eval_batch_size)

        test_intent_sampler = SequentialSampler(
            test_intent_dataset) if args.local_rank == -1 else DistributedSampler(
            test_intent_dataset)
        test_intent_dataloader = DataLoader(test_intent_dataset, sampler=test_intent_sampler,
                                              batch_size=args.eval_batch_size)
        checkpoints = checkpoints_name
        if args.eval_all_checkpoints:
            checkpoints = checkpoints_name
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
            tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)
            result_feedback = predict(args, model, test_intent_dataloader, test_answer_dataloader, prefix=prefix, type='answer')
            if global_step:
                result_feedback = {"test_answer_feedback_{}_{}".format(global_step, k): v for k, v in
                                   result_feedback.items()}
            results.update(result_feedback)
        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # testing2
    accelerator = Accelerator(fp16=args.fp16)
    results = {}
    if args.do_predict2 and args.local_rank in [-1, 0]:
        # loading eval dataloader
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        checkpoints = checkpoints_name
        if args.eval_all_checkpoints:
            checkpoints = checkpoints_name
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
            tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)

            test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = test_inference(args, model, tokenizer, accelerator, mode='test_inference')
            test_answer_intent_sampler = SequentialSampler(
                test_answer_intent_dataset) if args.local_rank == -1 else DistributedSampler(
                test_answer_intent_dataset)
            test_answer_intent_dataloader = DataLoader(test_answer_intent_dataset, sampler=test_answer_intent_sampler,
                                                batch_size=args.eval_batch_size)


            test_answer_sampler = SequentialSampler(
                test_answer_dataset) if args.local_rank == -1 else DistributedSampler(
                test_answer_dataset)
            test_answer_dataloader = DataLoader(test_answer_dataset, sampler=test_answer_sampler,
                                                batch_size=args.eval_batch_size)

            test_intent_sampler = SequentialSampler(
                test_intent_dataset) if args.local_rank == -1 else DistributedSampler(
                test_intent_dataset)
            test_intent_dataloader = DataLoader(test_intent_dataset, sampler=test_intent_sampler,
                                                batch_size=args.eval_batch_size)
            # _ai, pred_tags_ai, pred_logits_ai, vars_ai = _predict_batch(args, model,
            #                                                             test_intent_dataloader,
            #                                                             prefix='answer_i',
            #                                                             T=T, type='answer_i')
            # _a, pred_tags_a, pred_logits_a, vars_a = _predict_batch(args, model,
            #                                                             test_answer_dataloader,
            #                                                             prefix='answer',
            #                                                             T=T, type='answer')
            #################################
            result_answer_intent = predict_my(args, model, test_intent_dataloader, test_answer_intent_dataloader, test_answer_dataloader, prefix=prefix,
                                      type='answer_i', T=5)
            #################################
            # result_answer = predict(args, model, test_intent_dataloader, test_answer_intent_dataloader, prefix=prefix,
            #                           type='answer_i')
            #################################################
            # result_answer = predict(args, model, test_intent_dataloader, test_answer_dataloader, prefix=prefix,
            #                         type='answer')


if __name__ == "__main__":
    main()
