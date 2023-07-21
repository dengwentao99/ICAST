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
    loss, reward_logits = outputs

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    writer.add_scalar('loss_'+type, loss.item(), global_step)
    return loss


def training_fn(args,
                                     train_intent_dataset,
                                     train_answer_intent_dataset,
                                     train_answer_dataset,
                                     model, tokenizer):
    """ Train the model """
    accelerator = Accelerator(fp16=args.fp16)
    if not os.path.exists(args.output_dir + '/tensorboard'):
        os.mkdir(args.output_dir + '/tensorboard')
    writer = SummaryWriter(args.output_dir + '/tensorboard')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

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
    # Train!
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
    for epoch in range(int(args.num_train_epochs)):
        logger.info("Starting training epoch %d", epoch)
        type = 'ranking'
        start_time = datetime.now()


        type = 'answer'
        for answer_turn in range(answer_turns):
            for step, (batch_answer_intent, batch_answer) in enumerate(zip(train_answer_intent_dataloader, train_answer_dataloader)):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                if epoch % 1 == 0:
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
                model.train()
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
        batch = tuple(t.to(args.device) for t in batch)
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


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    processor = processors[task]()
    cached_features_intent_file = os.path.join(args.data_dir, 'cached-intent-{}_{}_{}'.format(
        data_type,
        str(args.train_max_seq_length if data_type in ['train', 'train_labeled',
                                                       'train_unlabeled'] else args.eval_max_seq_length),
        str(task)))
    cached_features_answer_file = os.path.join(args.data_dir, 'cached-answer-{}_{}_{}'.format(
        data_type,
        str(args.train_max_seq_length if data_type in ['train', 'train_labeled',
                                                       'train_unlabeled'] else args.eval_max_seq_length),
        str(task)))
    cached_features_answer_intent_file = os.path.join(args.data_dir, 'cached-answer_intent-{}_{}_{}'.format(
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
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == 'test':
            examples = processor.get_test_examples(args.data_dir)
        elif 'inference' in data_type:
            examples = processor.get_test_examples(args.data_dir, data_type)
        elif data_type == 'unlabeled':
            examples = processor.get_unlabeled_examples(args.data_dir)
        else:
            examples = processor.get_inference_examples(args.data_dir, data_type)
        features_intent, features_answer_intent, features_answer = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type == 'train'
                                            else args.eval_max_seq_length,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )
        if args.local_rank in [-1, 0]:
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
def inference(args, model, tokenizer, mode='test'):
    file = ZipFile(os.path.join(args.data_dir, 'unlabeled_for_generation.zip'), 'r')
    for f in file.namelist():
        file.extract(f, args.data_dir)
    with open(os.path.join(args.data_dir, mode + '_len_list.tsv'), 'r', encoding='utf-8') as f:
        len_list = f.readlines()
    len_list = [int(i.strip()) for i in len_list]
    os.remove(os.path.join(args.data_dir, mode + '_len_list.tsv'))
    for i in range(max(len_list) + 1):
        data_type = mode + '-' + str(i)
        results = {}
        if args.do_predict2 and args.local_rank in [-1, 0]:
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

            test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                            data_type=data_type)
            test_intent_sampler = SequentialSampler(
                        test_intent_dataset) if args.local_rank == -1 else DistributedSampler(
                        test_intent_dataset)
            test_intent_dataloader = DataLoader(test_intent_dataset, sampler=test_intent_sampler,
                                                batch_size=args.eval_batch_size)
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
                with open(os.path.join(args.data_dir, mode + '-'+str(cj), 'context.tsv'), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [i.strip('\n').split('\t') for i in lines]
                result_lines = []
                assert len(lines) == len(pred_tags)
                if cj == max(len_list):
                    for k_idx, k in enumerate(lines):
                        if len(k) <= 2 * i:
                            result_lines.append('\t'.join(k))
                        else:
                            k[2 * i] = '\t'.join([k[2 * i]] + ['_'.join(pred_tags[k_idx])])
                            result_lines.append('\t'.join(k))
                else:
                    for k_idx, k in enumerate(lines):
                        if len(k) <= i+1:
                            result_lines.append('\t'.join(k))
                        else:
                            if len(pred_tags[k_idx]) == 0:
                                pred_tags[k_idx] = ['']
                            k[i] = ' '.join([k[i]] + pred_tags[k_idx])
                            result_lines.append('\t'.join(k))
                with open(os.path.join(args.data_dir, mode + '-' + str(cj), 'context.tsv'), 'w', encoding='utf-8') as f:
                    for u in result_lines:
                        f.write(u + '\n')
            os.remove(os.path.join(args.data_dir, 'cached-intent-' + mode + '-' + str(i) + '_512_' + args.task_name))
            os.remove(os.path.join(args.data_dir, 'cached-answer-' + mode + '-' + str(i) + '_512_' + args.task_name))
    if os.path.exists(os.path.join(args.data_dir, 'test_inference')):
        shutil.rmtree(os.path.join(args.data_dir, 'test_inference'))
    os.rename(os.path.join(args.data_dir, mode + '-' + str(max(len_list))),
              os.path.join(args.data_dir, 'test_inference'))
    if os.path.exists(os.path.join(args.data_dir, 'cached-answer-' + 'test_inference' + '_512_' + args.task_name)):
        os.remove(os.path.join(args.data_dir, 'cached-answer-' + 'test_inference' + '_512_' + args.task_name))
    dataset_intent, dataset_answer_intent, dataset_answer = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test_inference')

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
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
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
    if args.do_train:
        train_intent_dataset, train_answer_intent_dataset, train_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                                                   data_type='train')
        global_step_re, \
        global_step_fe, \
        re_loss, \
        fe_loss = training_fn(args, train_intent_dataset, train_answer_intent_dataset,
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
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
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
        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print("eval", checkpoint)
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
    results = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                                                              data_type='test')
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

        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
            )
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
            result_feedback = predict(args, model, test_intent_dataloader, test_answer_dataloader, prefix=prefix, type='answer')
            if global_step:
                result_feedback = {"test_answer_feedback_{}_{}".format(global_step, k): v for k, v in
                                   result_feedback.items()}
            results.update(result_feedback)
        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    results = {}
    if args.do_predict2 and args.local_rank in [-1, 0]:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_output_dir = args.output_dir
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.model_name_or_path + "/**/" + WEIGHTS_NAME, recursive=True))
            )
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

            test_intent_dataset, test_answer_intent_dataset, test_answer_dataset = inference(args, model, tokenizer, mode='test')
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

            result_answer_intent = predict(args, model, test_intent_dataloader, test_answer_intent_dataloader, prefix=prefix,
                                      type='answer_i')
            result_answer = predict(args, model, test_intent_dataloader, test_answer_dataloader, prefix=prefix,
                                      type='answer')


if __name__ == "__main__":
    main()
