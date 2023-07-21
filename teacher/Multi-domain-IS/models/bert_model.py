import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from gpu_mem_track import MemTracker
from torch.nn.functional import one_hot
gpu_tracker = MemTracker()
import time


class PolicyLoss():
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sample_labels, sample_policy_logits, sample_reward_logits, policy_lambda=0.6, epsilon=5e-4):
        sample_policy_logits = sample_policy_logits.squeeze(1)
        sample_reward_logits = sample_reward_logits.squeeze(1)
        return self.bce_loss(sample_policy_logits, sample_reward_logits)


class Loss_(nn.Module):
    def __init__(self):
        super().__init__()
        self.nllloss = nn.NLLLoss()

    def forward(self, sample_reward_logits, sample_labels):
        return self.nllloss(torch.log(sample_reward_logits), sample_labels)


class ResponseWarmupTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(ResponseWarmupTraining, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_reward_3 = nn.Linear(config.hidden_size, 3)
        self.classifier_reward_1 = nn.Linear(config.hidden_size, 1)
        self.classifier_feedback = nn.Linear(config.hidden_size, 9)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights()
        self.multilabelsoftmarginloss = nn.SoftMarginLoss()
        self.loss_ = Loss_()
        self.softmax = nn.Softmax(dim=1)
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss()
        self.nllloss = nn.NLLLoss()

    def forward(self, sample_input_ids, sample_input_mask, segment_ids, label, feedback_idx, mode):
        if 'answer' in mode:
            return self.update_(sample_input_ids=sample_input_ids,
                                sample_input_mask=sample_input_mask,
                                sample_label=label,
                                sample_segment_ids=segment_ids,
                                sample_feedback_idx=feedback_idx)
        else: 
            return self.update_intent_(sample_input_ids=sample_input_ids,
                                sample_input_mask=sample_input_mask,
                                sample_label=label,
                                sample_segment_ids=segment_ids, 
                                sample_feedback_idx=feedback_idx)

    def update_(self, sample_input_ids, sample_input_mask,
                sample_label, sample_segment_ids, sample_feedback_idx):
        sample_reward_outputs = self.bert(input_ids=sample_input_ids, attention_mask=sample_input_mask,
                                          token_type_ids=sample_segment_ids)
        sample_reward_output = sample_reward_outputs.last_hidden_state[:, 0, :]
        sample_reward_logits = self.classifier_reward_1(sample_reward_output)
        sample_label = sample_label.squeeze(1)
        sample_reward_logits = sample_reward_logits.squeeze(1)

        loss_ = self.bcewithlogitsloss(sample_reward_logits.float(), sample_label.float())
        return loss_, self.sigmoid(sample_reward_logits)

    def update_intent_(self, sample_input_ids, sample_input_mask,
                sample_label, sample_segment_ids, sample_feedback_idx):
        sample_reward_outputs = self.bert(input_ids=sample_input_ids, attention_mask=sample_input_mask,
                                          token_type_ids=sample_segment_ids)
        sample_reward_output = torch.stack(
            [sample_feedback_idx.max(dim=1).indices.view(-1, 1)] * sample_reward_outputs.last_hidden_state.shape[-1],
            dim=-1)
        sample_reward_output = torch.gather(sample_reward_outputs.last_hidden_state, 1, sample_reward_output)

        sample_reward_logits = self.classifier_feedback(sample_reward_output)
        sample_label = sample_label.squeeze(1)
        sample_reward_logits = sample_reward_logits.squeeze(1)
        loss_ = self.bcewithlogitsloss(sample_reward_logits.float(), sample_label.float())
        return loss_, self.sigmoid(sample_reward_logits)


