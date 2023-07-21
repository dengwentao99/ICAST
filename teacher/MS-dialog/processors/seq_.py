""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ import DataProcessor

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_contexts, text_candidate, text_feedback, label, mode):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_contexts: list. contexts list.
            text_candidate: list. candidate utterance.
            label: num. The label for each example.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_contexts = text_contexts
        self.text_candidate = text_candidate
        self.label = label
        self.text_feedback = text_feedback
        self.mode = mode

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids_reward, input_mask_reward, input_len_reward, segment_ids_reward,
                 feedback_idx_ids_reward, label_is_answer, label):

        self.input_ids_reward = input_ids_reward
        self.input_mask_reward = input_mask_reward
        self.segment_ids_reward = segment_ids_reward
        self.input_len_reward = input_len_reward
        self.feedback_idx_ids_reward = feedback_idx_ids_reward
        self.label_is_answer = label_is_answer
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeaturesRanking(object):
    def __init__(self, input_ids_ranking, input_mask_ranking, input_len_ranking, segment_ids_ranking, label):

        self.input_ids_ranking = input_ids_ranking
        self.input_mask_ranking = input_mask_ranking
        self.segment_ids_ranking = segment_ids_ranking
        self.input_len_ranking = input_len_ranking
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeaturesFeedback(object):
    def __init__(self, input_ids_feedback, input_mask_feedback, input_len_feedback, segment_ids_feedback,
                 feedback_idx_feedback, label):

        self.input_ids_feedback = input_ids_feedback
        self.input_mask_feedback = input_mask_feedback
        self.segment_ids_feedback = segment_ids_feedback
        self.input_len_feedback = input_len_feedback
        self.feedback_idx_feedback = feedback_idx_feedback
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_sample_input_ids_reward, all_sample_input_mask_reward, all_sample_segment_ids_reward, \
    all_sample_feedback_idx_ids_reward, \
    all_sample_label_is_answer, all_sample_label = map(torch.stack, zip(*batch))

    return all_sample_input_ids_reward, all_sample_input_mask_reward, all_sample_segment_ids_reward, \
           all_sample_feedback_idx_ids_reward, all_sample_label_is_answer, all_sample_label


def collate_fn_ranking(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids_ranking, all_input_mask_ranking, all_segment_ids_ranking, all_label_ranking, \
    all_input_len_ranking = map(torch.stack, zip(*batch))
    max_len = max(all_input_len_ranking).item()
    all_input_ids_ranking = all_input_ids_ranking[:, :max_len]
    all_input_mask_ranking = all_input_mask_ranking[:, :max_len]
    all_segment_ids_ranking = all_segment_ids_ranking[:, :max_len]
    all_label_ranking = all_label_ranking[:, :max_len]

    return all_input_ids_ranking, all_input_mask_ranking, all_segment_ids_ranking, all_label_ranking


def collate_fn_feedback(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids_feedback, all_input_mask_feedback, all_segment_ids_feedback, all_feedback_pos_feedback, \
    all_label_feedback, all_input_len_feedback = map(torch.stack, zip(*batch))
    max_len = max(all_input_len_feedback).item()
    all_input_ids_feedback = all_input_ids_feedback[:, :max_len]
    all_input_mask_feedback = all_input_mask_feedback[:, :max_len]
    all_segment_ids_feedback = all_segment_ids_feedback[:, :max_len]
    all_feedback_pos_feedback = all_feedback_pos_feedback[:, :max_len]
    all_label_feedback = all_label_feedback[:, :max_len]

    return all_input_ids_feedback, all_input_mask_feedback, all_segment_ids_feedback, all_feedback_pos_feedback, \
           all_label_feedback


def get_context_num(text_contexts, max_contexts_len):
    for i in range(len(text_contexts), 0, -1):
        if len(" [SEP] ".join(text_contexts[-i:]).split()) <= max_contexts_len:
            return i
    return -1


def get_context_candidate(example, tokenizer, sep_token, cls_token, cls2_token, pad_token, feedback_token,
                                   max_seq_length, match_token_segment_id, mask_padding_with_zero, pad_token_segment_id):
    special_token_num = 7
    max_len_current = 0
    # candidate
    tokens_candidate = tokenizer.tokenize(example.text_candidate)[:100]
    # contexts
    tokens_contexts = tokenizer.tokenize(example.text_contexts)[-400:]

    tokens_dia = [cls_token] + tokens_contexts + [sep_token] + tokens_candidate
    segment_ids = [match_token_segment_id] + [1 - match_token_segment_id] * (len(tokens_contexts)) + [match_token_segment_id] * (1 + len(tokens_candidate))
    # label_is_answer_ids = [example.label_is_answer]
    label_ids = [example.label]

    feedback_idx_ids = [0] + [0] * (len(tokens_contexts) + 1 + len(tokens_candidate))

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [MASK] [CLS] u1 [SEP] u2 [SEP] u3 [SEP] [CLS] x [SEP]
    #  type_ids:   0      1    1    1   1   1    1    1    0   0   0

    input_ids = tokenizer.convert_tokens_to_ids(tokens_dia)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(input_ids)
    assert len(tokens_dia) == len(input_ids) == len(segment_ids) == len(feedback_idx_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length
    feedback_idx_ids += [0] * padding_length
    return input_ids, input_mask, input_len, segment_ids, feedback_idx_ids, tokens_dia, label_ids


def get_context_intent(example, tokenizer, sep_token, cls_token, cls2_token, pad_token, feedback_token,
                                   max_seq_length, match_token_segment_id, mask_padding_with_zero, pad_token_segment_id):
    special_token_num = 7
    max_len_current = 0
    # candidate
    tokens_candidate = tokenizer.tokenize(example.text_candidate)[:100]
    # contexts
    tokens_contexts = tokenizer.tokenize(example.text_contexts)[-400:]

    tokens_dia = [cls_token] + tokens_contexts + [sep_token]
    segment_ids = [match_token_segment_id] + [1 - match_token_segment_id] * (len(tokens_contexts)) + [match_token_segment_id]
    # label_is_answer_ids = [example.label_is_answer]
    label_ids = [example.label]

    feedback_idx_ids = [0] + [0] * (len(tokens_contexts)) + [1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [MASK] [CLS] u1 [SEP] u2 [SEP] u3 [SEP] [CLS] x [SEP]
    #  type_ids:   0      1    1    1   1   1    1    1    0   0   0

    input_ids = tokenizer.convert_tokens_to_ids(tokens_dia)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(input_ids)
    assert len(tokens_dia) == len(input_ids) == len(segment_ids) == len(feedback_idx_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length
    feedback_idx_ids += [0] * padding_length
    return input_ids, input_mask, input_len, segment_ids, feedback_idx_ids, tokens_dia, label_ids

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 cls2_token="[AR]",
                                 clsfeedback_token="[RF]",
                                 match_token_segment_id=1,
                                 pad_token="[PAD]", pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True,
                                 feedback_token="[FEEDBACK]",
                                 group_num=10):
    features_answer_intent = []
    features_answer = []
    features_intent = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d ", ex_index, len(examples))
        if example.mode == 'answer1':
            input_ids2, input_mask2, input_len2, segment_ids2, feedback_idx_ids2, tokens_dia2, label2 = \
                get_context_candidate(example, tokenizer, sep_token, cls_token, cls2_token, pad_token, feedback_token,
                                               max_seq_length, match_token_segment_id, mask_padding_with_zero, pad_token_segment_id)
            if ex_index < 0:
                logger.info("answer input_ids: %s", " ".join([str(x) for x in input_ids2]))
                logger.info("answer input_mask: %s", " ".join([str(x) for x in input_mask2]))
                logger.info("answer segment_ids: %s", " ".join([str(x) for x in segment_ids2]))
                logger.info("answer tokens_dia: %s", " ".join([str(x) for x in tokens_dia2]))
                logger.info("answer label: %s", " ".join([str(x) for x in label2]))
            features_answer_intent.append(InputFeaturesFeedback(input_ids_feedback=input_ids2,
                                                         input_mask_feedback=input_mask2,
                                                         input_len_feedback=input_len2,
                                                         segment_ids_feedback=segment_ids2,
                                                         feedback_idx_feedback=feedback_idx_ids2,
                                                         label=label2))

        elif example.mode == 'answer2':
            input_ids2, input_mask2, input_len2, segment_ids2, feedback_idx_ids2, tokens_dia2, label2 = \
                get_context_candidate(example, tokenizer, sep_token, cls_token, cls2_token, pad_token, feedback_token,
                                               max_seq_length, match_token_segment_id, mask_padding_with_zero, pad_token_segment_id)
            if ex_index < 0:
                logger.info("answer input_ids: %s", " ".join([str(x) for x in input_ids2]))
                logger.info("answer input_mask: %s", " ".join([str(x) for x in input_mask2]))
                logger.info("answer segment_ids: %s", " ".join([str(x) for x in segment_ids2]))
                logger.info("answer tokens_dia: %s", " ".join([str(x) for x in tokens_dia2]))
                logger.info("answer label: %s", " ".join([str(x) for x in label2]))
            features_answer.append(InputFeaturesFeedback(input_ids_feedback=input_ids2,
                                                         input_mask_feedback=input_mask2,
                                                         input_len_feedback=input_len2,
                                                         segment_ids_feedback=segment_ids2,
                                                         feedback_idx_feedback=feedback_idx_ids2,
                                                         label=label2))
        else:
            input_ids2, input_mask2, input_len2, segment_ids2, feedback_idx_ids2, tokens_dia2, label2 = \
                get_context_intent(example, tokenizer, sep_token, cls_token, cls2_token, pad_token, feedback_token,
                                      max_seq_length, match_token_segment_id, mask_padding_with_zero,
                                      pad_token_segment_id)
            if ex_index < 0:
                logger.info("intent input_ids: %s", " ".join([str(x) for x in input_ids2]))
                logger.info("intent input_mask: %s", " ".join([str(x) for x in input_mask2]))
                logger.info("intent segment_ids: %s", " ".join([str(x) for x in segment_ids2]))
                logger.info("intent tokens_dia: %s", " ".join([str(x) for x in tokens_dia2]))
                logger.info("intent label: %s", " ".join([str(x) for x in label2]))
            features_intent.append(InputFeaturesFeedback(input_ids_feedback=input_ids2,
                                                         input_mask_feedback=input_mask2,
                                                         input_len_feedback=input_len2,
                                                         segment_ids_feedback=segment_ids2,
                                                         feedback_idx_feedback=feedback_idx_ids2,
                                                         label=label2))

    logger.info("intent data size: %d", len(features_intent))
    logger.info("answer data size: %d", len(features_answer_intent))
    logger.info("answer data size: %d", len(features_answer))
    return features_intent, features_answer_intent, features_answer


class WarmupProcessor(DataProcessor):
    """Processor for the chinese ner data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, 'train')), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, 'valid')), "valid")

    def get_test_examples(self, data_dir, data_type=None):
        """See base class."""
        if data_type is None:
            return self._create_examples(self._read_text(os.path.join(data_dir, 'test')), "test")
        else:
            return self._create_examples(self._read_text(os.path.join(data_dir, data_type)), "test")

    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, 'unlabeled')), "unlabeled")

    def get_inference_examples(self, data_dir, path):
        return self._create_examples_inference(self._read_text(os.path.join(data_dir, path)), "unlabeled")

    def get_labels(self):
        """
        See base class.
        0: response & answer
        1: response & no answer
        2: no response
        """
        return [0, 1, 2]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "{}".format(i)

            context = line['context']
            assert len(context) % 2 == 0
            response = ' '.join(line['response'])
            feedback = ' '.join(line['feedback'])
            response_ns = line['response_ns']
            if len(response_ns) < 9:
                response_ns += [''] * (9 - len(response_ns))
            is_answer_label = line['is_answer']

            answer_context = []
            for idx, ct in enumerate(context):
                if idx % 2 == 1:
                    answer_context.append(' '.join(ct.split('_')))
                else:
                    answer_context.append(ct)
            answer_context = ' '.join(answer_context)
            for step in range(10):
                if step == 0:
                    examples.append(InputExample(guid=guid,
                                                 label=1,
                                                 text_contexts=answer_context,
                                                 text_candidate=response,
                                                 text_feedback=feedback,
                                                 mode='answer1'))
                else:
                    examples.append(InputExample(guid=guid,
                                                 label=0,
                                                 text_contexts=answer_context,
                                                 text_candidate=response_ns[step - 1],
                                                 text_feedback=feedback,
                                                 mode='answer1'))

            answer_context2 = []
            for idx, ct in enumerate(context):
                if idx % 2 == 1:
                    continue
                else:
                    answer_context2.append(ct)
            answer_context2 = ' '.join(answer_context2)
            for step in range(10):
                if step == 0:
                    examples.append(InputExample(guid=guid,
                                                 label=1,
                                                 text_contexts=answer_context2,
                                                 text_candidate=response,
                                                 text_feedback=feedback,
                                                 mode='answer2'))
                else:
                    examples.append(InputExample(guid=guid,
                                                 label=0,
                                                 text_contexts=answer_context2,
                                                 text_candidate=response_ns[step - 1],
                                                 text_feedback=feedback,
                                                 mode='answer2'))

            tags = ['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']
            intent_context = []
            for m in range(0, len(context), 2):
                for idx, ct in enumerate(context[: m+1]):
                    if idx % 2 == 1:
                        intent_context.append(' '.join(ct.split('_')))
                    else:
                        intent_context.append(ct)
                intent_tags = context[m+1].split('_')
                intent_labels = [0] * len(tags)
                for idx, item in enumerate(tags):
                    if item in intent_tags:
                        intent_labels[idx] = 1
                intent_context_ = ' '.join(intent_context)
                examples.append(InputExample(guid=guid,
                                             label=intent_labels,
                                             text_contexts=intent_context_,
                                             text_candidate=response,
                                             text_feedback=feedback,
                                             mode='intent'))
                intent_context = []
        return examples

    def _create_examples_inference(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "{}".format(i)

            context = line['context']
            response = ' '.join(line['response'])
            feedback = ' '.join(line['feedback'])
            response_ns = line['response_ns']
            if len(response_ns) < 9:
                response_ns += [''] * (9 - len(response_ns))
            is_answer_label = line['is_answer']
            answer_context = ' '.join(context)
            examples.append(InputExample(guid=guid,
                                         label=[0] * 12,
                                         text_contexts=answer_context,
                                         text_candidate=response,
                                         text_feedback=feedback,
                                         mode='intent'))
        return examples


task = ['answer', 'answerfeedback', 'answerintent', 'responsewarmup']
ResponseRank_processors = {
    "responsewarmup": WarmupProcessor
}
