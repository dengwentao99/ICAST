import json

from sklearn.metrics import classification_report, ndcg_score, average_precision_score
import numpy as np
import random
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys


def compute_mAP(labels, outputs):
    y_true = np.array(labels)
    y_pred = np.array(outputs)
    AP = []
    for i in range(len(y_true)):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def precision_recall_fscore_k(y_trues, y_preds, k=3, digs=6):
    y_preds = [pred[:k] for pred in y_preds]
    unique_labels = [1, 0]
    num_classes = 2
    results_dict = {}
    results = ''
    for label in unique_labels:
        current_label_result = []
        tp_fn = y_trues.count(label)
        tp_fp = 0
        for y_pred in y_preds:
            if label in y_pred:
                tp_fp += 1
        tp = 0
        for i in range(len(y_trues)):
            if y_trues[i] == label and label in y_preds[i]:
                tp += 1

        support = tp_fn

        try:
            precision = round(tp / tp_fp, digs)
            recall = round(tp / tp_fn, digs)
            f1_score = round(2 * (precision * recall) / (precision + recall), digs)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0

        current_label_result.append(precision)
        current_label_result.append(recall)
        current_label_result.append(f1_score)
        current_label_result.append(support)
        results_dict[str(label)] = {'precision': precision, 'recall': recall,
                                    'F-1': f1_score, 'support': support}
    sums = len(y_trues)
    macro_avg_results = [(results_dict['1']['precision'], results_dict['1']['recall'], results_dict['1']['F-1']),
                         (results_dict['0']['precision'], results_dict['0']['recall'], results_dict['0']['F-1'])]
    weighted_avg_results = [(results_dict['1']['precision'] * results_dict['1']['support'],
                             results_dict['1']['recall'] * results_dict['1']['support'],
                             results_dict['1']['F-1'] * results_dict['1']['support']),
                            (results_dict['0']['precision'] * results_dict['0']['support'],
                             results_dict['0']['recall'] * results_dict['0']['support'],
                             results_dict['0']['F-1'] * results_dict['0']['support'])]

    macro_precision = 0
    macro_recall = 0
    macro_f1_score = 0
    for macro_avg_result in macro_avg_results:
        macro_precision += macro_avg_result[0]
        macro_recall += macro_avg_result[1]
        macro_f1_score += macro_avg_result[2]
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1_score /= num_classes

    results_dict['macro_avg'] = {'precision': macro_precision, 'recall': macro_recall,
                                    'F-1': macro_f1_score}

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1_score = 0
    for weighted_avg_result in weighted_avg_results:
        weighted_precision += weighted_avg_result[0]
        weighted_recall += weighted_avg_result[1]
        weighted_f1_score += weighted_avg_result[2]

    weighted_precision /= sums
    weighted_recall /= sums
    weighted_f1_score /= sums

    results_dict['weighted_avg'] = {'precision': weighted_precision, 'recall': weighted_recall,
                                    'F-1': weighted_f1_score}
    return results_dict


def get_rank_index(score_list):
    score_list = [(i, j) for i, j in enumerate(score_list)]
    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    for i, j in enumerate(score_list):
        if j[0] == 0: 
            return i


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()
        self.classification_report = None

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        return self.classification_report

    def update(self, labels, preds):
        '''
        labels: [[0],[1],[0],....]
        preds: [[1],[0],[0],.....]

        :param labels:
        :param preds:
        :return:
        Example:
        '''
        self.classification_report = classification_report(preds, labels, output_dict=True)
        print(self.classification_report)


class ValidScore(object):
    def __init__(self, id2label, markup='bios', threshold=0.5):
        self.markup = markup
        self.threshold = threshold
        self.reset()
        self.classification_report = None
        self.classification_report_K = None
        self.mAPs = None
        self.nDCG_K = None

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        results = dict()
        if self.mAPs != None:
            results['mAPs'] = self.mAPs
        if self.nDCG_K:
            results['nDCG_K'] = self.nDCG_K
        if self.classification_report:
            results['binary_classification'] = {"1": self.classification_report['1'], 'accuracy': self.classification_report['accuracy']}
        if self.classification_report_K:
            results['classification_report_K'] = {"@1": self.classification_report_K['@1']['1'],
                                                  "@2": self.classification_report_K['@2']['1'],
                                                  "@5": self.classification_report_K['@5']['1']}
        return results

    def update(self, labels, preds, pred_logits=None):
        '''
        labels: [0,1,0,....]
        preds: [1,0,0,.....]

        :param labels:
        :param preds:
        :return:
        Example:
        '''
        self.classification_report = classification_report(labels, preds, output_dict=True)
        pred_logits = [pred_logits[i:i + 10] for i in range(0, len(pred_logits), 10)]

        pred_argmax_idx = [np.argmax(i) for i in pred_logits]
        pred_idx = [get_rank_index(i) for i in pred_logits]
        labels = [labels[i:i + 10] for i in range(0, len(labels), 10)]
        labels_ = [1 for i in range(len(labels))]
        results = [[0] * 10 for i in range(len(pred_logits))]
        for i in range(len(pred_idx)):
            results[i][pred_idx[i]] = 1
        classification_report_1 = precision_recall_fscore_k(labels_, results, k=1, digs=5)
        classification_report_2 = precision_recall_fscore_k(labels_, results, k=2, digs=5)
        classification_report_5 = precision_recall_fscore_k(labels_, results, k=5, digs=5)
        self.classification_report_K = {'@1': classification_report_1,
                                        '@2': classification_report_2,
                                        '@5': classification_report_5}
        self.mAPs = compute_mAP(labels=labels, outputs=pred_logits)

class TestScore(object):
    def __init__(self, id2label, markup='bios', threshold=0.5):
        self.id2label = id2label
        self.markup = markup
        self.threshold = threshold
        self.reset()
        self.classification_report = None
        self.classification_report_K = None
        self.mAPs = None
        self.nDCG_K = None

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        results = dict()
        if self.mAPs:
            results['mAPs'] = self.mAPs
        if self.nDCG_K:
            results['nDCG_K'] = self.nDCG_K
        if self.classification_report:
            results['binary_classification'] = self.classification_report
        if self.classification_report_K:
            results['classification_report_K'] = self.classification_report_K
        print('MAP:', self.mAPs)
        print('nDCG', self.nDCG_K)
        print('Classification report @ K:', self.classification_report_K)
        return results

    def update(self, labels, preds, pred_logits):
        '''
        labels: [0,1,0,....]
        preds: [1,0,0,.....]

        :param labels:
        :param preds:
        :return:
        Example:
        '''
        self.classification_report = classification_report(preds, labels, output_dict=True)
        print(self.classification_report)
        pred_logits = [pred_logits[i:i + 10] for i in range(0, len(pred_logits), 10)]
        pred_argmax_idx = [np.argmax(i) for i in pred_logits]
        pred_idx = [get_rank_index(i) if i[np.argmax(i)] > self.threshold else -1 for i in pred_logits]
        APs = [1 / (i + 1) if i != -1 else 0 for i in pred_idx]
        self.mAPs = sum(APs) / len(APs)
        labels = [labels[i:i + 10] for i in range(0, len(labels), 10)]
        labels_ = [1 for i in range(len(labels))]
        results = [[0] * 10 for i in range(len(pred_logits))]
        for i in range(len(pred_idx)):
            results[i][pred_idx[i]] = 1 if pred_idx[i] != -1 else 0
        classification_report_1 = precision_recall_fscore_k(labels_, results, k=1, digs=6)
        classification_report_3 = precision_recall_fscore_k(labels_, results, k=3, digs=6)
        classification_report_5 = precision_recall_fscore_k(labels_, results, k=5, digs=6)
        self.classification_report_K = {'@1': classification_report_1,
                                        '@3': classification_report_3,
                                        '@5': classification_report_5}
        ndcg_1 = ndcg_score(labels, pred_logits, k=1)
        ndcg_3 = ndcg_score(labels, pred_logits, k=3)
        ndcg_5 = ndcg_score(labels, pred_logits, k=5)
        self.nDCG_K = {'@1': ndcg_1,
                       '@3': ndcg_3,
                       '@5': ndcg_5}



if __name__ == '__main__':
    targets, pred_tags, pred_logits = [], [], []
    with open('./response_ranking_result.txt', 'r') as f:
        for line in f:
            t, pt, pl = int(line.split('\t')[0]), int(line.split('\t')[1]), float(line.split('\t')[2])
            targets.append(t)
            pred_tags.append(pt)
            pred_logits.append(pl)
    responses = []
    with open('./datasets/MSDialog/ResponseRank/test.tsv') as f:
        for line in f:
            responses.append(line.split('\t')[-1])
    print(len(responses), len(targets))
    with open('text.txt', 'w') as ff:
        for i, j, k in zip(targets, pred_tags, responses):
            if i == j == 1:
                ff.write(k)
    answers = []
    with open('./datasets/MSDialog/MSDialog-Complete.json') as f:
        a = json.load(f)
    for i in a:
        for us in a[i]['utterances']:
            if us['is_answer'] == 1:
                answers.append("<<<AGENT>>>: " + us['utterance'])
    print(len(answers))
    total = 0
    for r in responses:
        if r in answers:
            total += 1
    print(total / len(responses))
    metric = TestScore({0: 1, 1: 0})
    metric.update(targets, pred_tags, pred_logits)
    metric.result()
