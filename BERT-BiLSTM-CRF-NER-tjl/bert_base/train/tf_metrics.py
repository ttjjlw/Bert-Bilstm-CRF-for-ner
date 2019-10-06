"""
Multiclass
from: 
https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

"""

__author__ = "Guillaume Genthial"

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


__all__ = ['precision', 'recall', 'f1', 'fbeta', 'safe_div', 'pr_re_fbeta', 'pr_re_fbeta', 'metrics_from_confusion_matrix']


def precision(labels, predictions, num_classes, pos_indices=None,
              weights=None, average='micro'):
    """Multi-class precision metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    op, _, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (pr, op)


def recall(labels, predictions, num_classes, pos_indices=None, weights=None,
           average='micro'):
    """Multi-class recall metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)



def f1(labels, predictions, num_classes, pos_indices=None, weights=None,
       average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights,
                 average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None,
          average='micro', beta=1):
    """Multi-class fbeta metric for Tensorflow
    Parameters
    ----------
    labels : Tensor of tf.int32 or tf.int64
        The true labels
    predictions : Tensor of tf.int32 or tf.int64
        The predictions, same shape as labels
    num_classes : int
        The number of classes
    pos_indices : list of int, optional
        The indices of the positive classes, default is all
    weights : Tensor of tf.int32, optional
        Mask, must be of compatible shape with labels
    average : str, optional
        'micro': counts the total number of true positives, false
            positives, and false negatives for the classes in
            `pos_indices` and infer the metric from it.
        'macro': will compute the metric separately for each class in
            `pos_indices` and average. Will not account for class
            imbalance.
        'weighted': will compute the metric separately for each class in
            `pos_indices` and perform a weighted average by the total
            number of true labels for each class.
    beta : int, optional
        Weight of precision in harmonic mean
    Returns
    -------
    tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)

def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro',
                                  beta=1):
    """Precision, Recall and F1 from the confusion matrix
    Parameters
    ----------
    cm : tf.Tensor of type tf.int32, of shape (num_classes, num_classes)
        The streaming confusion matrix.
    pos_indices : list of int, optional
        The indices of the positive classes
    beta : int, optional
        Weight of precision in harmonic mean
    average : str, optional
        'micro', 'macro' or 'weighted'
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            cm_mask[idx, :] = 1
            n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

        if average == 'macro':
            pr = tf.reduce_mean(precisions)
            re = tf.reduce_mean(recalls)
            fbeta = tf.reduce_mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = tf.reduce_sum(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta

    else:
        raise NotImplementedError()


def BIO2line_file(BIO_file,result_file):
    '''
    BIO_file:每行一个字 一个tag，字和tag之间用\t或空格隔开。每个句子以一个换行符隔开
    '''
    f_write = open(result_file, 'w', encoding='utf-8')
    with open(BIO_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        for line in lines:
            if line == '':
                continue
            tokens = line.split('\n')
            features = []
            tags = []
            for token in tokens:
                feature_tag = token.split()
                if len(feature_tag) < 2:
                    print(feature_tag)
                    continue
                features.append(feature_tag[0])
                tags.append(feature_tag[-1])
            samples = []
            i = 0
            while i < len(features):
                sample = []
                if tags[i] == 'O':
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j] == 'O':
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/o')

                else:
                    if tags[i][0] != 'B':
                        print(tags[i][0] + ' error start')
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j].split('-')[0] == 'I' and tags[j].split('-')[-1] == tags[i].split('-')[-1]:
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][-1])
                i = j
            f_write.write('  '.join(samples) + '\n')
    f_write.close()


def recover_reduce_sentence_length(reduce_file,raw_file,output_file):
	'''
	#reduce_file：用reduce_length 的predict 文件预测出的结果文件
	#output_file：恢复原有的句子数量
	'''
	output=open(output_file,'w',encoding='utf-8')
	reduce=open(reduce_file,'r',encoding='utf-8')
	raw=open(raw_file,'r',encoding='utf-8')
	i=0
	j=0
	reduce_lines=reduce.readlines()
	raw_lines=raw.readlines()
	while i<len(reduce_lines) and j<len(raw_lines):
		# if j%1000==0:
		# 	print('dealwith {} line'.format(j))
		if reduce_lines[i][0]==raw_lines[j][0]:
			output.write(reduce_lines[i])
		else:
			if reduce_lines[i]=='\n':
				i+=1
				continue
		i+=1
		j+=1

	print((i,j,i-j))


def get_f1score(result_file, target_file):
    '''
    result_file：get_result_file生成的文件
    target_file:格式同 result_file
    '''
    result = open(result_file, 'r', encoding='utf-8')
    target = open(target_file, 'r', encoding='utf-8')
    r_lines = result.readlines()
    t_lines = target.readlines()
    total_tags = 0  #target样本的字段数
    correct_tags = 0  #result中抽取出的正确字段数
    total_tab_tags = 0  #result中抽取出的字段数
    for r_line, t_line in zip(r_lines, t_lines):
        r_lis = r_line.split('  ') #每段标记两个空格隔开 如：北京/a  是个美丽的城市，鲁迅/b  曾经去过。
        t_lis = t_line.split('  ')
        for r_tag, t_tag in zip(r_lis, t_lis):
            if t_tag[-1] in ['a', 'b', 'c']:
                total_tags += 1
            if r_tag[-1] in ['a', 'b', 'c']:
                total_tab_tags += 1
                if r_tag[-1] == t_tag[-1] and len(r_tag) == len(t_tag):
                    correct_tags += 1
    recall = round(correct_tags / total_tags, 4)
    precise = round(correct_tags / total_tab_tags, 4)
    f1score = round(2 * recall * precise / (recall + precise), 4)
    result.close()
    target.close()
    return f1score

# recover_reduce_sentence_length('../../dg_result.txt', '../../dg_NERdata/train_raw.txt', '../../dg_rc_result.txt')
# BIO2line_file('../../dg_rc_result.txt','../../dg_NERdata/result_file.txt')
# f1score=get_f1score(result_file='../../dg_NERdata/result_file.txt', target_file='../../dg_NERdata/train_t_8.txt')
# print('df_fiscore: {}'.format(f1score))
