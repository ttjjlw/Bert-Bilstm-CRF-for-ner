#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行 BERT NER Server
#@Time    : 2019/1/26 21:00
# @Author  : MaCan (ma_cancan@163.com)
# @File    : run.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def start_server():
    from bert_base.server import BertServer
    from bert_base.server.helper import get_run_args

    args = get_run_args()
    print(args)
    server = BertServer(args)
    server.start()
    server.join()


def train_ner():
    import os
    from bert_base.train.train_helper import get_args_parser
    from bert_base.train.bert_lstm_ner import train
    bert_path = r'D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl\chinese_L-12_H-768_A-12\MSRA'
    root_path = r'D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl'
    args = get_args_parser()
    args.clean = True
    args.max_seq_length=128
    args.do_train=True
    args.output_dir=os.path.join(root_path, 'output')
    args.num_train_epochs=30
    args.learning_rate=1e-4
    args.warmup_proportion=0.1
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    train(args=args)


if __name__ == '__main__':
    """
    如果想训练，那么直接 指定参数跑，如果想启动服务，那么注释掉train,打开server即可
    """
    train_ner()
    # start_server()
