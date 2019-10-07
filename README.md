# Bert-Bilstm-CRF-for-ner
## 一、准备bert模型
    2019达观信息抽取比赛，采用的是脱敏的文档，因此需要预训练bert模型（而无法采用google预训练的bert 模型）<br> 
我预训练bert模型：链接：https://pan.baidu.com/s/1N627LqHnyKPSMbeBuONUfQ 提取码：jfle <br> 
    注意：我的预训练bert模型是only mask LM而没训练上下句（但是实验证明，效果都差不多），only mask LM 最后的准确率在86%左右 <br> 
把预训练的bert模型、vocab.txt和bert_config.json在该目录下：BERT-BiLSTM-CRF-NER-tjl\chinese_L-12_H-768_A-12\DaGuan <br> 

## 二、准备数据
BERT-BiLSTM-CRF-NER-tjl\dg_NERdata 在该目录下放入以下数据（文件名称保持一致，不能任意起名）：<br> 
train.txt  ###内容为BIO格式，用作训练集 <br> 
train_raw.txt   ###因为要输入固定长度的句子，我为了保证句子长度不大于128（128是超参数）我进行的操作是：假如原本句子长度为160，那么依次取前128                  个词，再判断129词的tags是否是非O的tags,如果是继续取，再判断第130的词，如果第130个词的tags是O，则原本160的长度分成了129和41两                  个长度的句子（这个功能的实现见TJl_function.py中的reduce_sentence_length函数）。因此，train_raw.txt是未缩减前的，train.txt是                   由train_raw.txt变化来得。<br> 
