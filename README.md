# Bert-Bilstm-CRF-for-ner
重要：把TJL_function.py文件放入D:\Python_\Lib\site-packages（这是我的python安装的目录下的site_packages路径）下
===========
## 一、准备bert模型
>> 2019达观信息抽取比赛，采用的是脱敏的文档，因此需要预训练bert模型（而无法采用google预训练的bert 模型）<br> 
我预训练bert模型：链接：https://pan.baidu.com/s/1N627LqHnyKPSMbeBuONUfQ 提取码：jfle <br> 
注意：我的预训练bert模型是only mask LM而没训练上下句（但是实验证明，效果都差不多），only mask LM 最后的准确率在86%左右 <br> 
把预训练的bert模型、vocab.txt和bert_config.json在该目录下：BERT-BiLSTM-CRF-NER-tjl\chinese_L-12_H-768_A-12\DaGuan <br> 

### 如果不是针对DaGuan数据集 切换目录即可：如 把预训练的bert模型、vocab.txt和bert_config.json在该目录下：<br>
BERT-BiLSTM-CRF-NER-tjl\chinese_L-12_H-768_A-12\BERT

`clone 本项目 后其实可以跳过准备数据过程，因为数据也已上传。直接可进入第三步，设置相关超参数即可跑通。`

## 二、准备数据
>> BERT-BiLSTM-CRF-NER-tjl\dg_NERdata 在该目录下放入以下数据（文件名称保持一致，不能任意起名）：<br> 
train.txt  内容为BIO格式，用作训练集 <br> 
train_raw.txt  因为要输入固定长度的句子，我为了保证句子长度不大于128（128是超参数）我进行的操作是：假如原本句子长度为160，那么依次取前128  个词，再判断129词的tags是否是非O的tags,如果是继续取，再判断第130的词，如果第130个词的tags是O，则原本160的长度分成了129和41两                  个长度的句子（这个功能的实现见TJl_function.py中的reduce_sentence_length函数）。因此，train_raw.txt是未缩减前的，train.txt是                  由train_raw.txt变化来得。<br> 
test.txt  内容格式同train.txt 但也可不含标签。用于生成预测结果 <br>
test_raw.txt 同 train_raw含义 <br>
dev.txt 在train过程输出验证结果的信息 <br>

### 如果不是针对DaGuan数据集 切换目录即可：如把相应数据放置该目录下：
D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl\NERdata  <br>
train.txt 训练集 内容为BIO格式  <br>
test.txt 预测集 同上（可不含标签）  <br>
dev.txt 验证集  同上  <br>
打开BERT-BiLSTM-CRF-NER-tjl\bert_base\train\bert_lstm_ner.py文件 ctrl+f 搜索（如果不是针对daGuan比赛这段以下这段可以注释）注释指定内容，这样就不需要准备train_raw及test_raw.txt文件  <br>
train_v_8.txt 可不用管，放在这里不动即可，其实就是Daguan给的最原始的样本，用于计算test生成的文件的f1score <br>

## 三、运行模型
>> 1、准备模型（按以上介绍放入指定位置）  <br>
2、准备数据（按以上介绍放入指定位置） <br>
3、BERT-BiLSTM-CRF-NER-tjl\bert_base\train\train_helper.py 进入该目录设置默认参数  <br>
```
def get_args_parser():
    from .bert_lstm_ner import __version__
    parser = argparse.ArgumentParser()
    if os.name == 'nt':#windows 系统
        bert_path = r'D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl\chinese_L-12_H-768_A-12\DaGuan' #模型准备时的放置位置
        root_path = r'D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl'#BERT-BiLSTM-CRF-NER-tjl放置位置
```
4、运行BERT-BiLSTM-CRF-NER-tjl\run.py文件  <br>
5、输出结果如下：  <br>
```
INFO:tensorflow:Saving checkpoints for 0 into D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-tjl\output\model.ckpt.
INFO:tensorflow:loss = 194.51651, step = 0
INFO:tensorflow:global_steps = 0, loss = 194.51651
INFO:tensorflow:global_step/sec: 0.612548
INFO:tensorflow:loss = 43.69185, step = 100 (163.253 sec)
INFO:tensorflow:global_step/sec: 0.639746
```
## 模型运行过程解释
模型训练过程时会输出loss,每隔save_checkpoints_steps 会输出验证集结果 acc(准确率)，f1(以O为负，其他tags为正计算的)  <br>
max_steps_without_decrease（increase） f1没上升模型训练会早停 <br>
模型每隔save_checkpoints_steps时会自动保存，重新训练时，会自动检查是否有训练好的模型，有的话会载入接着训练 <br>
学习率会随steps增加而减少。 <br>
