# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date
import os,pickle,codecs
import time
from bert_base.client import BertClient
# 指定服务器的IP
with BertClient(ip='127.0.0.1', ner_model_dir=r'D:\localE\code\daguang_extract\BERT-BiLSTM-CRF-NER-master\output', show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '以毛泽东同志为核心的中国共产党带领人民实现了民族解放，并在北京建立了新中国'
    rst = bc.encode([str, str])  #测试同时输入两个句子，多个输入同理
    print('rst:', rst)
    print(time.perf_counter() - start_t)

exit()

def BIO_F1score(predict,target=None):
    import os
    if os.path.isfile(predict) and target:
        p_file=open(predict,'r',encoding='utf-8')
        predict=p_file.readlines()
        t_file=open(target,'r',encoding='utf-8')
        target=t_file.readlines()
        p_tag_num=0
        t_tag_num=0
        correct_tag_num=0
        for p_line,t_line in zip(predict,target):
            if p_line == '\n' and t_line=='\n':
                continue
            elif p_line == '\n' and t_line!='\n':
                raise AttributeError('预测换行符和目标换行符不匹配')
            elif p_line != '\n' and t_line=='\n':
                raise AttributeError('预测换行符和目标换行符不匹配')
            p_tag=p_line.strip().split()[-1]
            t_tag=t_line.strip().split()[-1]
            if p_tag !='O':
                p_tag_num+=1
            if t_tag !='O':
                t_tag_num+=1
            if p_tag == t_tag and p_tag != 'O':
                correct_tag_num+=1
        precision=round(correct_tag_num/p_tag_num,4)
        recall=round(correct_tag_num/t_tag_num,4)
        F1=round(2*precision*recall/(precision+recall),4)
        p_file.close()
        t_file.close()
        return F1
    elif os.path.isfile(predict) and target==None:
        with open(predict,'r',encoding='utf-8') as p:
            predict=p.readlines()
        p_tag_num = 0
        t_tag_num = 0
        correct_tag_num = 0
        if len(predict[0].strip().split())!=3:
            raise AttributeError('输入的predict每行不是三个字符')
        for line in predict:
            if line=='\n':
                continue
            p_tag=line.strip().split()[-1]
            t_tag=line.strip().split()[1]
            if p_tag != 'O':
                p_tag_num += 1
            if t_tag != 'O':
                t_tag_num += 1
            if p_tag == t_tag and p_tag !='O':
                correct_tag_num += 1
        precision = round(correct_tag_num / p_tag_num, 4)
        print('Precision: {}'.format(precision))
        recall = round(correct_tag_num / t_tag_num, 4)
        print('recall: {}'.format(recall))
        F1 = round(2 * precision * recall / (precision + recall), 4)
        return F1
    else:
        raise AttributeError('输入的predict格式不对')



def reduce_sentence_length(raw_file_path,result_file_path,max_sequence_length):
	'''
	raw_file_path:txt文件，每行一个字，每个句字以换行符隔开
	result_file_path:获得文本和raw_file_path相同，只是最长的句子不再超过max_sequence_length。
	'''
	import math,os
	if  os.path.isfile(result_file_path):  #如果文件存在会报错
		print('{}文件存在'.format(result_file_path))
		raise NameError
	max_lis=[max_sequence_length-1]
	f1=open(result_file_path,'w',encoding='utf-8')
	with open(raw_file_path,'r',encoding='utf-8') as f :
		lines=f.read()
		lis_lines=lines.split('\n\n')
		if lis_lines[-1]=='':
			print('最后一行为换行符,去除最后一行')
			lis_lines.pop()
		print('样本共有:',len(lis_lines))
		j=0
		for line in lis_lines:
			lis_line=line.split('\n')
			if lis_line[-1]=='':
				lis_line.pop()
			length=len(lis_line)
			if length<=max_sequence_length:
				f1.write(line+'\n\n')

			else:
				max_lis.append(max_sequence_length)

				str_lis = line.split('\n')
				if str_lis[-1]=='':
					str_lis.pop()
				j=j+math.ceil(length/max_sequence_length)-1
				print((length,j))
				while length>max_sequence_length:
					mark = max_sequence_length
					for i in range(max_sequence_length):
						f1.write(str_lis[i]+'\n')
						if i==max_sequence_length-1 and str_lis[max_sequence_length][-1]!='O':
							while mark<len(str_lis) and str_lis[mark][-1]!='O':
								print(str_lis[mark])
								f1.write(str_lis[mark]+'\n')
								mark+=1
							max_lis.append(mark)
							if mark==length-1:
								j-=1
					f1.write('\n')
					str_lis=str_lis[mark:]
					length=len(str_lis)
				if len(str_lis):
					for i in range(len(str_lis)):
						f1.write(str_lis[i]+'\n')
					f1.write('\n')
	print('共增加了{}行'.format(j))
	print('最长句子长度{}'.format(max(max_lis)))
	f1.close()
reduce_sentence_length('dg_NERdata/dev_raw.txt','dg_NERdata/dev.txt',120)
exit()
print("===================================")
f_raw=open('dg_NERdata/test_raw.txt','r')
with open('dg_NERdata/test.txt','r') as f2:
    lines=f2.readlines()

    lines_raw=f_raw.readlines()
    result=[]
    i=1
    for raw,test in zip(lines_raw,lines):

        if raw==test:
            i+=1
            continue
        else:
            result.append((i,raw,test))
    print(len(result))
    print(result)
    exit()
    lines=lines.split('\n\n')
    print(lines[-1])
    print(len(lines))
    for line in lines :
        if len(line.split())>128:
            print(len(line.split()))





# python bert_lstm_ner.py^
#                   --task_name='NER'^
#                   --do_train=True^
#                   --do_eval=True   \
#                   --do_predict=True
#                   --data_dir=../chinese_L-12_H-768_A-12\MSRA  \
#                   --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt  \
#                   --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json \
#                   --init_checkpoint=../chinese_L-12_H-768_A-12/bert_model.ckpt.meta   \
#                   --max_seq_length=128   \
#                   --train_batch_size=32   \
#                   --learning_rate=2e-5   \
#                   --num_train_epochs=3.0   \
#                   --output_dir=../output/result_dir/