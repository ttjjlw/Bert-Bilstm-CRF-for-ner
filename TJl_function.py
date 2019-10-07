#ner获取f1score
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
def compare_dg_result_file(output_dir=r'C:\Users\yxc\Desktop\Daguang'):
	'''
	output_dir:多个dg_result_file 存放的位置
	'''
	import os
	res_lis=os.listdir(output_dir)
	lis=[]
	for res01 in res_lis:
		result_file01=os.path.join(output_dir,res01)
		for res02 in res_lis:
			result_file02=os.path.join(output_dir,res02)
			if result_file01!=result_file02:
				f1score=get_f1score(result_file01,result_file02)
				print(f1score)
				if f1score<0.88:
					continue
				else:
					lis.append([res01,res02,f1score])
	print(lis)
#一行一句的文本文件，如：北京/LOC  是个美丽的城市，/o  鲁迅/PER  曾经去过。/o	
					#或：北_京/LOC  是_个_美_丽_的_城_市_，/o  鲁_迅_/PER  曾_经_去_过。/o
#变成BIO文件
def get_dg_train(train_dir,dg_train_dir):
    '''
    :param train_dir: 如：北京/LOC  是个美丽的城市，/o  鲁迅/PER  曾经去过。/o	
					#或：北_京/LOC  是_个_美_丽_的_城_市_，/o  鲁_迅_/PER  曾_经_去_过。/o
    :param dg_train_dir: 每行一个字 一个tag，字和tag之间用\t或空格隔开。每个句子以一个换行符隔开
    :return: 
    '''
	#有调整待检验
    with codecs.open(train_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            tags = []
            samples = line.strip().split('  ')
            for sample in samples:
                if len(sample.split('_'))>1:
                    sample=sample.split('/')
                    sample_list = sample[0].split('_')
                    tag = sample[-1]
                    features.extend(sample_list)
                else:
                    sample=sample.split('/')
                    sample_list = list(sample[0])
                    tag = sample[-1]
                    features.extend(sample_list)
                tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(
                    ['B-' + tag] + ['I-' + tag] * (len(sample_list) - 1))
            results.append(dict({'features': features, 'tags': tags}))
        # [{'features': ['7212', '17592', '21182', '8487', '8217', '14790', '19215', '4216', '17186', '6036',
        # '18097', '8197', '11743', '18102', '5797', '6102', '15111', '2819', '10925', '15274'],
        # 'tags': ['B-c', 'I-c', 'I-c', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        # 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}]
        train_write_list = []
        with codecs.open(dg_train_dir, 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['tags'])):
                    train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
                train_write_list.append('\n')
            f_out.writelines(train_write_list)
#一行一句的文本文件，如：北_京_是_个_美_丽_的_城_市_，_鲁_迅_曾_经_去_过_。 没有标签
#变成一字一行的文件
def get_dg_test(test_dir,dg_test_dir,separation=1):
    '''
    :param test_dir:
    :param dg_test_dir:
    :return: 获得dg_test与通过get_dg_train获得dg_train不同，每个句子之间隔两行，而且没有标签
    :separation: 1表示每句隔1个换行符，2表示每句隔2个换行符（针对crf做ner时获取test数据）
    '''
    with codecs.open(test_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            if separation==1:
                line=line.strip()
            sample_list = line.split('_')
            features.extend(sample_list)
            results.append(dict({'features': features}))
        test_write_list = []
        with codecs.open(dg_test_dir, 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['features'])):
                    test_write_list.append(result['features'][i] + '\n')
                test_write_list.append('\n')
            f_out.writelines(test_write_list)

#BIO文件变成，一行一句的文本文件 如：北京/a  是个美丽的城市，/o  鲁迅/b  曾经去过。/o
def BIO_2_line_file(BIO_file,result_file):
    '''
    BIO_file:每行一个字 一个tag，字和tag之间用\t或空格隔开。每个句子以一个换行符隔开
    '''
    f_write = open(result_file, 'w', encoding='utf-8')
    with open(dg_file, 'r', encoding='utf-8') as f:
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
                    while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][-1])
                i = j
            f_write.write('  '.join(samples) + '\n')
    f_write.close()


#缩减句子长度
def reduce_sentence_length(raw_file_path,result_file_path,max_sequence_length=128,file_type='predict.txt'):
	'''
	raw_file_path:txt文件，每行一个字\ttag\n，每个句字以换行符隔开
	result_file_path:获得文本和raw_file_path相同，只是最长的句子不再超过max_sequence_length+mark。
	file_type: 若为预测文件则没有tag
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
						if file_type!='predict.txt'and i==max_sequence_length-1 and str_lis[max_sequence_length][-1]!='O':
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
#恢复缩短的句子
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
		if j%1000==0:
			print('dealwith {} line'.format(j))
		if reduce_lines[i][0]==raw_lines[j][0]:
			output.write(reduce_lines[i])
		else:
			if reduce_lines[i]=='\n':
				i+=1
				continue
		i+=1
		j+=1

	print((i,j,i-j))
	
#获取时间
def Gettime(p=''):
    import time
    t = time.asctime(time.localtime(time.time()))  # t=Thu Jul 25 09:53:23 2019
    t = t.split()[2:4]  # t=['25', '09:54:56']
    d = [t[0]] + t[1].split(':')[:-1]
    d = '_'.join(d)+" "+p
    return d  # d=25_09_54

#寻找分类阈值
def find_best_threshold(all_predictions,all_labels):

    '''
    针对二分类问题，寻找最佳的分类边界，在0到1之间
    展平所有的预测结果和对应的标记
    all_predictions 为0到1之间的实数
    :param all_predictions:
    :param all_labels:
    :return:
    '''
    import numpy as np
    from sklearn.metrics import f1_score
    all_predictions=np.ravel(all_predictions)
    all_labels=np.ravel(all_labels)
    #从0到1以0.01为间隔定义99个备选阈值，分别是从0.01-0.99之间
    thresholds=[i/100 for i in range(100)]
    all_f1s=[]
    for threshold in thresholds:
        #计算当前阈值的f1 score
        preds=(all_predictions>=threshold).astype("int")
        f1=f1_score(y_true=all_labels,y_pred=preds)
        all_f1s.append(f1)
        #找出可以使f1 score最大的阈值
        best_threshold=thresholds[int(np.argmax(np.array(all_f1s)))]
        print('best_threshold is {}'.format(best_threshold))
        print(all_f1s)
        return best_threshold
		
#计算BIO.txt 的F1socre
def BIO_F1score(predict,target=None):
    import os
    if type(predict)==list and type(target)==list:
        p_tag_num = 0
        t_tag_num = 0
        correct_tag_num = 0
        for p_tag,t_tag in zip(predict,target):
            if p_tag != 'O':
                p_tag_num += 1
            if t_tag != 'O':
                t_tag_num += 1
            if p_tag == t_tag and p_tag != 'O':
                correct_tag_num += 1
        precision = round(correct_tag_num / p_tag_num, 4)
        print('Precision: {}'.format(precision))
        recall = round(correct_tag_num / t_tag_num, 4)
        print('recall: {}'.format(recall))
        F1 = round(2 * precision * recall / (precision + recall), 4)
        return F1
    elif os.path.isfile(predict) and target:
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
#隔两行变成隔一行
def twoline_to_1line_separation(line2_file,line1_file):
    '''
    :param line2_file: 一行一行的文本文件，有些文本之间隔了两行换行符
    :param line1_file: 一行一行的文本文件，把隔两行的换行符变成了隔一行
    :return: 
    '''
    df=open(line1_file,'w',encoding='utf-8')
    with open (line2_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        k=0
        for line in lines:
            if line=='\n':
                k+=1
                if k==2:
                    df.write('\n')
                    k=0
                    continue
                else:
                    continue
            df.write(line)
    df.close()
#求列表中出现最多次数的那个元素
def most_list(lt):
    '''
    :param lt: 列表
    :return: 返回列表中出现最多次数的那个元素
    '''
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            most_str = i
            temp = lt.count(i)
    return most_str
#tags IOB转化成IOBES
def iob_iobes(tags):
    """
    IOB -> IOBES
    tags:只含BIO等tags的列表
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

#tags IOBES转化成IOB
def iobes_iob(tags):
    """
    IOBES -> IOB
    tags:<class 'list'>: ['B-c', 'I-c', 'E-c', 'O', 'O']
    return: 返回新的tags，原tags不变
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags