
import json
import os
import re
import sys
import json
import os
from similarities import BertSimilarity


# 定义JSON文件的路径
WebQSP_train_file_path = os.path.join('../../data/WebQSP/generation/merged', 'WebQSP_train.json')
# 打开并读取JSON文件
with open(WebQSP_train_file_path, 'r') as file:
    WebQSP_train_data = json.load(file)


# 创建空列表来存储WebQSP_train_question_list、WebQSP_train_relation_list
WebQSP_train_question_list = []
WebQSP_train_relation_list = []
for item in WebQSP_train_data:
    normed_sexpr = item["normed_sexpr"]
    question = item['question']
    Question = question + ' '
    # print('1',normed_sexpr)

    pattern = r'\[.*?]'# 定义正则表达式模式
    matches = re.findall(pattern, normed_sexpr)# 使用正则表达式提取匹配的字符串
    # print('2',matches)
    output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
    # print('3',output_list)
    output = ''.join(output_list)
    # print('4',output)

    if output:
        WebQSP_train_relation_list.append(output.strip())
        WebQSP_train_question_list.append(Question)


#对WebQSP_train_relation_database列表进行去重
WebQSP_set = set(WebQSP_train_relation_list)
WebQSP_train_relation_database_setlist = list(WebQSP_set)
print('len(WebQSP_train_question_list)', len(WebQSP_train_question_list))
print('len(WebQSP_train_relation_database)',len(WebQSP_train_relation_list))
print('len(WebQSP_train_answer_setlist)',len(WebQSP_train_relation_database_setlist))


#根据相似度函数构建关系排序数据集
def sentences_similarity(sentences):
    model_file_path = os.path.join('shibing624/text2vec-base-multilingual')
    model = BertSimilarity(model_name_or_path= model_file_path )

    corpus = WebQSP_train_relation_database_setlist

    model.add_corpus(corpus)
    model.save_corpus_embeddings('WebQSP_en_corpus_emb.jsonl')
    # res = model.most_similar(queries=sentences, topn=10)
    # print('res1', res)
    del model
    model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual")
    model.load_corpus_embeddings('WebQSP_en_corpus_emb.jsonl')
    res = model.most_similar(queries=sentences, topn=3)
    # print('res2', res)
    sentences_sim_list = []
    for q_id, c in res.items():
        # print('query:', q_id, sentences[q_id])
        # print("search top 10:")
        sentences_sim = ''
        id = 0
        # for corpus_id, s in reversed(c.items()):
        for corpus_id, s in c.items():
            # print(f'\t{model.corpus[corpus_id]}: {s:.4f}')
            if id == 0:
                sentences_sim = sentences_sim + model.corpus[corpus_id]
            else:
                sentences_sim = sentences_sim + '\t' + model.corpus[corpus_id]
            id = id + 1
        sentences_sim_list.append(sentences_sim)
    return sentences_sim_list


# 对训练集的每个关系进行相似度分析
train_sim_list = sentences_similarity(WebQSP_train_relation_list)
# print(train_sim_list)
k = 0
for i in range(len(WebQSP_train_relation_list)):
    if WebQSP_train_relation_list[i] in train_sim_list[i]:
        k = k + 1
print('标准答案在相似答案集中的数量k',k)


# #构建数据训练集，将问题和关系拼接
WebQSP_train = []
print(len(WebQSP_train_question_list))
print(len(train_sim_list))
for i in range(len(train_sim_list)):
    rows = train_sim_list[i].split("\t")
    b = [ WebQSP_train_question_list[i] + str(j) for j in rows]
    # print(b)
    result = "\t".join(b)
    # 输出结果
    # print(result)
    WebQSP_train.append(result)
# print(WebQSP_train)


# 将拼接好的数据导出至文件
tsv_train_file = open('../data/reward_datasets/sentiment_analysis_WebQSP/train.tsv','w')
tsv_test_file = open('../data/reward_datasets/sentiment_analysis_WebQSP/dev.tsv','w')
for i in range(len(WebQSP_train)*4//5) :
    tsv_train_file.write(WebQSP_train[i] + '\n')
tsv_train_file.close()
for i in range(len(WebQSP_train)*4//5, len(WebQSP_train)) :
    tsv_test_file.write(WebQSP_train[i] + '\n')
tsv_test_file.close()
print('../data/reward_datasets/sentiment_analysis_WebQSP/train.tsv')
print('../data/reward_datasets/sentiment_analysis_WebQSP/test.tsv')

