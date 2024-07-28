
import torch
import os
import json
from rich import print
from transformers import AutoTokenizer
import numpy as np
import re


# 提取生成的预测关系
generated_predictions_ir = os.path.join('../Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/generated_predictions.jsonl')
generated_predictions_sexpr = os.path.join('../Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/generated_predictions.jsonl')
CWQ_train_file_path = os.path.join('../data/CWQ/generation/merged', 'CWQ_train.json')
CWQ_test_file_path = os.path.join('../data/CWQ/generation/merged', 'CWQ_test.json')



# CWQ_ir_predict是ir预测意图列表，CWQ_ir_label是ir意图标签，CWQ_ir_question是问题，CWQ_sexpr_predict是sexpr预测列表
CWQ_ir_predict =[]
CWQ_ir_label =[]
CWQ_ir_question =[]
CWQ_sexpr_predict = []
# CWQ_train是训练数据集，CWQ_test是测试集数据
CWQ_train= []
CWQ_test = []
with open(generated_predictions_ir, 'r', encoding="utf-8") as f:
    for line in f:
        question = json.loads(line)['question']
        predict = json.loads(line)['predict']
        label = json.loads(line)['label']
        CWQ_ir_predict.append(predict)
        CWQ_ir_label.append(label)
        CWQ_ir_question.append(question)
with open(generated_predictions_sexpr, 'r', encoding="utf-8") as f:
    for line in f:
        predict = json.loads(line)['predict']
        CWQ_sexpr_predict.append(predict)
with open(CWQ_train_file_path, 'r') as file:
    CWQ_train = json.load(file)
with open(CWQ_test_file_path, 'r') as file:
    CWQ_test = json.load(file)


# CWQ_sexpr_intent是sexpr预测意图列表
CWQ_sexpr_intent = []
for item in CWQ_sexpr_predict:
    item_list = []
    for i in range(len(item)):
        pattern = r'\[.*?]'# 定义正则表达式模式
        matches = re.findall(pattern, item[i])# 使用正则表达式提取匹配的字符串
        # print('2',matches)
        output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
        # print('3',output_list)
        output = ''.join(output_list)
        # print('4',output)
        item_list.append(output.strip())
    CWQ_sexpr_intent.append(item_list)
# CWQ_intent_setlist是intent意图库，CWQ_intent_label是测试集意图标签列表
CWQ_intent_list = []
CWQ_intent_label = []
for item in CWQ_train:
    normed_sexpr = item['normed_sexpr']
    pattern = r'\[.*?]'# 定义正则表达式模式
    matches = re.findall(pattern, normed_sexpr)# 使用正则表达式提取匹配的字符串
    # print('2',matches)
    output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
    # print('3',output_list)
    output = ''.join(output_list)
    # print('4',output)
    CWQ_intent_list.append(output.strip())
CWQ_intent_set = set(CWQ_intent_list)
CWQ_intent_setlist = list(CWQ_intent_set)
for item in CWQ_test:
    normed_sexpr = item['normed_sexpr']
    pattern = r'\[.*?]'# 定义正则表达式模式
    matches = re.findall(pattern, normed_sexpr)# 使用正则表达式提取匹配的字符串
    # print('2',matches)
    output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
    # print('3',output_list)
    output = ''.join(output_list)
    # print('4',output)
    CWQ_intent_label.append(output.strip())
CWQ_intent_list_all = CWQ_intent_list + CWQ_intent_label
CWQ_intent_set_all = set(CWQ_intent_list_all)
CWQ_intent_setlist_all = list(CWQ_intent_set_all)


# CWQ_ir_qp是问题+ir预测意图列表
CWQ_ir_qp = []
for i in range(len(CWQ_ir_predict)):
    rows = CWQ_ir_predict[i]
    b = [ CWQ_ir_question[i] + str(j) for j in rows]
    # print(b)
    CWQ_ir_qp.append(b)
print('len(CWQ_ir_qp)', len(CWQ_ir_qp))
print('len(CWQ_ir_question)', len(CWQ_ir_question))
print('len(CWQ_ir_predict)', len(CWQ_ir_predict))
print('len(CWQ_intent_setlist)', len(CWQ_intent_setlist))


device = "cpu"
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/reward_model/sentiment_analysis_CWQ/model_best/')
model = torch.load('./checkpoints/reward_model/sentiment_analysis_CWQ/model_best/model.pt')
model.to(device).eval()


# CWQ_ir_sort_id是ir预测意图经过排序后的id列表
CWQ_ir_sort_id = []
json_id_path = 'CWQ_ir_sort_id.json'
if not os.path.exists(json_id_path):
    for i in range(len(CWQ_ir_qp)):
        inputs = tokenizer(
            CWQ_ir_qp[i],
            max_length=1024,
            padding='max_length',
            return_tensors='pt'
        )
        r = model(**inputs)
        xx = r.tolist()
        value_list = [item for ii in xx for item in ii]
        value_list1 = [-10 * x for x in value_list]
        # print(value_list1)
        sorted_id = np.argsort(value_list1)
        sorted_id = sorted_id.tolist()
        # print(i)
        sorted_id_list = []
        for j in sorted_id:
            sorted_id_list.append(j)
        print('问题%s的元素索引序列从大到小是：%s' % (i, sorted_id_list))
        CWQ_ir_sort_id.append(sorted_id_list)
    json_data = json.dumps(CWQ_ir_sort_id)
    with open(json_id_path, "w") as file:
        file.write(json_data)
else:
    with open(json_id_path) as json_file:
        CWQ_ir_sort_id = json.load(json_file)


# CWQ_ir_predict_sort是排序后的ir意图预测列表
CWQ_ir_predict_sort = []
for i in range(len(CWQ_ir_predict)):
    # if CWQ_ir_predict[i][0] not in CWQ_intent_setlist_all:#如果ir预测意图0不在intent意图库
    #     list1 = []
    #     for j in range(len(CWQ_ir_sort_id[i])):
    #         list1.append(CWQ_ir_predict[i][CWQ_ir_sort_id[i][j]])
    #     CWQ_ir_predict_sort.append(list1)
    # 
    # elif CWQ_ir_predict[i][0] not in CWQ_sexpr_intent[i]:  # 如果ir预测意图0不在sexpr预测意图列表
    #     list1 = []
    #     for j in range(len(CWQ_ir_sort_id[i])):
    #         list1.append(CWQ_ir_predict[i][CWQ_ir_sort_id[i][j]])
    #     CWQ_ir_predict_sort.append(list1)
    # else:
    #     CWQ_ir_predict_sort.append(CWQ_ir_predict[i])

    if CWQ_ir_predict[i][0] not in CWQ_intent_setlist_all:  # 如果ir预测意图0不在intent意图库
        if len(CWQ_ir_predict[i]) > 1:
            CWQ_ir_predict[i][0] = CWQ_ir_predict[i][1]
CWQ_ir_predict_sort = CWQ_ir_predict
print('len(CWQ_ir_predict_sort)排序后的预测答案', len(CWQ_ir_predict_sort))


# CWQ_ir_predict_sort_export是导出的数据集合
CWQ_ir_predict_sort_export = []
for i in range(len(CWQ_ir_predict_sort)):
    CWQ_ir_predict_sort_export.append({'question':CWQ_ir_question[i],'label':CWQ_ir_label[i],'predict':CWQ_ir_predict_sort[i]})
output_dir = os.path.join('logs/reward_model/sentiment_analysis_CWQ/generated_predictions_CWQ.jsonl')
with open(output_dir, 'w') as f:
    for item in CWQ_ir_predict_sort_export:
        json_string = json.dumps(item)
        f.write(json_string + '\n')
print('logs/reward_model/sentiment_analysis_CWQ/generated_predictions_CWQ.jsonl')

