
import torch
import os
import json
from rich import print
from transformers import AutoTokenizer
import numpy as np
import re


# 提取生成的预测意图
generated_predictions_ir = os.path.join('../Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam12_WebQSP_Freebase_NQ_test/generated_predictions.jsonl')
generated_predictions_sexpr = os.path.join('../Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam8_WebQSP_Freebase_NQ_test/generated_predictions.jsonl')
WebQSP_train_file_path = os.path.join('../data/WebQSP/generation/merged', 'WebQSP_train.json')
WebQSP_test_file_path = os.path.join('../data/WebQSP/generation/merged', 'WebQSP_test.json')


# WebQSP_ir_predict是ir预测意图列表，WebQSP_ir_label是ir意图标签，WebQSP_ir_question是问题，WebQSP_sexpr_predict是sexpr预测列表
WebQSP_ir_predict =[]
WebQSP_ir_label =[]
WebQSP_ir_question =[]
WebQSP_sexpr_predict = []
# WebQSP_train是训练数据集，WebQSP_test是测试集数据
WebQSP_train= []
WebQSP_test = []
with open(generated_predictions_ir, 'r', encoding="utf-8") as f:
    for line in f:
        question = json.loads(line)['question']
        predict = json.loads(line)['predict']
        label = json.loads(line)['label']
        WebQSP_ir_predict.append(predict)
        WebQSP_ir_label.append(label)
        WebQSP_ir_question.append(question)
with open(generated_predictions_sexpr, 'r', encoding="utf-8") as f:
    for line in f:
        predict = json.loads(line)['predict']
        WebQSP_sexpr_predict.append(predict)
with open(WebQSP_train_file_path, 'r') as file:
    WebQSP_train = json.load(file)
with open(WebQSP_test_file_path, 'r') as file:
    WebQSP_test = json.load(file)


# WebQSP_sexpr_intent是sexpr预测意图列表
WebQSP_sexpr_intent = []
for item in WebQSP_sexpr_predict:
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
    WebQSP_sexpr_intent.append(item_list)
# WebQSP_intent_setlist是intent意图库，WebQSP_intent_label是测试集意图标签列表
WebQSP_intent_list = []
WebQSP_intent_label = []
for item in WebQSP_train:
    normed_sexpr = item['normed_sexpr']
    pattern = r'\[.*?]'# 定义正则表达式模式
    matches = re.findall(pattern, normed_sexpr)# 使用正则表达式提取匹配的字符串
    # print('2',matches)
    output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
    # print('3',output_list)
    output = ''.join(output_list)
    # print('4',output)
    WebQSP_intent_list.append(output.strip())
WebQSP_intent_set = set(WebQSP_intent_list)
WebQSP_intent_setlist = list(WebQSP_intent_set)
for item in WebQSP_test:
    normed_sexpr = item['normed_sexpr']
    pattern = r'\[.*?]'# 定义正则表达式模式
    matches = re.findall(pattern, normed_sexpr)# 使用正则表达式提取匹配的字符串
    # print('2',matches)
    output_list = list(f"{match.strip()}" for match in matches if match.count(',') >= 2)# 将匹配的字符串存储到列表中
    # print('3',output_list)
    output = ''.join(output_list)
    # print('4',output)
    WebQSP_intent_label.append(output.strip())
WebQSP_intent_list_all = WebQSP_intent_list + WebQSP_intent_label
WebQSP_intent_set_all = set(WebQSP_intent_list_all)
WebQSP_intent_setlist_all = list(WebQSP_intent_set_all)


# WebQSP_ir_qp是问题+ir预测意图列表
WebQSP_ir_qp = []
WebQSP_ir_qp8 = []
for i in range(len(WebQSP_ir_predict)):
    rows = WebQSP_ir_predict[i]
    p8 = ''.join(WebQSP_ir_predict[i])

    b = [ WebQSP_ir_question[i] + str(j) for j in rows]
    b8 = [ WebQSP_ir_question[i] + p8]
    # print(b)
    # print(b8)
    WebQSP_ir_qp.append(b)
    WebQSP_ir_qp8.append(b8)
print('len(WebQSP_ir_qp)', len(WebQSP_ir_qp))
print('len(WebQSP_ir_question)', len(WebQSP_ir_question))
print('len(WebQSP_ir_predict)', len(WebQSP_ir_predict))
print('len(WebQSP_intent_setlist)', len(WebQSP_intent_setlist))


device = "cpu"
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/reward_model/sentiment_analysis_WebQSP/model_best/')
model = torch.load('./checkpoints/reward_model/sentiment_analysis_WebQSP/model_best/model.pt')
model.to(device).eval()


# WebQSP_ir_sort_id是ir预测意图经过排序后的id列表
WebQSP_ir_sort_id = []
WebQSP_ir_sort_value = []
json_id_path = 'WebQSP_ir_sort_id.json'
json_value_path = 'WebQSP_ir_sort_value.json'
if not os.path.exists(json_id_path):
    for i in range(len(WebQSP_ir_qp)):
        inputs = tokenizer(
            WebQSP_ir_qp[i],
            max_length=1024,
            padding='max_length',
            return_tensors='pt'
        )
        r = model(**inputs)
        xx = r.tolist()
        value_list = [item for ii in xx for item in ii]
        value_list_sort = list(value_list)
        value_list_sort.sort(reverse=True)
        value_list1 = [-10 * x for x in value_list]
        # print(value_list_sort)
        sorted_id = np.argsort(value_list1)
        sorted_id = sorted_id.tolist()
        # print(i)
        sorted_id_list = []
        for j in sorted_id:
            sorted_id_list.append(j)
        print('问题%s的元素索引序列从大到小是：%s、%s' % (i, sorted_id_list, value_list_sort))
        WebQSP_ir_sort_id.append(sorted_id_list)
        WebQSP_ir_sort_value.append(value_list_sort)
    json_id = json.dumps(WebQSP_ir_sort_id)
    json_value = json.dumps(WebQSP_ir_sort_value)
    with open(json_id_path, "w") as file:
        file.write(json_id)
    # with open(json_value_path, "w") as file:
    #     file.write(json_value)
else:
    with open(json_id_path) as json_file:
        WebQSP_ir_sort_id = json.load(json_file)
    # with open(json_value_path) as json_file:
    #     WebQSP_ir_sort_value = json.load(json_file)


# WebQSP_ir_sort_value8 = []
# json_value8_path = 'WebQSP_ir_sort_value8.json'
# if not os.path.exists(json_value8_path):
#     for i in range(len(WebQSP_ir_qp8)):
#         inputs = tokenizer(
#             WebQSP_ir_qp8[i],
#             max_length=1024,
#             padding='max_length',
#             return_tensors='pt'
#         )
#         r = model(**inputs)
#         xx = r.tolist()
#         value_list = [item for ii in xx for item in ii]
#         value_list_sort = list(value_list)
#         value_list_sort.sort(reverse=True)
#         print('问题%s的元素索引序列从大到小是：%s' % (i, value_list_sort))
#
#         WebQSP_ir_sort_value8.append(value_list_sort)
#
#     json_value = json.dumps(WebQSP_ir_sort_value8)
#     with open(json_value8_path, "w") as file:
#         file.write(json_value)
# else:
#     with open(json_value8_path) as json_file:
#         WebQSP_ir_sort_value8 = json.load(json_file)


# WebQSP_ir_predict_sort是排序后的ir意图预测列表
WebQSP_ir_predict_sort = []
for i in range(len(WebQSP_ir_predict)):
    # if WebQSP_ir_predict[i][0] not in WebQSP_intent_setlist_all:#如果ir预测意图0不在intent意图库
    #     list1 = []
    #     for j in range(len(WebQSP_ir_sort_id[i])):
    #         list1.append(WebQSP_ir_predict[i][WebQSP_ir_sort_id[i][j]])
    #     WebQSP_ir_predict_sort.append(list1)
    #
    # elif WebQSP_ir_predict[i][0] not in WebQSP_sexpr_intent[i]:  # 如果ir预测意图0不在sexpr预测意图列表
    #     list1 = []
    #     for j in range(len(WebQSP_ir_sort_id[i])):
    #         list1.append(WebQSP_ir_predict[i][WebQSP_ir_sort_id[i][j]])
    #     WebQSP_ir_predict_sort.append(list1)
    # else:
    #     WebQSP_ir_predict_sort.append(WebQSP_ir_predict[i])

    if WebQSP_ir_predict[i][0] not in WebQSP_intent_setlist_all:#如果ir预测意图0不在intent意图库
        if len(WebQSP_ir_predict[i]) > 1:
            WebQSP_ir_predict[i][0] = WebQSP_ir_predict[i][1]
WebQSP_ir_predict_sort = WebQSP_ir_predict
print('len(WebQSP_ir_predict_sort)排序后的预测答案', len(WebQSP_ir_predict_sort))


# WebQSP_ir_predict_sort_export是导出的数据集合
WebQSP_ir_predict_sort_export = []
for i in range(len(WebQSP_ir_predict_sort)):
    WebQSP_ir_predict_sort_export.append({'question':WebQSP_ir_question[i],'label':WebQSP_ir_label[i],'predict':WebQSP_ir_predict_sort[i]})
output_dir = os.path.join('logs/reward_model/sentiment_analysis_WebQSP/generated_predictions_WebQSP.jsonl')
with open(output_dir, 'w') as f:
    for item in WebQSP_ir_predict_sort_export:
        json_string = json.dumps(item)
        f.write(json_string + '\n')
print('logs/reward_model/sentiment_analysis_WebQSP/generated_predictions_WebQSP.jsonl')




