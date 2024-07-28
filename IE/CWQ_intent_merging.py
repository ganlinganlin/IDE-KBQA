

import os
import json
import re


CWQ_ire_path = os.path.join('./logs/reward_model/sentiment_analysis_CWQ/generated_predictions_CWQ.jsonl')
CWQ_sexpr_file_path = os.path.join('../Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/', 'beam_test_top_k_predictions.json')
CWQ_train_file_path = os.path.join('../data/CWQ/generation/merged', 'CWQ_train.json')
CWQ_test_file_path = os.path.join('../data/CWQ/generation/merged', 'CWQ_test.json')


# CWQ_ire_data是CWQ的ire预测列表（包括预测和答案），CWQ_ire_intent1是intent1预测意图1列表，CWQ_ire_intent8是intent8预测意图8列表
CWQ_ire_data = []
CWQ_ire_intent1 = []
CWQ_ire_intent8 = []
# CWQ_train是训练数据集，CWQ_test是测试集数据
CWQ_train= []
CWQ_test = []
with open(CWQ_ire_path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        CWQ_ire_data.append(data)
for prediction in CWQ_ire_data:
    intent1 = prediction['predict'][0]
    intent8 = prediction['predict']
    CWQ_ire_intent1.append(intent1)
    CWQ_ire_intent8.append(intent8)


# CWQ_sexpr_data是CWQ的sexpr预测列表（包括预测和答案），CWQ_sexpr1是sexpr1预测列表，CWQ_sexpr8是sexpr8预测列表，CWQ_sexpr_label是sexpr的标签列表
with open(CWQ_sexpr_file_path, 'r') as file:
    CWQ_sexpr_data = json.load(file)
CWQ_sexpr1 = []
CWQ_sexpr8 = []
CWQ_sexpr_label = []
for prediction in CWQ_sexpr_data:
    sexpr_s1 = prediction['predictions'][0]
    sexpr_s8 = prediction['predictions']
    gen_label = prediction['gen_label']
    CWQ_sexpr1.append(sexpr_s1)
    CWQ_sexpr8.append(sexpr_s8)
    CWQ_sexpr_label.append(gen_label)
with open(CWQ_train_file_path, 'r') as file:
    CWQ_train = json.load(file)
with open(CWQ_test_file_path, 'r') as file:
    CWQ_test = json.load(file)


# CWQ_sexpr_intent8是sexpr预测意图列表
CWQ_sexpr_intent8 = []
for item in CWQ_sexpr8:
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
    CWQ_sexpr_intent8.append(item_list)


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


# 替换sexpr中的intent，返回字符串sexpr_new
def replace(sexpr, intent):
    pattern_rp = r'\[.*?]'  # 定义正则表达式模式
    matches_sexpr = re.findall(pattern_rp, sexpr)
    # print('01意图合并开始：')
    # print('02sexpr中匹配出的意图', matches_sexpr)
    matches_intent = re.findall(pattern_rp, intent)
    # print('02intent中匹配出的意图', matches_intent)
    i_match = -1
    sexpr_new = sexpr
    len_matches_sexpr = 0
    for number in matches_sexpr:
        if number.count(',') >= 2:
            len_matches_sexpr = len_matches_sexpr +1
    # print('03sexpr中的意图数量和intent中的意图数量分别是',len_matches_sexpr,len(matches_intent))
    if len_matches_sexpr == len(matches_intent):
        for match in matches_sexpr:
            if match.count(',') >= 2:
                i_match = i_match + 1
                intent_replace = matches_intent[i_match]
                # print('04sexpr意图',i_match, match)
                # print('04intent意图',i_match,intent_replace)
                # print('04意图替换前',sexpr_new)

                sexpr_new = sexpr_new.replace(match, intent_replace,1)
                # print('04意图替换后',sexpr_new)
    else:
        sexpr_new = 'null'
    # print('05意图合并结束')
    return sexpr_new


# data1 = "( JOIN ( R [ people , person , place of birth ] ) [ loaaaan , ntrssy , la spoddken ][ lonsss , nddddtry , la spofffken ][111, 11][] )"
# data2 = '[ location , country , languages spoken ][ lon , ntry , la spoken ][ loc , coy , languagen ][]['
# r1 = replace(data1,data2)
# print(r1)


CWQ_sexpr_ire_predict = []
mm = 0
for i in range(len(CWQ_sexpr8)):
    mm = mm + 1
    # print('开始处理第%s个意图', mm)
    CWQ_sexpr_data = list(CWQ_sexpr8[i])
    

    if CWQ_sexpr_intent8[i][0] not in CWQ_intent_setlist_all:#如果sexpr预测意图0不在intent意图库
        CWQ_data_new_r8 = replace(CWQ_sexpr8[i][0], CWQ_ire_intent8[i][0])
        if CWQ_data_new_r8 != 'null':
            CWQ_sexpr_data[0] = CWQ_data_new_r8
    for j in range(1,10):
        if len(CWQ_sexpr_intent8[i]) > j:
            if CWQ_sexpr_intent8[i][j] not in CWQ_intent_setlist_all:  # 如果sexpr预测意图0不在intent意图库
                CWQ_data_new_r8 = replace(CWQ_sexpr8[i][j], CWQ_ire_intent8[i][0])
                if CWQ_data_new_r8 != 'null':
                    CWQ_sexpr_data[j] = CWQ_data_new_r8


    # if len(CWQ_sexpr_intent8[i]) > 2:
    #     if CWQ_sexpr_intent8[i][2] not in CWQ_intent_setlist_all:  # 如果sexpr预测意图0不在intent意图库
    #         CWQ_data_new_r8 = replace(CWQ_sexpr8[i][2], CWQ_ire_intent8[i][0])
    #         if CWQ_data_new_r8 != 'null':
    #             CWQ_sexpr_data[2] = CWQ_data_new_r8


    # if CWQ_sexpr_intent8[i][0] not in CWQ_intent_setlist_all:#如果sexpr预测意图0不在intent意图库
    #     CWQ_data_new_r8 = replace(CWQ_sexpr8[i][0], CWQ_ire_intent8[i][0])
    #     if CWQ_data_new_r8 != 'null':
    #         CWQ_sexpr_data[0] = CWQ_data_new_r8
    # if len(CWQ_sexpr_intent8[i]) >= 2:
    #     if CWQ_sexpr_intent8[i][1] not in CWQ_intent_setlist_all:  # 如果sexpr预测意图0不在intent意图库
    #         CWQ_data_new_r8 = replace(CWQ_sexpr8[i][1], CWQ_ire_intent8[i][0])
    #         if CWQ_data_new_r8 != 'null':
    #             CWQ_sexpr_data[1] = CWQ_data_new_r8
    CWQ_sexpr_ire_predict.append(CWQ_sexpr_data)
print('处理的总数量：', mm)


CWQ_ire_generated_predictions = []
for i in range(len(CWQ_sexpr_ire_predict)):
    CWQ_ire_generated_predictions.append({'question':'','label':CWQ_sexpr_label[i],'predict':CWQ_sexpr_ire_predict[i]})
output_dir = os.path.join('../Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/generated_predictions.jsonl')
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))
with open(output_dir, 'w') as f:
    for item in CWQ_ire_generated_predictions:
        json_string = json.dumps(item)
        f.write(json_string + '\n')
print('../Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/generated_predictions.jsonl')

