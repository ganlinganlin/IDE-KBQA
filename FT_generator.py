
# 你导入了 `os`、`argparse` 和 `json` 模块，以及一个名为 `dump_json` 的函数，
#     但是你的代码中没有显示 `components.utils` 模块，因此无法确定 `dump_json` 函数的具体实现。
import os
import argparse
import json
from components.utils import dump_json
import re

# 这段代码看起来是一个用于准备数据加载器（dataloader）的函数。函数的主要功能包括：
# 1. 通过 `args.data_file_name` 打开一个 JSON 文件。
# 2. 读取 JSON 文件中的每一行，将其解析为字典，并将所有字典组成的列表存储在变量 `data` 中。
# 3. 打印数据集的长度。
# 4. 返回包含所有数据的列表 `data`。
# 这个函数适用于读取包含 JSON 数据的文件，并将其转换为 Python 字典的列表。如果你有关于该函数的具体问题或需要进一步的解释，请随时提问。
def prepare_dataloader(args):
    print('Loading data from:',args.data_file_name)
    with open(args.data_file_name, 'r', encoding='utf-8') as f:
        # 读取每一行并转换为字典
        data = [json.loads(line) for line in f]
    print(f'Dataset len: {len(data)}')
    return data

# 这个函数 `_parse_args` 是一个用于解析命令行参数的函数。在这个函数中，使用了 Python 内置的 `argparse` 模块，它提供了一种方便的方式来处理命令行参数。
# 函数的主要步骤如下：
# 1. 创建一个 `ArgumentParser` 对象，它用于定义和解析命令行参数。
# 2. 使用 `add_argument` 方法向解析器添加命令行参数。在这个例子中，只添加了一个参数 `--data_file_name`，默认值为指定的文件路径。
# 3. 使用 `parser.parse_args()` 方法解析命令行参数，并将结果存储在一个命名空间对象中。
# 4. 返回包含解析结果的命名空间对象。
# 这个函数允许通过命令行指定 `--data_file_name` 参数，如果没有提供，默认为指定的文件路径。如果你有关于命令行参数解析或其他方面的疑问，请随时询问。
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_name',default='Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl')

    args = parser.parse_args()
    return args

# 这个`run_prediction`函数用于运行预测，并提供了一些统计信息。函数接收以下参数：
# - `args`: 包含程序运行参数的对象。
# - `dataloader`: 数据加载器，用于加载测试数据。
# - `output_dir`: 输出结果的目录。
# - `output_predictions`: 一个布尔值，指示是否输出预测结果。
# 函数执行的主要步骤包括：
# 1. 遍历数据加载器中的每个预测。
# 2. 将预测结果、生成的标签等信息添加到`output_list`中。
# 3. 统计正确匹配的例子数量、包含正确匹配的例子数量以及实际例子的总数。
# 4. 打印统计信息，包括总样本数、正确匹配数、正确匹配率、实际正确匹配率、包含正确匹配数、包含正确匹配率、实际包含正确匹配率等。
# 5. 如果指定了`output_predictions`为True，则将预测结果和统计信息输出到对应的文件中。
# 这个函数主要用于评估模型在测试集上的性能。如果你有任何关于这个函数的具体问题，或者想了解其中某一部分的详细信息，请告诉我。
def run_prediction(args,dataloader,output_dir,output_predictions=True):
    print()
    print('Start predicting ')
            
    ex_cnt = 0
    ex_cnt_logic = 0
    contains_ex_cnt = 0
    contains_ex_cnt_logic = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(dataloader):
        predictions = pred['predict']
        gen_label = pred['label']

        output_list.append({
            'predictions':predictions,
            'gen_label':gen_label,
        })

        # 整体hit命中率：
        if predictions[0].lower()==gen_label.lower():
            ex_cnt+=1
        # 整体hit8命中率：
        if any([x.lower()==gen_label.lower() for x in predictions]):
            contains_ex_cnt+=1

        if re.sub(r'\[.*?]', '', predictions[0].lower()) == re.sub(r'\[.*?]', '', gen_label.lower()):
            ex_cnt_logic += 1
        if any([re.sub(r'\[.*?]', '', x.lower()) == re.sub(r'\[.*?]', '', gen_label.lower()) for x in predictions]):
            contains_ex_cnt_logic += 1

        if gen_label.lower() != 'null':
            real_total += 1
    
    print(f"""total:{len(output_list)}, 
            整体hit@1命中率：
                    ex_cnt:{ex_cnt}, 
                    output_list:{len(output_list)}, 
                    real_total:{real_total}, 
                    ex_rate:{ex_cnt/len(output_list)}, 
                    real_ex_rate:{ex_cnt/real_total}, 
            整体hit@8命中率：
                    contains_ex_cnt:{contains_ex_cnt}, 
                    contains_ex_rate:{contains_ex_cnt/len(output_list)},
                    real_contains_ex_rate:{contains_ex_cnt/real_total},
            logic-hit@1命中率：
                    ex_cnt_logic:{ex_cnt_logic}, 
                    real_total:{real_total}, 
                    ex_cnt_logic_rate:{ex_cnt_logic/len(output_list)}, 
                    real_ex_cnt_logic_rate:{ex_cnt_logic/real_total}, 
            logic-hit@8命中率：
                    contains_ex_cnt_logic:{contains_ex_cnt_logic}, 
                    contains_ex_cnt_logic_rate:{contains_ex_cnt_logic/len(output_list)},
                    real_contains_ex_cnt_logic_rate:{contains_ex_cnt_logic/real_total},
                    """)

        
    if output_predictions:
        file_path = os.path.join(output_dir,f'beam_test_top_k_predictions.json')

        gen_statistics_file_path = os.path.join(output_dir,f'beam_test_gen_statistics.json')
        gen_statistics = {
            'total':len(output_list),
            '整体hit@1命中率：'
            'exmatch_num': ex_cnt,
            'output_list': len(output_list),
            'real_total': real_total,
            'exmatch_rate': ex_cnt/len(output_list),
            'real_exmatch_rate':ex_cnt/real_total,
            '整体hit@8命中率：'
            'contains_ex_num':contains_ex_cnt,
            'contains_ex_rate':contains_ex_cnt/len(output_list),
            'real_contains_ex_rate':contains_ex_cnt/real_total,
            '关系hit@1命中率：'
            'ex_cnt_logic': ex_cnt_logic,
            'ex_cnt_logic_rate': ex_cnt_logic / len(output_list),
            'real_ex_cnt_logic_rate': ex_cnt_logic / real_total,
            '关系hit@8命中率：'
            'contains_ex_cnt_logic': contains_ex_cnt_logic,
            'contains_ex_cnt_logic_rate': contains_ex_cnt_logic / len(output_list),
            'real_contains_ex_cnt_logic_rate': contains_ex_cnt_logic / real_total,
        }
        dump_json(output_list, file_path, indent=4)
        dump_json(gen_statistics, gen_statistics_file_path,indent=4)


# 这段代码主要用于运行预测。首先，它通过`_parse_args()`函数解析命令行参数，然后准备测试数据集的数据加载器`test_dataloader`。
#     接着，它调用`run_prediction`函数运行预测，将结果输出到指定的目录，并设置了输出预测结果的标志。最后，打印出 "Prediction Finished" 表示预测完成。
# 这段代码看起来是一个用于命令行运行的脚本，可能是用于对某个模型进行测试或推断。如果你有具体的问题或需要更详细的解释，请告诉我。
if __name__=='__main__':
    
    args = _parse_args()
    print(args)

    test_dataloader = prepare_dataloader(args)
    run_prediction(args, test_dataloader, output_dir=os.path.dirname(args.data_file_name), output_predictions=True)

    print('Prediction Finished')

