
# import os：导入 os 模块，用于与操作系统交互，例如处理文件和目录。
# import json：导入 json 模块，用于对 JSON 数据进行编码和解码。
# import argparse：导入 argparse 模块，用于解析命令行参数。
# from components.utils import load_json：从名为 components 的包或模块中的 utils 模块导入 load_json 函数。
#     这表明在名为 components 的文件夹或包中有一个名为 utils.py 的文件，其中包含一个名为 load_json 的函数。
# from tqdm import tqdm：导入 tqdm 模块，提供用于循环中任务进度可视化的进度条。在处理大型数据集或耗时操作时，这对于可视化任务的进度非常有用。
import os
import json
import argparse
from components.utils import load_json
from tqdm import tqdm


# 这段代码定义了两个函数：`load_data` 和 `_parse_args`。
# 1. `load_data(split, args)`: 这个函数用于加载数据。它接受两个参数，`split` 表示数据集的拆分（例如训练集、验证集、测试集等）
#     ，而 `args` 是通过命令行参数传递的配置信息。函数构建了数据文件的路径，加载该文件并返回数据字典。
#    - `data_file_name`: 构建数据文件名的字符串，使用 `format` 方法插入 `args.dataset_type` 和 `split`。
#    - 打印加载数据的消息。
#    - 使用之前提到的 `load_json` 函数加载 JSON 文件，返回数据字典。
def load_data(split, args):
    data_file_name = 'data/{}/generation/merged/{}_{}.json'.format(args.dataset_type,args.dataset_type,split)
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict

# 2. `_parse_args()`: 这个函数使用 `argparse` 解析命令行参数。它定义了一个命令行解析器，添加了一个 `--dataset_type` 的参数，然后解析命令行输入，
#     将结果存储在 `args` 中，并返回 `args`。
#    - `parser`: 创建一个 `argparse` 的命令行解析器。
#    - `parser.add_argument(...)`: 添加一个命令行参数，这里是 `--dataset_type`，默认为 "WebQSP"。
#    - `parser.parse_args()`: 解析命令行参数，将结果存储在 `args` 中。
#    - 返回解析后的参数对象 `args`。
# 这两个函数可以一起使用，首先调用 `_parse_args` 解析命令行参数，然后将返回的 `args` 对象传递给 `load_data` 函数，加载相应数据集的数据。
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="WebQSP", type=str, help="CWQ | WebQSP")
    args = parser.parse_args()
    return args

# 这段代码定义了一个名为 `prepare_dataloader` 的函数。下面是代码的解释：
# 1. **输入验证**：代码首先通过 `assert` 语句确保 `split` 参数的取值在指定的范围内，
#     即 ['train', 'test', 'dev', 'train_sample', 'dev_sample', 'test_sample']。这确保只使用有效的数据集拆分。
# 2. **加载数据**：调用 `load_data` 函数使用给定的 `args`（命令行参数）加载指定拆分的数据。
# 3. **过滤示例**：根据拆分的类型，它过滤示例。对于 'train' 和 'dev' 拆分，仅保留 'sexpr' 字段不等于 "null" 的示例。对于其他拆分，包括所有示例。
# 4. **打印数据集长度**：打印原始数据集和过滤后数据集的长度。
# 5. **数据转换**：将数据转换为特定格式。对于每个示例，构建一个包含问题的输入字符串，并用 "Question: { ... }" 包装。同时提取 'normed_sexpr' 字段作为输出。
# 6. **保存到 JSON**：创建一个包含转换后数据的字典列表（`json_data`），然后将这些数据保存到一个具有特定格式的 JSON 文件。
# 7. **创建输出目录**：检查保存输出文件的目录是否存在，如果不存在则创建。
# 8. **写入到输出文件**：将准备好的数据写入到指定输出目录下的一个 JSON 文件中。
# 该函数的作用是为特定拆分准备数据，过滤示例，将其转换为期望的格式，然后将结果数据保存到一个 JSON 文件中。
def prepare_dataloader(args,split):
    assert split in ['train','test','dev','train_sample','dev_sample','test_sample']

    data = load_data(split, args)
    print(f'Origin {split} dataset len: {len(data)}')
    assert type(data)==list
    if 'train' in split or 'dev' in split:
        # for train and dev, filter the examples without sexpr
        examples = []
        for x in data:
            if x['sexpr'].lower()!="null":
                examples.append(x)                
    else:
        examples = [x for x in data]
    print(f'Real {split} dataset len: {len(examples)}')

    json_data=[]
    instruction='Generate a Logical Form query that retrieves the information corresponding to the given question. \n'
    # instruction='1.Generate dependency tree and syntax tree of the problem according to the problem.2.Generate a Logical Form query that retrieves the information corresponding to the given question by combining the problem, dependency tree and syntax tree. 3.Directly output the Logical Form query.\n'

    for cnt, item in tqdm(enumerate(examples)):
        # if item['normed_sexpr'] != "null":
        question=item['question']
        input = 'Question: { '+question+' }'
        output = item['normed_sexpr']
        json_data.append({"instruction":instruction,"input":input,"output":output,"history":[]})
               
    
    output_dir = 'LLMs/data_sexpr/{}_Freebase_NQ_{}/examples.json'.format(args.dataset_type, split)

    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(output_dir))   

    with open(output_dir, "w", encoding="utf-8") as file:
        json.dump(json_data, file)    
    
# 这段代码包含一个主程序的入口点，通过 `if __name__=='__main__':` 判断是否是直接运行的脚本而不是被导入为模块。如果是主程序，它会执行以下步骤：
# 1. **解析命令行参数**：调用 `_parse_args()` 函数解析命令行参数，并将结果存储在 `args` 变量中。
# 2. **打印参数**：输出解析后的参数，以便查看脚本的配置。
# 3. **准备训练集和测试集的数据**：调用 `prepare_dataloader` 函数两次，分别为训练集和测试集准备数据。使用之前解析的命令行参数 `args`。
# 4. **打印完成消息**：输出 'Finished'，表示脚本执行完成。
# 这个脚本的作用是根据命令行参数准备训练集和测试集的数据，然后输出一条完成消息。
if __name__=='__main__':
    
    args = _parse_args()
    print(args)
    prepare_dataloader(args,'train')
    prepare_dataloader(args, 'test')
    print('Finished')

