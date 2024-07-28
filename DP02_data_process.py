
# 这段代码导入了一系列 Python 模块和库，以下是对导入的主要模块的简要解释：
# 1. `collections.defaultdict`: 提供了一个默认值的字典，当访问不存在的键时，可以返回指定的默认值。
# 2. `random`: Python 内置的随机数生成模块，用于生成伪随机数。
# 3. `typing.Dict` 和 `typing.List`: 用于声明字典和列表的类型提示。
# 4. `components.utils`: 导入了自定义模块 `components.utils` 中的一些函数。这个模块可能包含了各种用于处理数据、生成查询等任务的工具函数。
# 5. `argparse`: 用于解析命令行参数的模块。
# 6. `tqdm`: 用于在循环中显示进度条的模块，提供了一个美观的进度显示。
# 7. `os`: 提供了与操作系统交互的功能，用于文件和目录的处理。
# 8. `torch`: PyTorch 深度学习框架的主要模块，提供张量（tensor）操作、梯度计算等功能。
# 9. `pandas`: 用于数据分析和处理的库，提供了数据结构和函数。
# 10. `executor.sparql_executor`: 导入了一个名为 `sparql_executor` 的自定义模块，其中包含了与 SPARQL 查询执行相关的函数。
# 这些模块和库在代码中的使用可能涉及到数据处理、文本生成、SPARQL 查询执行等多个方面。
from collections import defaultdict
import random
from typing import Dict, List
from components.utils import (
    _textualize_relation,
    load_json, 
    dump_json, 
    extract_mentioned_entities_from_sparql, 
    extract_mentioned_relations_from_sparql,
    vanilla_sexpr_linearization_method
)
import argparse
from tqdm import tqdm
import os
import torch
import pandas as pd
from executor.sparql_executor import (
    get_label_with_odbc,    
    get_types_with_odbc,
    get_out_relations_with_odbc,
    get_in_relations_with_odbc,
    get_entity_labels
)

# 这个函数 `_parse_args()` 创建了一个命令行参数解析器（parser），并定义了三个命令行参数：
# 1. `--action`: 操作的动作，默认值为 'merge_all'。用于指定程序应该执行的具体操作。
# 2. `--dataset`: 数据集的名称，默认值为 'WebQSP'。用于指定程序应该在哪个数据集上执行操作，可选值为 'CWQ' 或 'WebQSP'。
# 3. `--split`: 操作的数据集拆分，默认值为 'train'。用于指定在数据集的哪个拆分上执行操作，可选值为 'dev'、'test' 或 'train'。
# 这个函数最后调用 `parser.parse_args()` 来解析命令行参数，并返回一个包含解析结果的命名空间对象。这样，其他部分的代码可以通过访问这个对象的属性来获取命令行参数的值。
#     例如，可以通过 `args.action` 获取 `--action` 参数的值。
def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--action',default='merge_all',help='Action to operate')
    parser.add_argument('--dataset', default='WebQSP', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', default='train', help='split to operate on') # the split file: ['dev','test','train']

    return parser.parse_args()


# 这段代码定义了一个函数 `combine_entities_from_FACC1_and_elq(dataset, split, sample_size=10)`，该函数用于合并来自 FACC1 和 ELQ 的实体链接结果。
# 1. 通过 `load_json` 函数加载 FACC1 和 ELQ 的实体链接结果。
# 2. 初始化一个空字典 `combined_res` 用于存储合并后的结果。
# 3. 通过 `load_json` 函数加载 ELQ 在训练集上的实体信息，并将其转换为列表形式。
# 4. 遍历 ELQ 的实体链接结果，对每个问题进行以下操作：
#    - 从 ELQ 和 FACC1 的实体链接结果中取出候选实体，并按照得分进行排序。
#    - 将两者的实体链接结果逐个合并，确保合并后的结果不超过指定的 `sample_size`。
#    - 如果两个结果中的实体数量不足 `sample_size`，则从训练集的 ELQ 实体中随机抽样填充。
#    - 将合并后的实体链接结果保存到 `combined_res` 字典中。
# 5. 将合并后的结果写入 JSON 文件，并在特定情况下更新实体标签。
# 函数的主要目的是为特定问题生成一个混合的实体链接结果，包含来自 FACC1 和 ELQ 的候选实体。这样的混合可能有助于提高实体链接的鲁棒性和多样性。
def combine_entities_from_FACC1_and_elq(dataset, split, sample_size=10):
    """ Combine the linking results from FACC1 and ELQ """
    entity_dir = f'data/{dataset}/entity_retrieval/candidate_entities'

    facc1_disamb_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_facc1.json')
    elq_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_elq.json')

    combined_res = dict()

    train_entities_elq = {}
    elq_res_train = load_json(f'{entity_dir}/{dataset}_train_cand_entities_elq.json')
    for qid,cand_ents in elq_res_train.items():
        for ent in cand_ents:
            train_entities_elq[ent['id']] = ent['label']

    train_entities_elq = [{"id":mid,"label":label} for mid,label in train_entities_elq.items()]


    for qid in tqdm(elq_res,total=len(elq_res),desc=f'Merging candidate entities of {split}'):
        cur = dict() # unique by mid

        elq_result = elq_res[qid]
        facc1_result = facc1_disamb_res.get(qid,[])
        
        # sort by score
        elq_result = sorted(elq_result, key=lambda d: d.get('score', -20.0), reverse=True)
        facc1_result = sorted(facc1_result, key=lambda d: d.get('logit', -20.0), reverse=True)

        # merge the linking results of ELQ and FACC1 one by one
        idx = 0
        while len(cur.keys()) < sample_size:
            if idx < len(elq_result):
                cur[elq_result[idx]["id"]] = elq_result[idx]
            if len(cur.keys()) < sample_size and idx < len(facc1_result):
                cur[facc1_result[idx]["id"]] = facc1_result[idx]
            if idx >= len(elq_result) and idx >= len(facc1_result):
                break
            idx += 1

        if len(cur.keys()) < sample_size:
            # sample some entities to reach the sample size
            diff_entities = list(filter(lambda v: v["id"] not in cur.keys(), train_entities_elq))
            random_entities = random.sample(diff_entities, 10 - len(cur.keys()))
            for ent in random_entities:
                cur[ent["id"]] = ent

        assert len(cur.keys()) == sample_size, print(qid)
        combined_res[qid] = list(cur.values())

    merged_file_path = f'{entity_dir}/{dataset}_{split}_merged_cand_entities_elq_facc1.json'
    print(f'Writing merged candidate entities to {merged_file_path}')
    dump_json(combined_res, merged_file_path, indent=4)

    if dataset.lower() == 'cwq':
        update_entity_label(dirname=entity_dir, dataset=dataset)

# 这段代码定义了一个函数 `make_sorted_relation_dataset_from_logits(dataset, split)`，该函数用于根据预测的关系 logits 创建一个按 logits 排序的关系数据集。
#     以下是代码的主要步骤和功能：
# 1. 指定输出目录 `output_dir`。
# 2. 根据数据集和拆分名称构建文件路径，包括输入文件（TSV 文件和 logits 文件）和输出文件。
# 3. 通过 `torch.load` 函数加载 logits。
# 4. 将 logits 转换为列表，并将其与 TSV 文件进行比较，确保长度匹配。
# 5. 使用 `pd.read_csv` 函数加载 TSV 文件，得到一个 DataFrame 对象。
# 6. 构建一个映射 `rowid2qid`，将行号映射到问题 ID。
# 7. 初始化一个空字典 `cand_rel_bank` 用于存储候选关系及其 logits。
# 8. 遍历 logits 列表，对每个样本提取相关信息，并将关系及其 logits 存储到 `cand_rel_bank` 中。
# 9. 初始化一个空字典 `cand_rel_logit_map` 用于存储按 logits 排序的关系列表。
# 10. 遍历 `cand_rel_bank`，对每个问题的关系列表按 logits 进行排序。
# 11. 将 `cand_rel_logit_map` 存储为 JSON 文件。
# 12. 初始化一个空字典 `final_candRel_map` 用于存储最终的按 logits 排序的关系列表。
# 13. 遍历原始数据集，为每个问题获取按 logits 排序的关系列表，并存储到 `final_candRel_map` 中。
# 14. 将 `final_candRel_map` 存储为 JSON 文件。
# 整体而言，这个函数的目的是根据模型对关系的预测 logits，为每个问题生成一个按 logits 排序的关系列表，并将这些列表存储为 JSON 文件。这样的排序可以用于后续的关系推断任务。
def make_sorted_relation_dataset_from_logits(dataset, split):

    assert dataset in ['CWQ','WebQSP']
    if dataset == 'WebQSP':
        assert split in ['train', 'test', 'train_2hop', 'test_2hop']
    else:
        assert split in ['test','train','dev']

    output_dir = f'data/{dataset}/relation_retrieval/candidate_relations'
    
    
    if dataset=='CWQ':
        tsv_file = f'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{split}.tsv'
        logits_file = f'data/CWQ/relation_retrieval/cross-encoder/saved_models/mask_mention_1epoch_question_relation/CWQ_ep_1.pt_{split}/logits.pt'
        idmap = load_json(f'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ_{split}_id_index_map.json')
    elif dataset=='WebQSP':
        tsv_file = f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv'
        logits_file = f'data/WebQSP/relation_retrieval/cross-encoder/saved_models/rich_relation_3epochs_question_relation/WebQSP_ep_3.pt_{split}/logits.pt'
        idmap = load_json(f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{split}_id_index_map.json')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logits = torch.load(logits_file,map_location=torch.device('cpu'))

    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))
    print('tsv_file: {}'.format(tsv_file))
    tsv_df = pd.read_csv(tsv_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
                            

    print('Tsv len:', len(tsv_df))
    print('Question Num:',len(tsv_df['question'].unique()))

    # the length of predicted logits must match the num of input examples
    assert(len(logits_list)==len(tsv_df))

    
    if dataset.lower()=='webqsp':
        if split in ['train_2hop', 'train']:
            split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.train.expr.json')
        elif split in ['test', 'test_2hop']:
            split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.test.expr.json')
    else:
        split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')


    rowid2qid = {} # map rowid to qid
        

    for qid in idmap:
        rowid_start = idmap[qid]['start']
        rowid_end = idmap[qid]['end']
        #rowid2qid[rowid]=qid
        for i in range(rowid_start,rowid_end+1):
           rowid2qid[i]=qid


    # cand_rel_bank = {} # Dict[Question, Dict[Relation:logit]]
    cand_rel_bank = defaultdict(dict)
    rel_info_map = defaultdict(str)
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits of {split}'):
        logit = float(logit[1])
        row_id = tsv_df.loc[idx]['id']
        question = tsv_df.loc[idx]['question']
        rich_rel = tsv_df.loc[idx]['relation']
        rel = rich_rel.split("|")[0]
        rel_info = " | ".join(rich_rel.split("|")).replace("."," , ").replace("_"," ")
        #cwq_id = question2id.get(question,None)
        qid = rowid2qid[row_id]

        if not qid:
            print(question)
            cand_rel_bank[qid]= {}
        else:
            cand_rel_bank[qid][rel]=logit

        if not rel in rel_info_map:
            rel_info_map[rel] = rel_info
            

    cand_rel_logit_map = {}
    for qid in tqdm(cand_rel_bank,total=len(cand_rel_bank),desc='Sorting rels...'):
        cand_rel_maps = cand_rel_bank[qid]
        cand_rel_list = [(rel,logit,rel_info_map[rel]) for rel,logit in cand_rel_maps.items()]
        cand_rel_list.sort(key=lambda x:x[1],reverse=True)
        
        cand_rel_logit_map[qid]=cand_rel_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump_json(cand_rel_logit_map,os.path.join(output_dir,f'{dataset}_{split}_cand_rel_logits.json'),indent=4)

    final_candRel_map = defaultdict(list) # Dict[Question,List[Rel]]   sorted by logits

    for ori_data in tqdm(split_dataset,total=len(split_dataset),desc=f'{split} Dumping... '):
        if dataset=='CWQ':
            qid = ori_data['ID']
        else:
            qid = ori_data['QuestionId']
        # cand_rel_map = cand_rel_bank.get(qid,None)
        cand_rel_list = cand_rel_logit_map.get(qid,None)
        if not cand_rel_list:
            final_candRel_map[qid]=[]
        else:
            # cand_rel_list = list(cand_rel_map.keys())
            # cand_rel_list.sort(key=lambda x:float(cand_rel_map[x]),reverse=True)
            final_candRel_map[qid]=[x[0] for x in cand_rel_list]

    sorted_cand_rel_name = os.path.join(output_dir,f'{dataset}_{split}_cand_rels_sorted.json')
    dump_json(final_candRel_map,sorted_cand_rel_name,indent=4)   


# 这段代码定义了一个函数 `get_all_unique_candidate_entities(dataset) -> List[str]`，该函数用于获取指定数据集的所有唯一候选实体的标识符（entity ids）。
# 1. 指定实体目录 `ent_dir` 和存储唯一实体标识符的文件路径 `unique_entities_file`。
# 2. 如果存储唯一实体标识符的文件已存在，则直接从文件中加载唯一实体标识符。
# 3. 如果文件不存在，通过加载合并后的训练、测试和（如果存在的话）验证数据中的候选实体，构建一个包含所有唯一实体标识符的集合 `unique_entities`。
# 4. 将 `unique_entities` 转换为列表，并存储为 JSON 文件。
# 5. 返回包含唯一实体标识符的列表。
# 这个函数的主要目的是为了获取数据集中所有唯一的候选实体标识符。这些实体标识符可能用于后续的实体链接、实体关系等任务。
def get_all_unique_candidate_entities(dataset)->List[str]:
    """Get unique candidate entity ids of {dataset}"""

    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
    
    if os.path.exists(unique_entities_file):
        print(f'Loading unique candidate entities from {unique_entities_file}')
        unique_entities = load_json(unique_entities_file)
    else:
        print(f'Processing candidate entities...')

        train_data = load_json(f'{ent_dir}/{dataset}_train_merged_cand_entities_elq_facc1.json')
        test_data = load_json(f'{ent_dir}/{dataset}_test_merged_cand_entities_elq_facc1.json')

        if dataset=='CWQ':
            dev_data = load_json(f'{ent_dir}/{dataset}_dev_merged_cand_entities_elq_facc1.json')
        else:
            dev_data = None

        unique_entities = set()
        for qid in train_data.keys():
            for ent in train_data[qid]:
                unique_entities.add(ent["id"])
        
        
        for qid in test_data.keys():
            for ent in test_data[qid]:
                unique_entities.add(ent["id"])

        if dev_data:
            for qid in dev_data.keys():
                for ent in dev_data[qid]:
                    unique_entities.add(ent["id"])
                
        print(f'Wrinting unique candidate entities to {unique_entities_file}')
        dump_json(list(unique_entities), unique_entities_file ,indent=4)
    
    return unique_entities

# 这段代码定义了一个函数 `get_entities_in_out_relations(dataset, unique_candidate_entities) -> Dict[str, Dict[str, List[str]]]`，
#     该函数用于获取指定数据集中候选实体的入边和出边关系。以下是代码的主要步骤和功能：
# 1. 指定实体目录 `ent_dir` 和存储入边和出边关系的文件路径 `in_out_rels_file`。
# 2. 如果存储关系的文件已存在，则直接从文件中加载入边和出边关系。
# 3. 如果文件不存在，通过加载所有候选实体，获取每个实体的入边和出边关系。
# 4. 忽略指定的领域（`IGONORED_DOMAIN_LIST`），对每个实体获取其出边和入边关系。
# 5. 将关系信息存储为字典，并将字典存储为 JSON 文件。
# 6. 返回包含候选实体入边和出边关系的字典。
# 这个函数的主要目的是为了获取数据集中所有候选实体的入边和出边关系。这些关系信息可能用于后续的实体关系推断等任务。
def get_entities_in_out_relations(dataset,unique_candidate_entities)->Dict[str,Dict[str,List[str]]]:
    
    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    in_out_rels_file = f'{ent_dir}/{dataset}_candidate_entities_in_out_relations_new.json'

    if os.path.exists(in_out_rels_file):
        print(f'Loading cached 1hop relations from {in_out_rels_file}')
        in_out_rels = load_json(in_out_rels_file)
    else:

        if unique_candidate_entities:
            entities = unique_candidate_entities
        else:
            unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
            entities = load_json(unique_entities_file)            
        
        IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld', 'freebase', 'user']
        in_out_rels = dict()
        
        for ent in tqdm(entities,total=len(entities),desc='Fetching 1hop relations of candidate entities'):            
            
            relations_out = get_out_relations_with_odbc(ent)
            relations_out = [x for x in relations_out if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            relations_in = get_in_relations_with_odbc(ent)
            relations_in = [x for x in relations_in if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            in_out_rels[ent] = {
                'out_relations': relations_out,
                'in_relations': relations_in
            }
        
        print(f'Writing 1hop relations to {in_out_rels_file}')
        dump_json(in_out_rels, in_out_rels_file)

    return in_out_rels


# 这段代码定义了一个函数 `merge_all_data_for_logical_form_generation(dataset, split)`，
#     该函数用于合并数据集中包含语法树（sexpr）和SPARQL查询的示例，并进行标准化。以下是代码的主要步骤和功能：
# 1. 加载包含语法树的数据集 (`dataset_with_sexpr`)。
# 2. 初始化全局实体标签映射 (`global_ent_label_map`)、全局关系标签映射 (`global_rel_label_map`)和全局类型标签映射 (`global_type_label_map`)。
# 3. 对于每个示例，提取其问题 (`question`)、组合类型 (`comp_type`)、答案 (`answer`)、SPARQL查询 (`sparql`) 和语法树 (`sexpr`)。
# 4. 使用 `vanilla_sexpr_linearization_method` 函数对语法树进行线性化，并将其作为标准化的语法树 (`normed_sexpr`)。
# 5. 从SPARQL查询中提取实体和关系，并构建与示例中实体和关系对应的地图。
# 6. 构建新的示例对象，包括问题、组合类型、答案、SPARQL查询、语法树、标准化的语法树以及实体、关系和类型的地图。
# 7. 将新的示例对象添加到 `merged_data_all` 列表中。
# 8. 创建目录 `merged_data_dir`，如果目录不存在的话。
# 9. 将 `merged_data_all` 列表以JSON格式写入文件，并命名为 `{dataset}_{split}.json`。
# 10. 打印合并数据的文件路径，并输出 "Writing finished"。
# 这个函数的主要目的是将包含语法树和SPARQL查询的数据集合并，并对数据进行标准化，以便进一步用于生成逻辑形式查询。
def merge_all_data_for_logical_form_generation(dataset, split):

    dataset_with_sexpr = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')
        
    global_ent_label_map = {}
    global_rel_label_map = {}
    global_type_label_map = {}
    
    merged_data_all = []

    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Processing {dataset}_{split}'):
        
        new_example = {}
        
        if dataset=='CWQ':
            qid = example["ID"]
        elif dataset=='WebQSP':
            qid = example['QuestionId']
        question = example['question'] if dataset=='CWQ' else example['ProcessedQuestion']
        comp_type = example["compositionality_type"] if dataset=='CWQ' else None                
        
        if dataset=='CWQ':
            sexpr = example['SExpr']
            sparql = example['sparql']
            if split=='test':
                answer = example["answer"]
            else:
                answer = [x['answer_id'] for x in example['answers']]
        elif dataset=='WebQSP':
            # for WebQSP choose 
            # 1. shortest sparql
            # 2. s-expression converted from this sparql should leads to same execution results.
            parses = example['Parses']
            shortest_idx = 0
            shortest_len = 9999
            for i in range(len(parses)):
                if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                    if len(parses[i]['Sparql']) < shortest_len:
                        shortest_idx = i
                        shortest_len = len(parses[i]['Sparql'])
                
            sexpr = parses[shortest_idx]['SExpr']
            sparql = parses[shortest_idx]['Sparql']
            answer = [x['AnswerArgument'] for x in parses[shortest_idx]['Answers']]
                
        gold_ent_label_map = {}
        gold_rel_label_map = {}
        gold_type_label_map = {}
        # normed_sexpr = example['question']
            
        normed_sexpr = vanilla_sexpr_linearization_method(sexpr)
        gold_entities = extract_mentioned_entities_from_sparql(sparql)
        gold_relations = extract_mentioned_relations_from_sparql(sparql)
        
        for entity in gold_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)
            if entity_label is not None:
                gold_ent_label_map[entity] = entity_label
                global_ent_label_map[entity] = entity_label

            if is_type and entity_label is not None:
                gold_type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

            
        for rel in gold_relations:
            linear_rel = _textualize_relation(rel)
            gold_rel_label_map[rel] = linear_rel
            global_rel_label_map[rel] = linear_rel

        
        new_example['ID']=qid
        new_example['question'] = question
        new_example['comp_type'] = comp_type
        new_example['answer'] = answer
        new_example['sparql'] = sparql
        new_example['sexpr'] = sexpr
        new_example['normed_sexpr'] = normed_sexpr
        new_example['gold_entity_map'] = gold_ent_label_map
        new_example['gold_relation_map'] = gold_rel_label_map
        new_example['gold_type_map'] = gold_type_label_map


        merged_data_all.append(new_example)
    
    merged_data_dir = f'data/{dataset}/generation/merged'
    if not os.path.exists(merged_data_dir):
        os.makedirs(merged_data_dir)
    merged_data_file = f'{merged_data_dir}/{dataset}_{split}.json'

    print(f'Wrinting merged data to {merged_data_file}...')
    dump_json(merged_data_all,merged_data_file,indent=4)
    print('Writing finished')

# 这段代码定义了一个名为 `get_merged_disambiguated_entities` 的函数，该函数用于获取通过实体检索器（entity retrievers）进行消歧的实体。以下是代码的主要步骤和功能：
# 1. 确定存储结果的目录 `disamb_ent_dir` 和文件名 `disamb_ent_file`。
# 2. 检查是否已经存在缓存文件，如果存在，则直接加载并返回。
# 3. 如果缓存文件不存在，加载候选实体的文件（ELQ 和 FACC1 提供的候选实体）。
# 4. 对于 ELQ 提供的候选实体，按照查询 ID（qid）和提及（mention）进行分组，每个提及只保留得分最高的实体。
# 5. 对于 FACC1 提供的候选实体，按照查询 ID 和提及进行分组，每个提及只保留得分最高的实体。
# 6. 合并 ELQ 和 FACC1 提供的消歧实体信息。
# 7. 对于 CWQ 数据集，处理不同实体 ID 的情况，保留标签信息，并在得分满足条件的情况下，加入 ELQ 的实体。
# 8. 对于 WebQSP 数据集，根据 ELQ 提供的实体构建消歧实体信息。
# 9. 将消歧实体信息写入文件 `disamb_ent_file`。
# 10. 返回消歧实体的映射。
# 这个函数的主要目的是通过合并 ELQ 和 FACC1 提供的候选实体，生成消歧的实体信息。
def get_merged_disambiguated_entities(dataset, split):
    """Get disambiguated entities by entity retrievers (one entity for one mention)"""
    
    disamb_ent_dir = f"data/{dataset}/entity_retrieval/disamb_entities"
    
    disamb_ent_file = f"{disamb_ent_dir}/{dataset}_merged_{split}_disamb_entities.json"

    if os.path.exists(disamb_ent_file):
        print(f'Loading disamb entities from {disamb_ent_file}')
        disamb_ent_map = load_json(disamb_ent_file)
        return disamb_ent_map
    else:
        cand_ent_dir = f"data/{dataset}/entity_retrieval/candidate_entities"
        elq_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_elq.json"
        facc1_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_facc1.json"
        
        elq_cand_ents = load_json(elq_cand_ent_file)
        facc1_cand_ents = load_json(facc1_cand_ent_file)

        # entities linked and ranked by elq
        elq_disamb_ents = {}        
        for qid,cand_ents in elq_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent
            
            elq_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]

        # entities linked and ranked by facc1
        facc1_disamb_ents = {}
        for qid,cand_ents in facc1_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent            
            facc1_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]


        disamb_ent_map = {}
        
        # merge the disambed entities        
        for qid in elq_disamb_ents:
            disamb_entities = {}

            facc1_entities = facc1_disamb_ents[qid]
            elq_entities = elq_disamb_ents[qid]

            if dataset.lower() == 'cwq':
                for ent in facc1_entities:
                    disamb_entities[ent['id']]={
                        "id":ent["id"],
                        "label":ent["label"],
                        "mention":ent["mention"],
                        "perfect_match":ent["perfect_match"]
                    }

                elq_entities = [ent for ent in elq_entities if ent['score'] > -1.5]
                for ent in elq_entities:
                    if ent["id"] not in disamb_entities: # different id
                        if ent["label"]:
                            disamb_entities[ent['id']] = {
                                "id":ent["id"],
                                "label":ent["label"],
                                "mention":ent["mention"],
                                "perfect_match":ent["perfect_match"]
                            }

            elif dataset.lower() == 'webqsp':
                for ent in elq_entities:
                    disamb_entities[ent['id']]={
                        "id":ent["id"],
                        "label": get_label_with_odbc(ent['id']),
                        "mention":ent["mention"],
                        "perfect_match":ent["perfect_match"]
                    }
                
                for ent in facc1_entities:
                    if ent['id'] not in disamb_entities: # different id
                        disamb_entities[ent['id']]={
                            "id":ent["id"],
                            "label":get_label_with_odbc(ent['id']),
                            "mention":ent["mention"],
                            "perfect_match":ent["perfect_match"]
                        }
            
            disamb_entities = [ent for (_,ent) in disamb_entities.items()]

            disamb_ent_map[qid] = disamb_entities

        print(f'Writing disamb entities into {disamb_ent_file}')
        if not os.path.exists(disamb_ent_dir):
            os.makedirs(disamb_ent_dir)

        dump_json(disamb_ent_map, disamb_ent_file, indent=4)

        return disamb_ent_map
        


# 这段代码定义了一个名为 `extract_type_label_from_dataset` 的函数，该函数用于从给定数据集和拆分中提取实体类型标签。以下是代码的主要步骤和功能：
# 1. 加载指定数据集和拆分的语法树数据（通过 `load_json` 函数）。
# 2. 初始化全局实体类型标签映射 `global_type_label_map`。
# 3. 遍历每个数据样本，提取其查询 ID 和 SPARQL 查询。
# 4. 对于每个数据样本，提取 SPARQL 查询中提及的实体，并获取其类型标签。
# 5. 将提取的实体类型标签添加到 `type_label_map` 中，并更新全局映射 `global_type_label_map`。
# 6. 如果不存在目录 `label_maps`，则创建该目录。
# 7. 将全局实体类型标签映射写入文件 `{dir_name}/{dataset}_{split}_type_label_map.json`。
# 8. 打印消息表示完成操作。
# 该函数的主要目的是从给定数据集和拆分中提取实体类型标签，并将结果保存到文件中。
def extract_type_label_from_dataset(dataset, split):


    train_databank =load_json(f"data/{dataset}/sexpr/{dataset}.{split}.expr.json")

    global_type_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        qid = data['ID']
        sparql = data['sparql']

        type_label_map = {}

        # extract entity labels
        gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
        for entity in gt_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)

            if is_type and entity_label is not None:
                type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

    dir_name = f"data/{dataset}/generation/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dump_json(global_type_label_map, f'{dir_name}/{dataset}_{split}_type_label_map.json',indent=4)

    print("done")


# 这段代码定义了一个名为 `extract_type_label_from_dataset_webqsp` 的函数，该函数用于从给定的 WebQSP 数据集和拆分中提取实体类型标签。以下是代码的主要步骤和功能：
# 1. 加载指定 WebQSP 数据集和拆分的语法树数据（通过 `load_json` 函数）。
# 2. 初始化全局实体类型标签映射 `global_type_label_map`。
# 3. 遍历每个数据样本，提取其问题 ID 和包含多个 "Parse" 的列表。
# 4. 对于每个 "Parse"，提取其中的 SPARQL 查询。
# 5. 对于每个 SPARQL 查询，提取其中提及的实体，并获取其类型标签。
# 6. 将提取的实体类型标签添加到 `type_label_map` 中，并更新全局映射 `global_type_label_map`。
# 7. 如果不存在目录 `label_maps`，则创建该目录。
# 8. 将全局实体类型标签映射写入文件 `{dir_name}/{dataset}_{split}_type_label_map.json`。
# 9. 打印消息表示完成操作。
# 该函数的主要目的是从给定的 WebQSP 数据集和拆分中提取实体类型标签，并将结果保存到文件中。
#     不同之处在于 WebQSP 数据集的每个问题可能包含多个 "Parse"，因此对每个 "Parse" 都会提取类型标签。
def extract_type_label_from_dataset_webqsp(dataset, split):
    # Each WebQSP question may have more than one "Parse"，get label_map of all "Parse"s
    
    train_databank =load_json(f"data/{dataset}/sexpr/{dataset}.{split}.expr.json")

    global_type_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        qid = data['QuestionId']
        type_label_map = {}
        
        for parse in data["Parses"]:
            sparql = parse["Sparql"]
            # extract entity labels
            gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
            for entity in gt_entities:
                is_type = False
                entity_types = get_types_with_odbc(entity)
                if "type.type" in entity_types:
                    is_type = True

                entity_label = get_label_with_odbc(entity)

                if is_type and entity_label is not None:
                    type_label_map[entity] = entity_label
                    global_type_label_map[entity] = entity_label

    dir_name = f"data/{dataset}/generation/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    dump_json(global_type_label_map, f'{dir_name}/{dataset}_{split}_type_label_map.json',indent=4)

    print("done")


# 这段代码定义了一个名为 `get_all_entity` 的函数，该函数的目的是从给定目录中的特定数据集的训练、开发和测试集中提取所有实体，并将结果保存到一个 JSON 文件中。
# 1. 初始化一个空的集合 `all_entity`，用于存储所有实体的唯一标识符。
# 2. 遍历指定数据集（`dataset`）的三个拆分（'train'、'dev' 和 'test'）。
# 3. 对于每个拆分，加载已经合并的候选实体结果（`{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json` 文件）。
# 4. 遍历每个问题的实体结果，提取实体的唯一标识符（'id' 字段）。
# 5. 将提取的实体标识符添加到 `all_entity` 集合中。
# 6. 将 `all_entity` 集合转换为列表，并将列表写入名为 `{dirname}/{dataset}_all_entities.json` 的 JSON 文件中。
# 该函数的主要作用是收集指定数据集中所有拆分的实体标识符，并将它们保存到一个 JSON 文件中，以便进一步使用。
def get_all_entity(dirname, dataset):
    all_entity = set()
    for split in ['train', 'dev', 'test']:
        el_res = load_json(f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json')
        for qid in el_res:
            values = el_res[qid]
            for item in values:
                all_entity.add(item['id'])
                    
    dump_json(list(all_entity), f'{dirname}/{dataset}_all_entities.json')


# 这段代码定义了一个名为 `update_entity_label` 的函数，其主要目的是将数据集中所有实体的标签进行标准化。以下是代码的主要步骤和功能：
# 1. 检查三个拆分（'train'、'dev' 和 'test'）的文件是否都存在，如果不存在，则函数直接返回。确保已准备好所有数据集的拆分。
# 2. 检查包含所有实体标识符的文件 `{dirname}/{dataset}_all_entities.json` 是否存在，如果不存在，则调用 `get_all_entity` 函数以获取所有实体标识符，并将其保存到该文件中。
# 3. 检查包含实体标签映射的文件 `{dirname}/{dataset}_all_label_map.json` 是否存在，如果不存在，则调用 `get_entity_labels` 函数以获取实体标签映射，并将其保存到该文件中。
# 4. 遍历三个拆分的候选实体结果。
# 5. 对于每个实体，如果其标识符在实体标签映射中，则更新实体的标签字段为映射中的对应标签；否则，输出实体标识符。
# 6. 将更新后的候选实体结果保存到文件 `{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json` 中。
# 这个函数的主要作用是确保数据集中的所有实体标签都是经过标准化的，即使用了从所有实体中学到的标签映射。
def update_entity_label(dirname, dataset):
    """Stardardize all entity labels"""
    if not (
        os.path.exists(f'{dirname}/{dataset}_train_merged_cand_entities_elq_facc1.json') and
        os.path.exists(f'{dirname}/{dataset}_dev_merged_cand_entities_elq_facc1.json') and
        os.path.exists(f'{dirname}/{dataset}_test_merged_cand_entities_elq_facc1.json')
    ):
        return # Update label when all dataset splits are ready

    if not os.path.exists(f'{dirname}/{dataset}_all_entities.json'):
        get_all_entity(dirname, dataset)
    assert os.path.exists(f'{dirname}/{dataset}_all_entities.json')

    if not os.path.exists(f'{dirname}/{dataset}_all_label_map.json'):
        get_entity_labels(
            f'{dirname}/{dataset}_all_entities.json',
            f'{dirname}/{dataset}_all_label_map.json'
        )
    assert os.path.exists(f'{dirname}/{dataset}_all_label_map.json')
    
    for split in ['train', 'dev', 'test']:
        el_res = load_json(f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json')
        all_label_map = load_json(f'{dirname}/{dataset}_all_label_map.json')
        
        updated_res = dict()

        for qid in el_res:
            values = el_res[qid]
            for item in values:
                if item["id"] in all_label_map:
                    item['label'] = all_label_map[item['id']]
                else:
                    print(item["id"])
                    
            updated_res[qid] = values
        
        dump_json(updated_res, f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json'.format(split))


# 这个函数的主要目的是替换先前合并的数据文件中的候选关系列表 (`cand_relation_list`)，使用新的关系 logits。以下是函数的主要步骤和功能：
# 1. 从先前合并的文件 (`prev_merged_path`) 中加载先前的合并数据。
# 2. 从排序后的关系文件 (`sorted_relations_path`) 中加载关系 logits，该文件是对2-hop推理的结果。
# 3. 从额外的关系文件 (`addition_relations_path`) 中加载额外的关系 logits，该文件是对bi-encoder top100推理的结果。
# 4. 对于每个样本，根据以下规则选择新的候选关系列表：
#    - 如果关系 logits 排序的文件中没有对应的 qid 或者 logits 数量小于 `topk`，则使用额外的关系 logits。
#    - 否则，使用排序的关系 logits。
# 5. 将新的候选关系列表替换先前合并数据中的 `cand_relation_list` 属性。
# 6. 将更新后的合并数据保存到输出文件 (`output_path`)。
# 总体而言，该函数用于更新先前合并的数据文件中的关系列表，确保使用了最新的关系 logits。
def substitude_relations_in_merged_file(
    prev_merged_path, 
    output_path, 
    sorted_relations_path,
    addition_relations_path,
    topk=10
):
    """
    replace "cand_relation_list" property in previous merged data
    with new relation logits
    """
    prev_merged = load_json(prev_merged_path)
    sorted_relations = load_json(sorted_relations_path) # inference on 2hop
    additional_relation = load_json(addition_relations_path) # inference on bi-encoder top100
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_relations or len(sorted_relations[qid]) < topk: # need exactly 10 relations
            print(qid)
            cand_relations = additional_relation[qid][:topk]
        else:
            cand_relations = sorted_relations[qid][:topk]
        example["cand_relation_list"] = cand_relations
        new_merged.append(example)
    assert len(prev_merged) == len(new_merged)
    dump_json(new_merged, output_path)


# `validation_merged_file` 函数用于验证先前的合并文件和新的合并文件的一致性。具体而言，它执行以下步骤：
# 1. 从先前的合并文件 (`prev_file`) 和新的合并文件 (`new_file`) 中加载数据。
# 2. 确保两个文件中的样本数量相同。
# 3. 逐个比较先前合并文件和新合并文件中的每个样本，确保它们的非 `cand_relation_list` 属性相等。
# 4. 对于 `cand_relation_list` 属性，确保其长度都为 10。
# 在执行这些验证步骤后，如果所有的验证条件都通过，说明两个文件的数据是一致的。如果有不一致之处，将抛出异常，同时输出不一致的样本数量。
def validation_merged_file(prev_file, new_file):
    prev_data = load_json(prev_file)
    new_data = load_json(new_file)
    assert len(prev_data) == len(new_data), print(len(prev_data), len(new_data))
    for (prev, new) in tqdm(zip(prev_data, new_data), total=len(prev_data)):
        for key in prev.keys():
            if key != 'cand_relation_list':
                assert prev[key] == new[key]
            else:
                assert len(prev[key]) == 10
                assert len(new[key]) == 10, print(len(new[key]))


# `substitude_relations_in_merged_file_cwq` 函数用于将先前合并的文件中的关系替换为新的关系logits。以下是该函数的详细解释：
# - `prev_merged_path`：先前合并的数据文件的路径。
# - `output_path`：新合并数据将保存的输出文件路径。
# - `sorted_logits_path`：包含排序的关系logits的文件路径。
# - `topk`：要考虑的前k个关系的数量。
# 该函数执行以下步骤：
# 1. 从指定路径(`prev_merged_path`)加载先前合并的数据。
# 2. 从指定路径(`sorted_logits_path`)加载排序的关系logits。
# 3. 遍历先前合并的数据中的每个示例。
# 4. 从示例中提取问题ID(`qid`)。
# 5. 如果问题ID在排序的关系logits中不存在，则打印该ID。
# 6. 用排序logits中的前k个关系替换示例中的"cand_relation_list"属性。
# 7. 将修改后的示例追加到新合并的数据中。
# 8. 将新合并的数据保存到指定的输出路径。
# 该函数实际上通过基于排序的logits更新先前合并的数据中的"cand_relation_list"属性，使用了新的关系。
def substitude_relations_in_merged_file_cwq(
    prev_merged_path, 
    output_path, 
    sorted_logits_path,
    topk=10,
):
    prev_merged = load_json(prev_merged_path)
    sorted_logits = load_json(sorted_logits_path)
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_logits:
            print(qid)
        example["cand_relation_list"] = sorted_logits[qid][:topk]
        new_merged.append(example)
    dump_json(new_merged, output_path)

# `get_candidate_unique_entities_cwq` 函数用于从CWQ数据集的实体候选文件中获取唯一的实体标识符。以下是该函数的详细解释：
# - `folder`：包含实体候选文件的文件夹路径。
# 该函数执行以下步骤：
# 1. 遍历CWQ数据集的不同拆分（`train`、`dev`、`test`）。
# 2. 对于每个拆分，加载相应的实体候选文件（`CWQ_merged_{split}_disamb_entities.json`）。
# 3. 遍历每个问题ID和相关的实体候选。
# 4. 将每个实体的唯一标识符添加到集合`unique_entities` 中。
# 5. 将集合`unique_entities` 转换为列表并将其保存到文件 `unique_entities.json` 中。
# 该函数的目的是创建一个包含CWQ数据集中所有实体唯一标识符的文件。
def get_candidate_unique_entities_cwq():
    folder = 'data/CWQ/entity_retrieval/disamb_entities'
    unique_entities = set()
    for split in ['train', 'dev', 'test']:
        cand_entity_file = load_json(os.path.join(folder, f'CWQ_merged_{split}_disamb_entities.json'))
        for qid in cand_entity_file:
            for item in cand_entity_file[qid]:
                if item["id"] != "":
                    unique_entities.add(item["id"])
    dump_json(list(unique_entities), os.path.join(folder, 'unique_entities.json'))


# `serialize_rich_relation` 函数用于将包含丰富关系信息的关系序列化为字符串。以下是该函数的详细解释：
# - `relation`：要序列化的关系。
# - `domain_range_dict`：包含关系领域和范围信息的字典。
# - `seperator`：用于连接不同部分的分隔符。
# 该函数执行以下步骤：
# 1. 如果给定的关系 `relation` 不在 `domain_range_dict` 中，直接返回原始关系。
# 2. 如果 `domain_range_dict` 中包含关系 `relation`：
#    - 如果字典中包含 'label' 键，将其值（标签）添加到结果字符串中。
#    - 如果字典中包含 'domain' 键，将其值（领域）添加到结果字符串中。
#    - 如果字典中包含 'range' 键，将其值（范围）添加到结果字符串中。
# 3. 将组合后的字符串返回为序列化的关系。
# 该函数的目的是在保留关系标识的同时，将关系的附加信息（标签、领域、范围）添加到字符串中。
def serialize_rich_relation(relation, domain_range_dict, seperator="|"):
    if relation not in domain_range_dict:
        return relation
    else:
        res = relation
        if 'label' in domain_range_dict[relation]:
            if relation.lower() != domain_range_dict[relation]['label'].lower().replace(' ', ''):
                res += (seperator + domain_range_dict[relation]['label'])
        if 'domain' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['domain'])
        if 'range' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['range'])
        return res


# `construct_common_data` 函数用于构建共享数据，主要包括以下步骤：
# - 从已过滤的关系中提取富关系（附加了标签、领域、范围信息的关系字符串）。
# - 创建关系到富关系的映射和富关系到关系列表的映射。
# - 保存结果到指定的文件路径。
#
# 以下是函数的详细解释：
# - `filtered_relations_path`：已过滤的关系文件的路径。
# - `domain_range_label_map_path`：包含关系领域和范围信息的映射文件的路径。
# - `output_relation_rich_map_path`：输出关系到富关系映射的文件路径。
# - `output_rich_relation_map_path`：输出富关系到关系列表映射的文件路径。
# - `output_filtered_rich_relation_path`：输出已过滤的富关系列表的文件路径。
#
# 函数执行以下步骤：
# 1. 从已过滤的关系中提取富关系，使用 `serialize_rich_relation` 函数。
# 2. 创建关系到富关系的映射和富关系到关系列表的映射。
# 3. 将结果保存到指定的文件路径。
# 该函数的目的是为后续的数据处理和分析提供关系及其富关系的映射。
def construct_common_data(
    filtered_relations_path,
    domain_range_label_map_path,
    output_relation_rich_map_path,
    output_rich_relation_map_path,
    output_filtered_rich_relation_path,
):
    filtered_relations = load_json(filtered_relations_path)
    domain_range_label_map = load_json(domain_range_label_map_path)
    relation_rich_map = dict()
    rich_relation_map = defaultdict(list)
    filtered_rich_relations = []
    for rel in filtered_relations:
        richRelation = serialize_rich_relation(rel, domain_range_label_map).replace('\n', '')
        relation_rich_map[rel] = richRelation
        rich_relation_map[richRelation].append(rel)
        filtered_rich_relations.append(richRelation)
    dump_json(relation_rich_map, output_relation_rich_map_path)
    dump_json(rich_relation_map, output_rich_relation_map_path)
    dump_json(filtered_rich_relations, output_filtered_rich_relation_path)


# 上述代码段是一个 Python 脚本，根据提供的命令行参数执行不同的操作。以下是对代码的简要解释：
# - `_parse_args()`: 解析命令行参数的函数。从命令行获取了动作（action）、数据集（dataset）和数据拆分（split）等参数。
# - `if action.lower()=='merge_entity':`：如果动作是 "merge_entity"，则调用 `combine_entities_from_FACC1_and_elq` 函数，将来自 FACC1 和 ELQ 的实体合并。
# - `elif action.lower()=='merge_relation':`：如果动作是 "merge_relation"，则调用 `make_sorted_relation_dataset_from_logits` 函数，从 logits 中构建排序的关系数据集。
# - `elif action.lower()=='merge_all':`：如果动作是 "merge_all"，则调用 `merge_all_data_for_logical_form_generation` 函数，合并所有数据以用于逻辑形式生成。
# - `elif action.lower()=='get_type_label_map':`：如果动作是 "get_type_label_map"，
#     则根据数据集类型调用 `extract_type_label_from_dataset` 或 `extract_type_label_from_dataset_webqsp` 函数，提取类型标签映射。
# - `else:`：如果动作不匹配上述条件，则打印使用帮助信息。
# 最后的注释中包含了一个被注释掉的 `construct_common_data` 函数调用，该函数的目的是构建一些通用数据。
if __name__=='__main__':
    
    
    args = _parse_args()
    action = args.action

    if action.lower()=='merge_entity':
        combine_entities_from_FACC1_and_elq(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_relation':
        make_sorted_relation_dataset_from_logits(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_all':
        merge_all_data_for_logical_form_generation(dataset=args.dataset, split=args.split)
    elif action.lower()=='get_type_label_map':
        if args.dataset == "CWQ":
            extract_type_label_from_dataset(dataset=args.dataset, split=args.split)
        elif args.dataset == "WebQSP":
            extract_type_label_from_dataset_webqsp(dataset=args.dataset, split=args.split)
    else:
        print('usage: data_process.py action [--dataset DATASET] --split SPLIT ')

    # construct_common_data(
    #     'data/common_data/freebase_relations_filtered.json',
    #     'data/common_data/fb_relations_domain_range_label.json',
    #     'data/common_data/fb_relation_rich_map.json',
    #     'data/common_data/fb_rich_relation_map.json',
    #     'data/common_data/freebase_richRelations_filtered.json',
    # )
