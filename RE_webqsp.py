
# 这段代码主要涉及到以下内容：
# 1. **导入模块和库**：
#    - `argparse`：用于解析命令行参数的库。
#    - `tqdm`：创建进度条以显示循环迭代进度的库。
#    - `dump_json` 和 `load_json`：用于将数据保存为 JSON 文件和从 JSON 文件加载数据的自定义函数。
#    - `sparql_executor` 模块：包含了执行 SPARQL 查询的相关函数。
#    - `SimCSE`：一个用于计算文本相似度的模型。
# 2. **初始化 SimCSE 模型**：
#    创建了一个 SimCSE 模型实例，使用的是 RoBERTa-large 模型的预训练权重。
import argparse
from generation.cwq_evaluate import cwq_evaluate_valid_results
from generation.webqsp_evaluate_offcial import webqsp_evaluate_valid_results
from components.utils import dump_json, load_json
from tqdm import tqdm
from executor.sparql_executor import execute_query_with_odbc, get_2hop_relations_with_odbc_wo_filter
from executor.logic_form_util import lisp_to_sparql
import re
import os
from entity_retrieval import surface_index_memory
import difflib
import itertools
from simcse import SimCSE
import shutil
model = SimCSE("princeton-nlp/unsup-simcse-roberta-large")


# 这个函数用于判断一个字符串是否表示一个数字。函数首先尝试使用 `float()` 函数将字符串转换为浮点数，如果成功则返回 `True`，表示该字符串是一个数字。
#     如果 `float()` 转换失败，函数还尝试使用 `unicodedata.numeric()` 函数（处理 ASCII 字符）将字符串转换为浮点数，同样成功则返回 `True`。
#         最后，如果上述两个转换都失败，函数返回 `False`，表示该字符串不是一个数字。
# 这里还需要注意的是，字符串在被转换之前，通过 `replace` 方法替换了一些可能存在的逗号形式的小数点。这是因为有些地区使用逗号而不是点来表示小数。
def is_number(t):
    t = t.replace(" , ",".")
    t = t.replace(", ",".")
    t = t.replace(" ,",".")
    try:
        float(t)
        return True
    except ValueError:
        pass
    try:
        import unicodedata  # handle ascii
        unicodedata.numeric(t)  # string of number --> float
        return True
    except (TypeError, ValueError):
        pass
    return False


# 这个函数定义了一个命令行参数解析器，用于解析脚本运行时传入的参数。具体的参数包括：
# - `--split`: 用于指定操作的数据集划分，可以是 `test`、`dev` 或 `train`。
# - `--pred_file`: 用于指定预测结果文件的路径，这是一个 JSON 文件。
# - `--server_ip` 和 `--server_port`: 用于调试时指定服务器的 IP 地址和端口。
# - `--qid`: 用于指定单个问题的 ID 以进行调试，如果不指定则默认为 `None`。
# - `--test_batch_size`: 用于指定测试时的批处理大小，默认为 2。
# - `--dataset`: 用于指定数据集类型，可以是 `CWQ` 或 `WebQSP`。
# - `--beam_size`: 用于指定束搜索的大小，默认为 50。
# - `--golden_ent`: 如果指定该参数，表示使用黄金实体进行评估。
# 函数返回一个包含解析结果的命名空间。
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', help='split to operate on, can be `test`, `dev` and `train`')
    parser.add_argument('--pred_file', default='Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json', help='topk prediction file')
    parser.add_argument('--server_ip', default=None, help='server ip for debugging')
    parser.add_argument('--server_port', default=None, help='server port for debugging')
    parser.add_argument('--qid',default=None,type=str, help='single qid for debug, None by default' )
    parser.add_argument('--test_batch_size', default=2)
    parser.add_argument('--dataset', default='WebQSP', type=str, help='dataset type, can be `CWQ、`WebQSP`')
    parser.add_argument('--beam_size', default=50, type=int)
    parser.add_argument('--golden_ent', default=False, action='store_true')

    args = parser.parse_args()

    print(f'split:{args.split}, topk_file:{args.pred_file}')
    return args

# 这个函数是一个简单的类型检查器，用于检测给定的字符串 token 是否符合某些特定的模式，如整数、浮点数或日期，并返回相应的类型。具体的类型检测规则如下：
# - 如果 token 符合年份的格式（如 "2022"），并且年份小于 3000，则将其类型标记为 `http://www.w3.org/2001/XMLSchema#dateTime`。
# - 如果 token 符合年份和月份的格式（如 "2022-01"），则将其类型标记为 `http://www.w3.org/2001/XMLSchema#dateTime`。
# - 如果 token 符合年份、月份和日期的格式（如 "2022-01-15"），则将其类型标记为 `http://www.w3.org/2001/XMLSchema#dateTime`。
# - 如果 token 不符合以上任何一种格式，则保持原样。
# 最后，返回带有类型标记的 token 或原始的 token。
def type_checker(token:str):
    """Check the type of a token, e.g. Integer, Float or date.
       Return original token if no type is detected."""
    
    pattern_year = r"^\d{4}$"
    pattern_year_month = r"^\d{4}-\d{2}$"
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(pattern_year, token):
        if int(token) < 3000: # >= 3000: low possibility to be a year
            token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month_date, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    else:
        return token

    return token


# 这个函数是用于处理日期字符串的后处理函数。在查询知识库时，有时候知识库会自动将日期进行补全。例如：
# - 对于 "1996"，知识库可能会自动补全为 "1996-01-01"。
# - 对于 "1906-04-18"，知识库可能会自动补全为 "1906-04-18 05:12:00"。
# 该函数的作用是对这种被自动补全的日期字符串进行处理，将其还原为更符合期望的格式。具体处理规则如下：
# - 如果日期字符串以 "yyyy-mm-dd" 的格式结尾，且时间部分为 "05:12:00"，则去除时间部分。
# - 如果日期字符串以 "yyyy-mm-dd" 的格式结尾，且日期部分为 "01-01"，则去除日期部分。
# 最后返回处理后的日期字符串。
def date_post_process(date_string):
    """
    When quering KB, (our) KB tends to autoComplete a date
    e.g.
        - 1996 --> 1996-01-01
        - 1906-04-18 --> 1906-04-18 05:12:00
    """
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    pattern_year_month_date_moment = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"

    if re.match(pattern_year_month_date_moment, date_string):
        if date_string.endswith('05:12:00'):
            date_string = date_string.replace('05:12:00', '').strip()
    elif re.match(pattern_year_month_date, date_string):
        if date_string.endswith('-01-01'):
            date_string = date_string.replace('-01-01', '').strip()
    return date_string
        

# 这个函数用于将规范化后的逻辑表达式（Lisp 格式）进行反规范化处理，即还原为原始的逻辑表达式。
#     函数的输入包括规范化后的逻辑表达式 `normed_expr`、实体标签映射 `entity_label_map`、类型标签映射 `type_label_map` 以及 `surface_index`。
def denormalize_s_expr_new(normed_expr, entity_label_map, type_label_map, surface_index):

    # 函数的处理流程如下：
    # 1. 将规范化的逻辑表达式中的一些操作符缩写进行还原，例如将 "( greater equal" 缩写为 "( ge"。
    expr = normed_expr
    convert_map ={
        '( greater equal': '( ge',
        '( greater than':'( gt',
        '( less equal':'( le',
        '( less than':'( lt'
    }

    for k in convert_map:
        expr = expr.replace(k,convert_map[k])
        expr = expr.replace(k.upper(),convert_map[k])

    # 2. 将规范化的逻辑表达式拆分为单词（token）。
    # expr = expr.replace(', ',' , ')
    tokens = expr.split(' ')

    segments = []
    prev_left_bracket = False
    prev_left_par = False
    cur_seg = ''

    # 3. 遍历单词列表，根据不同的情况处理不同的标记：
    #    - 如果是 "["，说明进入了一个新的子表达式，将前一个子表达式添加到结果列表中。
    #    - 如果是 "]"，说明子表达式结束，对当前子表达式进行处理：
    #      - 如果子表达式是一个类型（type），则查找类型标签映射，将其替换为具体的类型标签。
    #      - 如果子表达式是一个关系或未链接的实体，根据不同的情况进行处理：
    #        - 如果包含 ", "，视为关系，将其处理为适当的形式。
    #        - 否则，考虑是数字还是实体，进行处理。
    #    - 如果是 "("，说明进入了一个新的子表达式，将 "(" 添加到结果列表中。
    #    - 如果是其他标记，根据不同情况处理：
    #      - 如果是关系操作符（"ge", "gt", "le", "lt"）将其添加到结果列表中。
    #      - 如果是其他单词，查找实体标签映射，将其替换为具体的实体标签。
    # 处理完成后，函数返回多个反规范化后的逻辑表达式，以处理可能的多义性。
    for t in tokens:

        if t=='[':
            prev_left_bracket=True
            if cur_seg:
                segments.append(cur_seg)
        elif t==']':
            prev_left_bracket=False
            cur_seg = cur_seg.strip()

            # find in linear origin map
            processed = False

            if not processed:
                if cur_seg.lower() in type_label_map: # type
                    cur_seg = type_label_map[cur_seg.lower()]
                    processed = True
                else: # relation or unlinked entity
                    if ' , ' in cur_seg:
                        if is_number(cur_seg):
                            # p32-p47,is_number
                            # check if it is a number
                            cur_seg = cur_seg.replace(" , ",".")
                            cur_seg = cur_seg.replace(" ,",".")
                            cur_seg = cur_seg.replace(", ",".")
                        else:
                            # view as relation
                            cur_seg = cur_seg.replace(' , ',',')
                            cur_seg = cur_seg.replace(',','.')
                            cur_seg = cur_seg.replace(' ', '_')
                        processed = True
                    else:
                        search = True
                        if is_number(cur_seg):
                            # p32-p47,is_number
                            search = False
                            cur_seg = cur_seg.replace(" , ",".")
                            cur_seg = cur_seg.replace(" ,",".")
                            cur_seg = cur_seg.replace(", ",".")
                            cur_seg = cur_seg.replace(",","")
                        elif len(entity_label_map.keys()) != 0:
                            search = False
                            if cur_seg.lower() in entity_label_map:
                                cur_seg = entity_label_map[cur_seg.lower()]     
                            else:
                                similarities = model.similarity([cur_seg.lower()], list(entity_label_map.keys()))  
                                merged_list = list(zip([v for _,v in entity_label_map.items()], similarities[0]))
                                sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)[0]
                                if sorted_list[1] > 0.2:
                                    cur_seg = sorted_list[0]
                                else:       
                                    search = True                         
                        if search:
                            facc1_cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(cur_seg,top_k=50)
                            if facc1_cand_entities:
                                temp = []
                                for key in list(facc1_cand_entities.keys())[1:]:
                                    if facc1_cand_entities[key] >= 0.001:
                                        temp.append(key)
                                if len(temp) > 0:
                                    cur_seg = [list(facc1_cand_entities.keys())[0]]+temp
                                else:
                                    cur_seg = list(facc1_cand_entities.keys())[0]

            segments.append(cur_seg)
            cur_seg = ''
        else:
            if prev_left_bracket:
                # in a bracket
                cur_seg = cur_seg + ' '+t
            else:
                if t=='(':
                    prev_left_par = True
                    segments.append(t)
                else:
                    if prev_left_par:
                        if t in ['ge', 'gt', 'le', 'lt']: # [ge, gt, le, lt] lowercase
                            segments.append(t)
                        else:                
                            segments.append(t.upper()) # [and, join, r, argmax, count] upper case
                        prev_left_par = False 
                    else:
                        if t != ')':
                            if t.lower() in entity_label_map:
                                t = entity_label_map[t.lower()]
                            else:
                                t = type_checker(t) # number
                                # p83-p100: type_checker
                        segments.append(t)

    combinations = [list(comb) for comb in itertools.islice(itertools.product(*[item if isinstance(item, list) else [item] for item in segments]),10000)]

    exprs = [" ".join(s) for s in combinations]

    return exprs


# 这段代码定义了一个函数 `execute_normed_s_expr_from_label_maps`，该函数根据输入的标准化表达式（`normed_expr`）
#     以及实体标签映射（`entity_label_map`）、类型标签映射（`type_label_map`）和表面形式索引（`surface_index`）执行一系列操作，并返回查询表达式和查询结果。
# normed_expr，"( AND ( JOIN [ common , topic , notable types ] [ Country ] ) ( JOIN ( R [ location , location , contains ] ) [ United Kingdom ] ) )"
# entity_label_map，# "gold_entity_map": {
#     #     "m.01tzh": "United Kingdom",
#     #     "m.01mp": "Country"
#     # },
# type_label_map：    "m.01mp": "Country",
# 这个函数用于执行规范化的逻辑表达式。函数的输入包括规范化的逻辑表达式 `normed_expr`、实体标签映射 `entity_label_map`、类型标签映射 `type_label_map` 以及 `surface_index`。
def execute_normed_s_expr_from_label_maps(normed_expr, entity_label_map, type_label_map, surface_index):

    # 1. 尝试通过调用 `denormalize_s_expr_new` 函数来对标准化表达式进行反标准化。如果出现异常，函数返回字符串 'null' 和空列表。
    # 函数首先尝试通过 `denormalize_s_expr_new` 函数将规范化的逻辑表达式反规范化为多个可能的逻辑表达式。
    # 这个函数用于将规范化后的逻辑表达式（Lisp 格式）进行反规范化处理，即还原为原始的逻辑表达式。
    # p131-259
    try:
        denorm_sexprs = denormalize_s_expr_new(normed_expr, entity_label_map, type_label_map, surface_index)
    except:
        return 'null', []

    #     然后，对每个反规范化后的逻辑表达式，将其转换为 SPARQL 查询语句，并通过 `lisp_to_sparql` 函数进行转换。
    #     接着，通过 `execute_query_with_odbc` 函数执行 SPARQL 查询，获取查询结果的 denotation。
    #     如果 SPARQL 查询执行成功并返回非空的 denotation，则直接返回查询语句和 denotation。
    # 2. 对反标准化的表达式进行处理，去除额外的空格，然后取前500个字符的子表达式。这是为了避免处理过长的表达式可能导致的性能问题。
    # 3. 对每个子表达式进行进一步处理：
    #    - 如果子表达式中包含 'OR'、'WITH' 或 'PLUS'，则直接将 `denotation` 设置为空列表。
    #    - 否则，将子表达式转换为 SPARQL 查询，并使用 `execute_query_with_odbc` 函数执行查询，得到查询结果 `denotation`。
    #    - 如果查询结果为空，说明没有找到符合条件的结果，进入处理逻辑，重新构建查询并再次执行。
    #    - 最后，如果查询结果非空，则结束循环。
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in denorm_sexprs]
    for query_expr in query_exprs[:500]:
        try:
            # invalid sexprs, may leads to infinite loops
            if 'OR' in query_expr or 'WITH' in query_expr or 'PLUS' in query_expr:
                denotation = []
            else:
                sparql_query = lisp_to_sparql(query_expr)
                denotation = execute_query_with_odbc(sparql_query)
                denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
                if len(denotation) == 0 :
                    
                    ents = set ()
                    
                    for item in sparql_query.replace('(', ' ( ').replace(')', ' ) ').split(' '):
                        if item.startswith("ns:m."):
                            ents.add(item)
                    addline = []
                    for i, ent in enumerate(list(ents)):
                        addline.append(f'{ent} rdfs:label ?en{i} . ')
                        addline.append(f'?ei{i} rdfs:label ?en{i} . ')
                        addline.append(f'FILTER (langMatches( lang(?en{i}), "EN" ) )')
                        sparql_query = sparql_query.replace(ent, f'?ei{i}')
                    clauses = sparql_query.split('\n')
                    for i, line in enumerate(clauses):
                        if line == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
                            clauses = clauses[:i+1] + addline + clauses[i+1:]
                            break
                    sparql_query = '\n'.join(clauses)
                    denotation = execute_query_with_odbc(sparql_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]                    
        except:
            denotation = []
        if len(denotation) != 0 :
            break

    # 如果 SPARQL 查询失败或 denotation 为空，
    #     则尝试进行一些修正，例如添加额外的语句以改善查询结果，然后重新执行 SPARQL 查询，直到成功为止。
    # 最终，函数返回执行成功的 SPARQL 查询语句和 denotation，或者在多次尝试后依然失败时返回默认值 `'null'` 和空列表 `[]`。
    # 4. 如果最终查询结果为空，将返回第一个子表达式（`query_exprs[0]`）作为查询表达式。
    # 5. 最后，函数返回查询表达式和查询结果。
    # 请注意，这段代码处理了一些特殊情况，比如包含特定关键词的表达式以及执行查询后未能获取到结果的情况。
    if len(denotation) == 0 :
        query_expr = query_exprs[0]
    
    return query_expr, denotation

# 这个函数与之前的函数类似，也是执行规范化的逻辑表达式。不同之处在于，这个函数尝试通过 `try_relation` 函数执行反规范化后的逻辑表达式，
#     其中 `try_relation` 函数是一个处理关系的特定方法。在尝试执行前50个反规范化的逻辑表达式时，如果成功执行了其中某一个，就会直接返回查询语句和 denotation。
# 如果在尝试了前50个反规范化的逻辑表达式后仍然没有成功，那么就返回默认值 `'null'` 和空列表 `[]`。
# 这个函数的目的是通过尝试不同的逻辑表达式，特别是关系表达式，来获取有效的 SPARQL 查询语句和 denotation。
def execute_normed_s_expr_from_label_maps_rel(normed_expr, entity_label_map, type_label_map, surface_index):
    try:
        denorm_sexprs = denormalize_s_expr_new(normed_expr, entity_label_map,type_label_map,surface_index)
    except:
        return 'null', []
    
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in denorm_sexprs]

    for d in tqdm(denorm_sexprs[:50]):
        # p375-p445,try_relation
        query_expr, denotation = try_relation(d)
        if len(denotation) != 0 :
            break          
        
    if len(denotation) == 0 :
        query_expr = query_exprs[0]
    
    return query_expr, denotation


# 这个函数的目的似乎是处理一种类似于语义表达式的数据，并在一个知识库中进行查询。以下是代码的大致解释：
# 1. **实体和关系提取：** 从输入字符串 `d` 中提取实体和关系，将其分别放入 `ent_list` 和 `rel_list` 集合中。
# 2. **获取二跳关系：** 对于每个实体，调用 `get_2hop_relations_with_odbc_wo_filter` 函数获取其二跳关系（入关系和出关系），并将这些关系添加到 `cand_rels` 集合中。
# 3. **计算相似度：** 使用 `model.similarity` 计算输入关系和候选关系之间的相似度。
# 4. **选择相似关系：** 对每个输入关系，根据相似度排序并选择相似度较高的前15个关系，存储在 `change` 字典中。
# 5. **替换原始表达式：** 将原始表达式中的关系替换为相似度较高的关系。
# 6. **生成组合表达式：** 使用 `itertools.product` 生成所有可能的组合，并最多保留10000个。
# 7. **生成 SPARQL 查询：** 将组合转换为 SPARQL 查询表达式，并执行查询。
# 8. **处理无效查询：** 如果查询失败（结果为空），则尝试修改查询表达式中的过滤条件，并再次执行查询。
# 9. **返回结果：** 返回查询表达式和查询结果。
# 请注意，代码还包含了一些异常处理，用于处理可能导致无限循环的无效查询表达式。如果在这段代码中有其他方面或具体细节您需要更深入了解的，请告诉我。
def try_relation(d):
    ent_list = set()
    rel_list = set()
    denorm_sexpr = d.split(' ')
    for item in denorm_sexpr:
        if item.startswith('m.'):
            ent_list.add(item)
        elif '.' in item:
            rel_list.add(item)
    ent_list = list(ent_list)
    rel_list = list(rel_list)
    cand_rels = set()
    for ent in ent_list:
        in_rels, out_rels, _ = get_2hop_relations_with_odbc_wo_filter(ent)
        cand_rels = cand_rels | set(in_rels) | set(out_rels)
    cand_rels = list(cand_rels)
    if len(cand_rels) == 0 or len(rel_list) == 0:
        return d.replace('( ','(').replace(' )', ')'), []
    similarities = model.similarity(rel_list, cand_rels)
    change = dict()
    for i, rel in enumerate(rel_list):
        merged_list = list(zip(cand_rels, similarities[i]))
        sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
        change_rel = []
        for s in sorted_list:
            if s[1] > 0.01:
                change_rel.append(s[0])
        change[rel] = change_rel[:15]
    for i, item in enumerate(denorm_sexpr):
        if item in rel_list:
            denorm_sexpr[i] = change[item]
    combinations = [list(comb) for comb in itertools.islice(itertools.product(*[item if isinstance(item, list) else [item] for item in denorm_sexpr]),10000)]
    exprs = [" ".join(s) for s in combinations][:4000]
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in exprs]
    for query_expr in query_exprs:
        try:
            # invalid sexprs, may leads to infinite loops
            if 'OR' in query_expr or 'WITH' in query_expr or 'PLUS' in query_expr:
                denotation = []
            else:
                sparql_query = lisp_to_sparql(query_expr)
                denotation = execute_query_with_odbc(sparql_query)
                denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
                if len(denotation) == 0 :
                    
                    ents = set ()
                    
                    for item in sparql_query.replace('(', ' ( ').replace(')', ' ) ').split(' '):
                        if item.startswith("ns:m."):
                            ents.add(item)
                    addline = []
                    for i, ent in enumerate(list(ents)):
                        addline.append(f'{ent} rdfs:label ?en{i} . ')
                        addline.append(f'?ei{i} rdfs:label ?en{i} . ')
                        addline.append(f'FILTER (langMatches( lang(?en{i}), "EN" ) )')
                        sparql_query = sparql_query.replace(ent, f'?ei{i}')
                    clauses = sparql_query.split('\n')
                    for i, line in enumerate(clauses):
                        if line == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
                            clauses = clauses[:i+1] + addline + clauses[i+1:]
                            break
                    sparql_query = '\n'.join(clauses)
                    denotation = execute_query_with_odbc(sparql_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
        except:
            denotation = []
        if len(denotation) != 0 :
            break              
    if len(denotation) == 0 :
        query_expr = query_exprs[0]      
    return query_expr, denotation  


# 这段代码看起来是一个用于评估模型在问答任务中性能的 Python 函数，采用了一种基于 top-k 的评估方法。函数接受几个参数：
# - `split`：一个字符串，指定要评估的数据集拆分（'dev'、'train' 或 'test'）。
# - `predict_file`：包含模型预测的文件。
# - `dataset`：一个字符串，指定用于评估的数据集（'CWQ' 或 'WebQSP'）。
# default='test'
# Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json
# WebQSP
# 主函数
def aggressive_top_k_eval_new(split, predict_file, dataset):
    """Run top k predictions, using linear origin map"""

    # 下面是代码的逐步解释：
    # 1. **数据集加载：**
    #    - 根据指定的数据集（`CWQ` 或 `WebQSP`），函数加载相应的训练、测试和开发数据集（如果有的话）。
    # 2. **加载预测结果：**
    #    - 从指定的 `predict_file` 文件中加载模型的预测结果。
    if dataset == "CWQ":
        train_gen_dataset = load_json('data/CWQ/generation/merged/CWQ_train.json')
        test_gen_dataset = load_json('data/CWQ/generation/merged/CWQ_test.json')
        dev_gen_dataset = None
        # dev_gen_dataset = load_json('data/CWQ/generation/merged/CWQ_dev.json')
    elif dataset == "WebQSP":
        train_gen_dataset = load_json('data/WebQSP/generation/merged/WebQSP_train.json')
        test_gen_dataset = load_json('data/WebQSP/generation/merged/WebQSP_test.json')
        dev_gen_dataset = None
    predictions = load_json(predict_file)

    # 3. **数据目录信息：**
    #    - 提取 `predict_file` 的目录和文件名信息。
    # 4. **类型映射：**
    #    - 对于两个数据集（`CWQ` 和 `WebQSP`），从标签映射文件加载类型映射。
    print(os.path.dirname(predict_file))
    dirname = os.path.dirname(predict_file)
    filename = os.path.basename(predict_file)

    if split=='dev':
        gen_dataset = dev_gen_dataset
    elif split=='train':
        gen_dataset = train_gen_dataset
    else:
        gen_dataset = test_gen_dataset

    if dataset == "CWQ":
        train_type_map = load_json(f"data/CWQ/generation/label_maps/CWQ_train_type_label_map.json")
        train_type_map = {l.lower():t for t,l in train_type_map.items()}
    elif dataset == "WebQSP":
        train_type_map = load_json(f"data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json")
        train_type_map = {l.lower():t for t,l in train_type_map.items()}

    # 5. **实体表面索引初始化：**
    #    - 使用常见数据文件中的数据创建实体表面索引。
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention", "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention")

    # 6. **评估循环：**
    #    - 函数遍历预测和相应的真实例子对。
    #    - 它尝试通过执行标准化的语义表达式并将其与真实值进行比较来找到第一个可执行的逻辑形式（LF）。
    #    - 代码检查第一个排名的预测是否可执行，并将其与真实值进行比较。
    #    - 如果找到匹配，结果将存储在各种数据结构中。
    # 这段代码看起来是对模型生成的预测进行评估的一部分。以下是代码的主要功能：
    # 1. **变量初始化：**
    #    - `ex_cnt`、`top_hit`：成功执行和排名第一的计数器。
    #    - `lines`：存储每个预测信息的列表。
    #    - `official_lines`：存储官方信息（问题ID和答案）的列表。
    #    - `failed_preds`：存储失败预测信息的列表。
    #    （稍后代码中还有为不同情景初始化的类似计数器和列表。）
    ex_cnt = 0
    top_hit = 0
    lines = []
    official_lines = []
    failed_preds = []


    # 2. **循环遍历预测和数据集：**
    #    - 代码使用 `zip(predictions, gen_dataset)` 遍历每个预测和相应的数据集条目。
    #    - 使用 `tqdm` 库提供迭代时的进度条。
    gen_executable_cnt = 0
    final_executable_cnt = 0
    processed = 0
    # predictions是大模型生成的测试集的答案Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json
    # gen_dataset是测试集的标准答案data/WebQSP/generation/merged/WebQSP_test.json
    # "gold_entity_map": {
    #     "m.01tzh": "Central America",
    #     "m.01mp": "Country"
    # },
    for (pred,gen_feat) in tqdm(zip(predictions,gen_dataset), total=len(gen_dataset), desc=f'Evaluating {split}'):

        denormed_pred = []
        qid = gen_feat['ID']
            
        if args.golden_ent:
            entity_label_map = {v.lower(): k for k, v in list(gen_feat['gold_entity_map'].items())}
        else:
            entity_label_map = {}

        executable_index = None # index of LF being finally executed

        # 3. **处理预测：**
        #    - 对于每个预测，它处理预测的逻辑形式 (`pred['predictions']`) 并将其与实际情况 (`gen_feat['sexpr']`) 进行比较。
        #    - 执行标准化的逻辑形式并检查答案。
        #    - 如果找到答案，它将相关信息记录在 `lines` 和 `official_lines` 列表中。
        #    - 还检查第一名预测是否与实际情况匹配，并相应更新计数器 (`ex_cnt`、`top_hit`)。
        # find the first executable lf
        for rank, p in enumerate(pred['predictions']):
            # p265: execute_normed_s_expr_from_label_maps
            # entity_label_map, train_type_map, surface_index均是全局变量
            # 这个函数用于执行规范化的逻辑表达式。函数的输入包括规范化的逻辑表达式 `normed_expr`、实体标签映射 `entity_label_map`、类型标签映射 `type_label_map` 以及 `surface_index`。
            # lf, answers逻辑形式、答案
            lf, answers = execute_normed_s_expr_from_label_maps(p, entity_label_map, train_type_map, surface_index)

            answers = [date_post_process(ans) for ans in list(answers)]
            # p110-p126: date_post_process

            denormed_pred.append(lf)

            if rank == 0 and lf.lower() ==gen_feat['sexpr'].lower():
                ex_cnt +=1

            if answers:
                executable_index = rank
                lines.append({
                    'qid': qid, 
                    'execute_index': executable_index,
                    'logical_form': lf,
                    'answer':answers,
                    'gt_sexpr': gen_feat['sexpr'], 
                    'gt_normed_sexpr': pred['gen_label'],
                    'pred': pred, 
                    'denormed_pred':denormed_pred
                })

                official_lines.append({
                    "QuestionId": qid,
                    "Answers": answers
                })
               
                if rank==0:
                    top_hit +=1
                break
            elif p.lower() ==gen_feat['normed_sexpr'].lower():
                print(p.lower())
                print(lf.lower())
                print(gen_feat['sexpr'].lower())

        # 4. **处理不可执行的预测：**
        #    - 如果第一次尝试中找不到可执行的查询，它使用另一种方法 (`execute_normed_s_expr_from_label_maps_rel`) 进行重试。
        #    - 如果在重试中找到可执行的查询，它更新计数器 (`gen_executable_cnt`)。
        #    - 如果在两次尝试中都找不到可执行的查询，它在 `failed_preds` 列表中记录失败预测的信息。
        if executable_index is not None:
            # found executable query from generated model
            gen_executable_cnt +=1
        else:
            denormed_pred = []
            
            # find the first executable lf
            for rank, p in enumerate(pred['predictions']):
                lf, answers = execute_normed_s_expr_from_label_maps_rel(p, entity_label_map, train_type_map, surface_index)

                answers = [date_post_process(ans) for ans in list(answers)]
                
                denormed_pred.append(lf)

                if rank == 0 and lf.lower() ==gen_feat['sexpr'].lower():
                    ex_cnt +=1
                
                if answers:
                    executable_index = rank
                    lines.append({
                        'qid': qid, 
                        'execute_index': executable_index,
                        'logical_form': lf, 
                        'answer':answers,
                        'gt_sexpr': gen_feat['sexpr'], 
                        'gt_normed_sexpr': pred['gen_label'],
                        'pred': pred, 
                        'denormed_pred':denormed_pred
                    })

                    official_lines.append({
                        "QuestionId": qid,
                        "Answers": answers
                    })
                
                    if rank==0:
                        top_hit +=1
                    break
                    
            if executable_index is not None:
                # found executable query from generated model
                gen_executable_cnt +=1
                
            else:
            
                failed_preds.append({'qid':qid, 
                                'gt_sexpr': gen_feat['sexpr'], 
                                'gt_normed_sexpr': pred['gen_label'],
                                'pred': pred, 
                                'denormed_pred':denormed_pred})
        
            
        if executable_index is not None:
            final_executable_cnt+=1

        # 5. **更新计数器：**
        #    - 代码根据每个预测的结果更新计数器 (`gen_executable_cnt`、`final_executable_cnt`、`processed`)。
        #    - 每100次迭代时，打印进度信息。
        # 6. **注意：**
        #    - 代码中有被注释掉的代码 (`# if processed==5: break`)，这表明在处理一定数量的例子后存在提前停止的条件。
        # 这个脚本似乎是模型的一个评估或测试过程的一部分。如果您有具体的问题或需要帮助，随时提问！
        processed+=1
        if processed%100==0:
            print(f'Processed:{processed}, gen_executable_cnt:{gen_executable_cnt}')
        # if processed==5:
        #     break

    # 7. **结果分析：**
    #    - 函数打印统计信息，如字符串匹配比例、 top-1 可执行比例、生成的可执行比例和最终可执行比例。
    # 8. **结果导出：**
    #    - 将结果以详细信息的形式存储在 JSON 文件中，包括问题 ID、执行索引、逻辑形式、答案、地面真实语义表达式和预测。
    print('STR Match', ex_cnt/ len(predictions))
    print('TOP 1 Executable', top_hit/ len(predictions))
    print('Gen Executable', gen_executable_cnt/ len(predictions))
    print('Final Executable', final_executable_cnt/ len(predictions))

    result_file = os.path.join(dirname,f'{filename}_gen_sexpr_results.json')
    official_results_file = os.path.join(dirname,f'{filename}_gen_sexpr_results_official_format.json')
    dump_json(lines, result_file, indent=4)
    dump_json(official_lines, official_results_file, indent=4)

    # 9. **失败的预测：**
    #    - 单独存储未成功执行的预测。
    # write failed predictions
    dump_json(failed_preds,os.path.join(dirname,f'{filename}_gen_failed_results.json'),indent=4)
    dump_json({
        'STR Match': ex_cnt/ len(predictions),
        'TOP 1 Executable': top_hit/ len(predictions),
        'Gen Executable': gen_executable_cnt/ len(predictions),
        'Final Executable': final_executable_cnt/ len(predictions)
    }, os.path.join(dirname,f'{filename}_statistics.json'),indent=4)

    # 10. **最终评估：**
    #     - 根据数据集的不同，函数调用不同的评估函数（`cwq_evaluate_valid_results` 或 `webqsp_evaluate_valid_results`）来评估生成的结果。
    # 11. **输出：**
    #     - 函数打印和导出各种评估指标和结果。
    # 这段代码似乎专门用于问答任务，针对的数据集可能是 ComplexWebQuestions（`CWQ`）和 WebQSP（`WebQSP`）。
    #     它涉及将预测的逻辑形式与地面真实表达式进行比较，并根据执行和匹配标准评估模型的性能。
    # evaluate
    if dataset == "CWQ":
        args.pred_file = result_file
        cwq_evaluate_valid_results(args)
    else:
        args.pred_file = official_results_file
        webqsp_evaluate_valid_results(args)


# 这段代码是一个程序的入口，主要是在满足一定条件时调用前面提到的 `aggressive_top_k_eval_new` 函数。让我们逐步解释它：
# 1. `if __name__=='__main__':`
#    - 这是 Python 中一个常见的条件，表示以下代码块只有在当前脚本被直接运行时才会执行，而不是被导入为模块时执行。
if __name__=='__main__':
    """go down the top-k list to get the first executable locial form"""

    # 2. `args = _parse_args():`
    #    - 调用 `_parse_args` 函数，该函数可能是用于解析命令行参数的，但在提供的代码片段中没有显示。
    # 3. Debugger Attach:
    #    - 如果命令行参数中指定了 `server_ip` 和 `server_port`，则尝试启用调试器附加。这在调试时很有用，允许通过远程调试器附加到程序。
    # p60-p75,_parse_args
    args = _parse_args()

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach...",flush=True)
        ptvsd.enable_attach(address=(args.server_ip, args.server_port))
        ptvsd.wait_for_attach()

    # 4. `if args.qid:`
    #    - 如果命令行参数中指定了 `qid`，则执行一些相应的操作。在提供的代码中，这一部分是空的。
    # 5. `else:`
    #    - 如果没有指定 `qid`，则执行以下操作。
        # 6. `if args.golden_ent:`
        #    - 如果命令行参数中指定了 `golden_ent`，则创建一个新的目录，将原始的预测文件复制到这个目录中，并更新 `args.pred_file` 为新的文件路径。
        #     这是为了保存原始预测文件的备份，以便稍后使用。
        # 7. 调用 `aggressive_top_k_eval_new` 函数:
        #    - 调用前面定义的 `aggressive_top_k_eval_new` 函数，传递命令行参数中指定的 `split`、`pred_file` 和 `dataset`。
    # 这段代码的目的是根据命令行参数的设置来执行相应的操作，主要是调用 `aggressive_top_k_eval_new` 函数进行模型预测结果的评估。
    if args.qid:
        pass
    else:
        if args.golden_ent:
            new_dir_path = os.path.join(os.path.dirname(args.pred_file),'golden_ent_predict')
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            new_dir_name = os.path.join(new_dir_path,args.pred_file.split('/')[-1])
            shutil.copyfile(args.pred_file, new_dir_name)
            args.pred_file = new_dir_name
        aggressive_top_k_eval_new(args.split, args.pred_file, args.dataset)
        # p457-p702,aggressive_top_k_eval_new

