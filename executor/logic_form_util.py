import networkx as nx
from typing import List, Union
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from executor.sparql_executor import execute_query
import re
import json


REVERSE = True  # if REVERSE, then reverse relations are also taken into account for semantic EM

path = str(Path(__file__).parent.absolute())

reverse_properties = {}
with open(path + '/../ontology/reverse_properties', 'r') as f:
    for line in f:
        reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

with open(path + '/../ontology/fb_roles', 'r') as f:
    content = f.readlines()

relation_dr = {}
relations = set()
for line in content:
    fields = line.split()
    relation_dr[fields[1]] = (fields[0], fields[2])
    relations.add(fields[1])

with open(path + '/../ontology/fb_types', 'r') as f:
    content = f.readlines()

upper_types = defaultdict(lambda: set())

types = set()
for line in content:
    fields = line.split()
    upper_types[fields[0]].add(fields[2])
    types.add(fields[0])
    types.add(fields[2])

function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}

# 这段代码实现了一个函数 `lisp_to_nested_expression`，该函数将逻辑表达式的Lisp形式字符串转换为嵌套的列表表示形式。
#   这种表示方式更易于处理和分析。以下是代码的主要逻辑和解释：
# 1. **函数参数：**
#    - `lisp_string`: 包含逻辑表达式的Lisp形式的字符串。
# 2. **算法逻辑：**
#    - `stack` 用于跟踪当前正在构建的嵌套表达式的栈。
#    - `current_expression` 用于跟踪当前正在构建的表达式。
#    - `tokens` 使用空格分隔Lisp字符串生成的令牌列表。
# 3. **解析过程：**
#    - 对于每个令牌，检查其是否以 '(' 或 ')' 开头或结尾。
#    - 如果是 '(', 创建一个嵌套表达式，并将其添加到 `current_expression` 中，然后将其推送到堆栈，将 `current_expression` 更新为新的嵌套表达式。
#    - 如果是 ')', 则将 `current_expression` 更新为堆栈的顶部元素，表示当前表达式已完成。
# 4. **返回值：**
#    - 返回最终构建的嵌套表达式的列表形式，即 `current_expression[0]`。
# 该函数通过迭代Lisp字符串的令牌，并构建嵌套表达式，实现了从Lisp形式到嵌套列表的转换。
# 输入："sexpr": '(JOIN (R location.country.languages_spoken) m.03_r3)'
# 输出：['JOIN', ['R', 'location.country.languages_spoken'], 'm.03_r3']
def lisp_to_nested_expression(lisp_string):
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def get_symbol_type(symbol: str) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations:
        return 4
    elif symbol:
        return 1


def same_logical_form(form1: str, form2: str) -> bool:
    if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
        return False
    try:
        G1 = logical_form_to_graph(lisp_to_nested_expression(form1))
    except Exception:
        return False
    try:
        G2 = logical_form_to_graph(lisp_to_nested_expression(form2))
    except Exception:
        return False

    def node_match(n1, n2):
        if n1['id'] == n2['id'] and n1['type'] == n2['type']:
            func1 = n1.pop('function', 'none')
            func2 = n2.pop('function', 'none')
            tc1 = n1.pop('tc', 'none')
            tc2 = n2.pop('tc', 'none')

            if func1 == func2 and tc1 == tc2:
                return True
            else:
                return False
            # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
            #     return True
            # elif 'function' not in n1 and 'function' not in n2:
            #     return True
            # else:
            #     return False
        else:
            return False

    def multi_edge_match(e1, e2):
        if len(e1) != len(e2):
            return False
        values1 = []
        values2 = []
        for v in e1.values():
            values1.append(v['relation'])
        for v in e2.values():
            values2.append(v['relation'])
        return sorted(values1) == sorted(values2)

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)


def logical_form_to_graph(expression: List) -> nx.MultiGraph:
    G = _get_graph(expression)
    G.nodes[len(G.nodes())]['question_node'] = 1
    return G


def _get_graph(
        expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
    if isinstance(expression, str):
        G = nx.MultiDiGraph()
        if get_symbol_type(expression) == 1:
            G.add_node(1, id=expression, type='entity')
        elif get_symbol_type(expression) == 2:
            G.add_node(1, id=expression, type='literal')
        elif get_symbol_type(expression) == 3:
            G.add_node(1, id=expression, type='class')
            # G.add_node(1, id="common.topic", type='class')
        elif get_symbol_type(expression) == 4:  # relation or attribute
            domain, rang = relation_dr[expression]
            G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
            G.add_node(2, id=domain, type='class')
            G.add_edge(2, 1, relation=expression)

            if REVERSE:
                if expression in reverse_properties:
                    G.add_edge(1, 2, relation=reverse_properties[expression])

        return G

    if expression[0] == 'R':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        mapping = {}
        for n in G.nodes():
            mapping[n] = size - n + 1
        G = nx.relabel_nodes(G, mapping)
        return G

    elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
        G1 = _get_graph(expression=expression[1])
        G2 = _get_graph(expression=expression[2])

        size = len(G2.nodes())
        qn_id = size
        if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
            if G2.nodes[qn_id]['id'] in upper_types[G1.nodes[1]['id']]:
                G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G = nx.compose(G1, G2)

        if expression[0] != 'JOIN':
            G.nodes[1]['function'] = function_map[expression[0]]

        return G

    elif expression[0] == 'AND':
        G1 = _get_graph(expression[1])
        G2 = _get_graph(expression[2])

        size1 = len(G1.nodes())
        size2 = len(G2.nodes())
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']
            # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
            # So here for the AND function we force it to choose the type explicitly provided in the logical form
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'COUNT':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['function'] = 'count'

        return G

    elif expression[0].__contains__('ARG'):
        G1 = _get_graph(expression[1])
        size1 = len(G1.nodes())
        G2 = _get_graph(expression[2])
        size2 = len(G2.nodes())
        # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
        G2.nodes[1]['id'] = 0
        G2.nodes[1]['type'] = 'literal'
        G2.nodes[1]['function'] = expression[0].lower()
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']

        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'TC':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['tc'] = (expression[2], expression[3])

        return G


def graph_to_logical_form(G, start, count: bool = False):
    if count:
        return '(COUNT ' + none_function(G, start) + ')'
    else:
        return none_function(G, start)


def get_end_num(G, s):
    end_num = defaultdict(lambda: 0)
    for edge in list(G.edges(s)):  # for directed graph G.edges is the same as G.out_edges, not including G.in_edges
        end_num[list(edge)[1]] += 1
    return end_num


def set_visited(G, s, e, relation):
    end_num = get_end_num(G, s)
    for i in range(0, end_num[e]):
        if G.edges[s, e, i]['relation'] == relation:
            G.edges[s, e, i]['visited'] = True


def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print("error: binary function should have 2 parameters!")
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    else:
        if len(elements) == 2:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' \
                   + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'


def count_function(G, start):
    return '(COUNT ' + none_function(G, start) + ')'


def none_function(G, start, arg_node=None, type_constraint=True):
    if arg_node is not None:
        arg = G.nodes[arg_node]['function']
        path = list(nx.all_simple_paths(G, start, arg_node))
        assert len(path) == 1
        arg_clause = []
        for i in range(0, len(path[0]) - 1):
            edge = G.edges[path[0][i], path[0][i + 1], 0]
            if edge['reverse']:
                relation = '(R ' + edge['relation'] + ')'
            else:
                relation = edge['relation']
            arg_clause.append(relation)

        # Deleting edges until the first node with out degree > 2 is meet
        # (conceptually it should be 1, but remember that add edges is both directions)
        while i >= 0:
            flag = False
            if G.out_degree[path[0][i]] > 2:
                flag = True
            G.remove_edge(path[0][i], path[0][i + 1], 0)
            i -= 1
            if flag:
                break

        if len(arg_clause) > 1:
            arg_clause = binary_nesting(function='JOIN', elements=arg_clause)
            # arg_clause = ' '.join(arg_clause)
        else:
            arg_clause = arg_clause[0]

        return '(' + arg.upper() + ' ' + none_function(G, start) + ' ' + arg_clause + ')'

    # arg = -1
    # for nei in G[start]:
    #     if G.nodes[nei]['function'].__contains__('arg'):
    #         arg = nei
    #         arg_function = G.nodes[nei]['function']
    # if arg != -1:
    #     edge = G.edges[start, arg, 0]
    #     if edge['reverse']:
    #         relation = '(R ' + edge['relation'] + ')'
    #     else:
    #         relation = edge['relation']
    #     G.remove_edge(start, arg, 0)
    #     return '(' + arg_function.upper() + ' ' + none_function(G, start) + ' ' + relation + ')'

    if G.nodes[start]['type'] != 'class':
        return G.nodes[start]['id']

    end_num = get_end_num(G, start)
    clauses = []

    if G.nodes[start]['question'] and type_constraint:
        clauses.append(G.nodes[start]['id'])
    for key in end_num.keys():
        for i in range(0, end_num[key]):
            if not G.edges[start, key, i]['visited']:
                relation = G.edges[start, key, i]['relation']
                G.edges[start, key, i]['visited'] = True
                set_visited(G, key, start, relation)
                if G.edges[start, key, i]['reverse']:
                    relation = '(R ' + relation + ')'
                if G.nodes[key]['function'].__contains__('<') or G.nodes[key]['function'].__contains__('>'):
                    if G.nodes[key]['function'] == '>':
                        clauses.append('(gt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '>=':
                        clauses.append('(ge ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<':
                        clauses.append('(lt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<=':
                        clauses.append('(le ' + relation + ' ' + none_function(G, key) + ')')
                else:
                    clauses.append('(JOIN ' + relation + ' ' + none_function(G, key) + ')')

    if len(clauses) == 0:
        return G.nodes[start]['id']

    if len(clauses) == 1:
        return clauses[0]
    else:
        return binary_nesting(function='AND', elements=clauses)


def get_lisp_from_graph_query(graph_query):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    for node in graph_query['nodes']:
        #         G.add_node(node['nid'], id=node['id'].replace('.', '/'), type=node['node_type'], question=node['question_node'], function=node['function'])
        G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                   function=node['function'], cla=node['class'])
        if node['question_node'] == 1:
            qid = node['nid']
        if node['function'] != 'none':
            aggregation = node['function']
            if node['function'].__contains__('arg'):
                arg_node = node['nid']
    for edge in graph_query['edges']:
        G.add_edge(edge['start'], edge['end'], relation=edge['relation'], reverse=False, visited=False)
        G.add_edge(edge['end'], edge['start'], relation=edge['relation'], reverse=True, visited=False)
    if 'count' == aggregation:
        # print(count_function(G, qid))
        return count_function(G, qid)
    else:
        # print(none_function(G, qid))
        return none_function(G, qid, arg_node=arg_node)

# 这段代码是一个将Lisp查询语言转换为SPARQL查询语言的函数。Lisp是一种编程语言，而SPARQL是一种用于查询RDF数据的查询语言。
# 以下是对代码的主要解释：
# 1. `lisp_to_sparql` 函数接受一个Lisp查询字符串作为输入，并返回相应的SPARQL查询字符串。
# 2. 代码首先初始化一些变量，包括存储子句、排序子句和实体的列表，以及一些用于处理变量的字典。
# 3. 函数使用 `lisp_to_nested_expression` 函数将Lisp查询字符串转换为嵌套表达式（nested expression）。
# 4. 通过检查表达式的第一个元素，确定是否是超级拉代词（superlative），并相应地进行处理。
# 5. 使用 `_linearize_lisp_expression` 函数将嵌套表达式线性化，生成子程序的列表。
# 6. 遍历子程序列表，生成相应的SPARQL子句。
# 7. 处理JOIN子句，包括处理实体、变量和文字。
# 8. 处理AND子句，处理变量之间的关系。
# 9. 处理比较子句（le、lt、ge、gt），包括处理2-hop约束。
# 10. 处理时间约束（TC子句）。
# 11. 处理ARGMIN和ARGMAX子句，用于处理最大和最小值。
# 12. 处理COUNT子句，用于计数。
# 13. 处理变量的合并，通过相同变量的等价关系进行合并。
# 14. 处理实体过滤器和变量过滤器。
# 15. 生成最终的SPARQL查询字符串，包括SELECT子句、WHERE子句、FILTER子句等。
# 这段代码的主要目的是将Lisp查询翻译为可以在RDF数据上运行的SPARQL查询。其中包括处理JOIN、AND、比较、时间约束等各种查询操作。在转换过程中，还处理了一些特殊情况，例如2-hop约束、实体过滤器和变量过滤器。
def lisp_to_sparql(lisp_program: str):

    # 这段代码的目的是将Lisp查询语言转换为SPARQL查询语言。以下是代码的主要部分：
    # 1. `clauses` 是用于存储生成的SPARQL查询子句的列表。
    # 2. `order_clauses` 用于存储排序相关的SPARQL子句。
    # 3. `entities` 是一个集合，用于收集实体，以便后续进行过滤。
    # 4. `identical_variables_r` 是一个字典，用于记录变量之间的等价关系。字典的键应该比值大，因为在后续处理中，将使用小变量替换大变量。
    # 5. 通过调用 `lisp_to_nested_expression` 函数将Lisp查询转换为嵌套表达式（nested expression）。
    #    - 函数开始时初始化了一些变量，如`clauses`、`order_clauses`和`entities`。
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False

    #    - 它检查LISP表达式是否表示超级拉丁查询（ARGMAX或ARGMIN）。
    #    - 如果表达式包含n-hop关系，它会处理并展开这些关系。
    # 6. `superlative` 变量标志是否存在超级拉代词（ARGMAX 或 ARGMIN）。
    # 7. 如果存在超级拉代词，则进行特殊处理，将 `JOIN` 从关系链中移除，并根据第二个参数中的关系数量确定其arity。
    # check SUPERLATIVE
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list): # n-hop relations
            # TODO: in denormalization, ARGMAX and JOIN may wrongly concat two relations to one
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    # 2. **LISP表达式的线性化：**
    #    - 函数通过`_linearize_lisp_expression`线性化LISP表达式，并收集问题变量（`question_var`）。
    # 8. 使用 `_linearize_lisp_expression` 函数将嵌套表达式线性化，生成子程序的列表。
    # 9. `question_var` 记录问题变量的位置。
    # 10. `count` 变量标志是否存在COUNT子句。
    # 这段代码的剩余部分包括遍历子程序列表，根据不同的子程序类型生成相应的SPARQL子句，处理JOIN、AND、比较、时间约束、ARGMIN、ARGMAX、COUNT等操作，
    #     并在转换过程中处理特殊情况，如2-hop约束、实体过滤器和变量过滤器。最终，生成完整的SPARQL查询字符串。
    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1 # get the question_var (last sub_formula_id)
    count = False

    # 这是一个用于找到变量的根（Root）的辅助函数。函数名为 `get_root`，它接收一个整数参数 `var`，表示要查找根的变量。
    #     在这个函数中，使用了一个循环，不断地从 `identical_variables_r` 字典中查找变量的等价关系，直到找到根为止。
    # 具体步骤如下：
    # 1. 函数通过 `while var in identical_variables_r` 条件检查，不断循环，直到变量 `var` 不再在 `identical_variables_r` 中出现为止。
    # 2. 在循环体内，通过 `var = identical_variables_r[var]` 更新变量 `var` 的值，将其更新为 `identical_variables_r` 中对应的等价关系的值。
    # 3. 当循环结束时，说明找到了根，函数返回最终的根变量。
    # 4. **变量等价性处理：**
    #    - 该函数包括一个名为`get_root`的辅助函数，用于找到等价变量的根。
    #    - 它管理变量等价性并更新`identical_variables_r`字典。
    # 这个函数的主要作用是在处理等价关系时，通过逐级查找，找到变量的根节点。在处理相同变量的等价关系时，通常会将变量连接到一个共同的根节点，以简化等价关系的表示。
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var


    # 3. **遍历子程序：**
    #    - 函数然后通过线性化的子程序进行迭代。
    #    - 如果子程序是JOIN操作，它生成相应的SPARQL子句。
    #    - 如果子程序是AND操作，它处理变量等价性并生成相应的子句。
    #    - 处理比较操作（le、lt、ge、gt），处理数字值和2-hop约束。
    #    - 对于时间约束（TC），根据时间间隔生成过滤器。
    #    - 对于超级拉丁查询（ARGMIN、ARGMAX），处理排序和限制子句。
    #    - 对于COUNT操作，设置变量等价性。
    for i, subp in enumerate(sub_programs):
        i = str(i)

        # 这部分代码是用于处理查询子句中的 'JOIN' 类型的子程序。以下是代码的主要逻辑解释：
        # 1. **判断是否为 R 关系：**
        #    - 首先检查子程序的第一个元素是否为 'JOIN'。
        #    - 接着判断第二个元素是否为列表，以确定是否为 R 关系。
            # 2. **处理 R 关系：**
            #    - 如果是 R 关系，则进一步判断第三个元素的类型。
            #    - 如果是实体（以 "m." 或 "g." 开头），则生成相应的三元组，将实体与关系连接。
            #    - 如果是变量（以 '#' 开头），则生成相应的三元组，将变量与关系连接。
            #    - 如果是文字（literal），则将文字处理为标准形式并生成相应的三元组。
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            # subp[2] = subp[2].split("^^")[0] + '-08:00^^' + subp[2].split("^^")[1]
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")

            # 3. **处理非 R 关系（实体、变量、文字或两跳关系）：**
            #    - 如果不是 R 关系，那么根据第三个元素的类型生成相应的三元组。
            #    - 如果是实体，则生成实体之间的三元组。
            #    - 如果是变量，则生成变量与关系之间的三元组。
            #    - 如果是文字（literal），则将文字处理为标准形式并生成相应的三元组。
            #    - 如果是两跳关系，根据特定的规则处理。
            # 上述代码段主要处理了 'JOIN' 子程序中不同情况下的三元组生成逻辑。在具体的查询构建过程中，这些生成的三元组将构成 SPARQL 查询语句的一部分。
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                else:  # literal or  2 hop relation (JOIN r1 r2) 
                    if re.match(r'[\w_]*\.[\w_]*\.[\w_]*',subp[2]):
                        # 2-hop relation
                        pass
                    else:
                        # literal or number or type
                        if subp[2].__contains__('^^'): # literal with datatype
                            data_type_string = subp[2].split("^^")[1]
                            if '#' in data_type_string:
                                data_type = data_type_string.split("#")[1]
                            elif 'xsd:' in data_type_string:
                                data_type = data_type_string.split('xsd:')[1]
                            else:
                                data_type = 'dateTime'    
                            if data_type not in ['integer', 'float', 'dateTime','date']:
                                subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            else:
                                subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                        elif re.match("[a-zA-Z_]*\.[a-zA-Z_]*",subp[2]): # type e.g. education.university
                            subp[2]='ns:'+subp[2]
                        elif len(subp)>3: # error splitting, e.g. "2100 Woodward Avenue"@en
                            subp[2]=" ".join(subp[2:])

                        clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        # 这段代码处理了子程序中的逻辑 'AND' 条件。以下是代码的主要逻辑解释：
        # 1. **检测是否为 'AND' 条件：**
        #    - 通过检查子程序的第一个元素，判断它是否为 'AND'。
        # 2. **处理 'AND' 条件：**
        #    - 如果子程序的第二个元素以 '#' 开头，说明是两个变量之间的 'AND' 关系。
        #    - 记录两个变量的根节点，并将它们设置为相等，以便后续合并相同变量。
        #    - 如果子程序的第二个元素不以 '#' 开头，说明是一个变量和一个类之间的 'AND' 关系。
        #    - 在 SPARQL 查询中添加相应的条件，表示该变量是指定类的实例。
        # 这段代码将 'AND' 条件转化为相应的 SPARQL 查询约束，实现了对逻辑 'AND' 条件的处理。
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        # 这段代码处理了子程序中的数值比较约束（'le', 'lt', 'ge', 'gt'）。以下是代码的主要逻辑解释：
        # 1. **检测是否为数值比较约束：**
        #    - 通过检查子程序的第一个元素，判断它是否为 'le', 'lt', 'ge', 'gt' 中的一种。
        # 2. **处理数值比较约束：**
        #    - 如果子程序的第二个元素以 '#' 开头，说明是二跳约束，需要处理两个关系之间的连接。
        #    - 根据是否是反向关系，构建相应的连接条件。
        #    - 如果子程序的第二个元素不是以 '#' 开头，直接构建数值比较的约束条件，包括过滤器（'FILTER'）和操作符（'le', 'lt', 'ge', 'gt'）。
        # 这段代码将子程序中的数值比较约束转化为相应的 SPARQL 查询约束，实现了对比较约束的处理。
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            if subp[1].startswith('#'): # 2-hop constraint
                line_num = int(subp[1].replace('#',''))
                first_relation = sub_programs[line_num][1]
                second_relation = sub_programs[line_num][2]
                
                if isinstance(first_relation,list): # first relation is reversed
                    clauses.append("?cvt" + " ns:" + first_relation[1] + " ?x"+i+" .")
                else:
                    clauses.append("?x"+i + " ns:"+ first_relation+ " ?cvt .")
                
                if isinstance(second_relation,list): #second relation is reversed
                    clauses.append("?y"+ i + " ns:"+ second_relation[1] + " ?cvt .")
                else:
                    clauses.append("?cvt"+ " ns:" + second_relation+ " ?y"+i +" .")
            else:
                clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'

            if re.match(r'\d+', subp[2]) or re.match(r'"\d+"^^xsd:integer', subp[2]): # integer
                clauses.append(f"FILTER (xsd:integer(?y{i}) {op} {subp[2]})")
            else: # others
                clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        # 这段代码处理了 'TC'（Transitive Closure）类型的子程序。以下是代码的主要逻辑解释：
        # 1. **检测是否为 'TC' 类型的子程序：**
        #    - 通过检查子程序的第一个元素，判断它是否为 'TC' 类型的操作。
        # 2. **处理 'TC' 类型的子程序：**
        #    - 获取 'TC' 子程序的相关参数，如变量、年份等。
        #    - 构建 'TC' 的时间范围参数，如果年份为 'NOW' 或 'now'，则设定默认时间范围，否则使用给定的年份。
        #    - 获取关系的最后一个标记，并根据标记构建相应的约束条件。
        #    - 添加 'TC' 子程序的查询约束条件，包括时间范围的 'FROM' 和 'TO' 约束。
        # 这段代码通过将 'TC' 子程序的逻辑转化为对应的 SPARQL 查询约束，实现了对传递闭包的处理。
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW' or year == 'now':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                if "^^" in year:
                    year = year.split("^^")[0]
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            # get the last relation token
            rel_from_property = subp[2].split('.')[-1]
            if rel_from_property == 'from':
                rel_to_property = 'to'
            elif rel_from_property =='end_date':
                # swap end_date and start_date
                subp[2] = subp[2].replace('end_date','start_date')
                rel_from_property = 'start_date'
                rel_to_property = 'end_date'
            else: # from_date -> to_date
                rel_to_property = 'to_date'
            opposite_rel = subp[2].replace(rel_from_property,rel_to_property)

            # add <= constraint
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            
            # add >= constraint
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{opposite_rel} ?sk2}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{opposite_rel} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

            # if subp[2][-4:] == "from":
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            # elif subp[2][-8:] =='end_date': # end_date -> start_date
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-8] + "start_date"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-8] + "start_date"} ?sk3 . ')
            # else:  # from_date -> to_date
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
        # 这段代码处理了 'ARGMIN' 和 'ARGMAX' 类型的子程序。以下是代码的主要逻辑解释：
        # 1. **检测是否为 'ARGMIN' 或 'ARGMAX' 类型的子程序：**
        #    - 通过检查子程序的第一个元素，判断它是否为 'ARGMIN' 或 'ARGMAX' 类型的操作。
        # 2. **处理 'ARGMIN' 或 'ARGMAX' 类型的子程序：**
        #    - 判断第二个参数是变量还是类。如果是变量，则将其与当前子程序的索引关联，表示它们相同，以便在查询中合并相同的变量。
        #    - 如果第二个参数是类，则将其添加到查询中，表示从该类开始查询。
        #    - 根据子程序的长度，判断是否有多跳关系（multi-hop relations），并添加相应的三元组模式。
        #    - 如果是 'ARGMIN'，添加升序排序条件，如果是 'ARGMAX'，添加降序排序条件，然后设置 LIMIT 1，以获取排序后的第一个结果。
        # 这段代码构建了相应的查询模式，用于处理 'ARGMIN' 或 'ARGMAX' 操作。在构建完整的 SPARQL 查询语句时，这些信息将用于合并相同的变量和生成排序条件。
        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3: # 1-hop relations
                clauses.append(f'?x{i} ns:{subp[2]} ?arg0 .')
            elif len(subp) > 3: # multi-hop relations, containing cvt
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?arg0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?arg0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?arg0)")
            order_clauses.append("LIMIT 1")
        # 这部分代码处理了 'COUNT' 类型的子程序，该子程序用于对查询结果进行计数。以下是代码的主要逻辑解释：
        # 1. **检测是否为 'COUNT' 类型的子程序：**
        #    - 通过判断子程序的第一个元素是否为 'COUNT'，确定是否处理计数操作。
        # 2. **处理 'COUNT' 类型的子程序：**
        #    - 获取 'COUNT' 操作的变量，并使用 `get_root` 函数获取其根节点。
        #    - 将该变量的根节点与当前子程序的索引关联，表示它们相同，这样可以合并相同的变量。
        #    - 设置 `count` 变量为 `True`，表示在查询构建过程中遇到了计数操作。
        # 在构建完整的 SPARQL 查询语句时，这些信息将用于合并相同的变量和生成相应的计数操作。
        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True



    # 这部分代码的作用是合并相同的变量，并在需要时调整问题变量，同时为超级拉丁查询（superlative）做一些准备工作。
    # 1. **合并相同的变量：**
    #    - 使用`identical_variables_r`字典，将相同根的变量替换为相同的变量。这有助于简化和规范化生成的SPARQL查询。
    # 2. **调整问题变量：**
    #    - 将问题变量（`question_var`）更新为其根，以确保在生成查询时使用相同的变量标识。
    # 3. **超级拉丁查询准备：**
    #    - 如果是超级拉丁查询（superlative），则复制当前的SPARQL子句，以便在后续添加特定于超级拉丁查询的额外子句。
    # 这段代码的目标是在SPARQL查询生成的过程中，通过合并相同的变量和调整问题变量，使得生成的查询更加清晰和可读。
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')


    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    # add entity filters


    # 5. **子句生成：**
    #    - 基于LISP操作的类型，函数生成相应的SPARQL子句并将它们添加到`clauses`列表中。
    # 这部分代码的作用是添加变量过滤器到 SPARQL 查询中，以确保查询结果符合预期。这通常涉及将特定变量排除在结果之外，以满足查询的条件。
    # 1. **提取所有变量并创建过滤器列表：**
    #    - 使用正则表达式(`re.findall(r"\?\w*", clause)`)提取 SPARQL 查询子句中的所有变量。
    #    - 创建一个过滤器列表 (`filter_variables`)，其中包含需要排除的变量，例如，与问题变量相同的实体和其他特定变量。
    # 2. **添加实体过滤器：**
    #    - 对于每个实体，添加一个过滤器，确保问题变量不等于该实体。
    # 3. **添加其他变量过滤器：**
    #    - 对于之前提取的变量列表中的每个变量，添加一个过滤器，确保问题变量不等于该变量。
    # 这些过滤器的目的是确保生成的 SPARQL 查询的结果符合预期条件，同时排除一些不需要的变量或实体。
    # add variable filters
    filter_variables = []
    for clause in clauses:
        variables = re.findall(r"\?\w*",clause)
        if variables:
            for var in variables:
                var = var.strip()
                if var not in filter_variables and var != '?x' and not var.startswith('?sk'):
                    filter_variables.append(var)
                    
    
    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')

    for var in filter_variables:
        clauses.append(f"FILTER (?x != {var})")


    # 这段代码似乎是为了处理 SPARQL 查询中的日期时间变量，并确保它们的格式正确。具体而言，
    #     它通过检查每个子句是否包含日期时间变量，然后将这些变量替换为新的变量，并添加一个 `FILTER` 子句以确保这些新变量的格式正确。
    # 以下是代码的主要步骤：
    # 1. **遍历 SPARQL 子句列表：**
    #    - 将每个 SPARQL 子句存储在 `sentences` 列表中。
    # 2. **处理日期时间变量：**
    #    - 对于每个 SPARQL 子句，检查其长度是否为4，最后一个词是否是句点。
    #    - 如果是，则进一步检查倒数第二个词是否以双引号开头并以双引号结尾，或者是否以特定字符串结尾（表示日期时间类型）。
    #    - 如果是其中之一，将该变量替换为新的变量 `?st{num}`，并添加一个 `FILTER` 子句以确保格式正确。
    # 此代码段的目的是处理日期时间变量的格式，并将其替换为新的变量。请确保代码的其他部分正确处理这些新变量。
    num = 0
    sentences = [s for s in clauses]
    for c , sentence in enumerate(sentences):
        if len(sentence.split(' '))==4 and sentence.split(' ')[-1]=='.':
            if sentence.split(' ')[-2].startswith('"') and sentence.split(' ')[-2].endswith('"'):
                name = sentence.split(' ')[-2]
                clauses[c] = clauses[c].replace(name,f'?st{num}')
                clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                num += 1
            elif sentence.split(' ')[-2].endswith('"^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
                name = sentence.split(' ')[-2].replace("^^<http://www.w3.org/2001/XMLSchema#dateTime>","")
                clauses[c] = clauses[c].replace(sentence.split(' ')[-2],f'?st{num}')
                clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                num += 1
    
    
    # num = 0        
    # sentences = [s for s in clauses]
    # for c , sentence in enumerate(sentences):
    #     sentence = sentence.replace('(',' ( ').replace(')',' ) ')
    #     for sent in sentence.split(' '):
    #         if '"' in sent:
    #             name = re.findall(r'(".*?")', sent)[0]
    #             clauses[c] = clauses[c].replace(sent,f'?st{num}')
    #             clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
    #             num += 1

    # 6. **最终输出：**
    #    - 生成的SPARQL子句存储在`clauses`列表中。
    # 请注意，代码的完整功能可能取决于`_linearize_lisp_expression`的实现以及未包含的代码的其他部分。
    # 这段代码负责构建最终的 SPARQL 查询语句。以下是主要步骤：
    # 1. **插入基本的 `FILTER` 子句和 `WHERE` 子句：**
    #    - 插入一些基本的 `FILTER` 子句，以确保结果中包含的是英文文本或非文字值。
    #    - 插入 `WHERE {` 以开始 SPARQL 查询。
    # 2. **插入选择和前缀子句：**
    #    - 根据是否是 `COUNT` 查询或超级拉丁查询，插入相应的 `SELECT` 子句。
    #    - 插入默认的 `SELECT DISTINCT ?x`。
    # 3. **插入 `}` 以结束 `WHERE` 子句：**
    #    - 插入 `}` 以结束 SPARQL 查询的 `WHERE` 子句。
    # 4. **插入排序子句：**
    #    - 如果存在排序子句，则插入排序子句。
    # 5. **返回最终的 SPARQL 查询语句：**
    #    - 将构建的 SPARQL 查询语句通过换行符连接起来并返回。
    # 这段代码的目的是将之前处理的子句组合成完整的 SPARQL 查询。请确保其他部分的处理逻辑正确，以便生成正确的查询。
    clauses.insert(0,f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        # clauses.insert(0, "{SELECT ?arg0")
        # clauses = arg_clauses + clauses
        # clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)

    # if superlative:
    #     clauses.append('}')
    #     clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


# 这段代码实现了对嵌套的Lisp表达式进行线性化的功能。以下是代码的主要逻辑解释：
# 1. **函数参数：**
#    - `expression`: 一个嵌套的Lisp表达式，表示为一个列表。
#    - `sub_formula_id`: 用于跟踪子公式的唯一ID。
# 2. **递归线性化：**
#    - 函数使用递归方式线性化嵌套的Lisp表达式。
#    - 对于表达式中的每个元素，如果该元素是一个非关系（不是以 'R' 开头），则递归调用 `_linearize_lisp_expression` 函数，将其进一步线性化。
#    - 对于线性化后的子表达式，将其添加到 `sub_formulas` 列表中。
# 3. **替换原表达式：**
#    - 将原始表达式中的非关系元素替换为相应的子表达式的标识符（以 '#' 开头的数字）。
# 4. **唯一ID跟踪：**
#    - 每次线性化一个子表达式时，增加 `sub_formula_id` 的值，确保生成唯一的标识符。
# 5. **返回结果：**
#    - 返回一个包含所有线性化子表达式的列表。
# 这段代码的目的是将嵌套的Lisp表达式转化为线性结构，并用唯一的标识符替代子表达式，以便后续处理和解析。
# linearize nested lisp exressions
# 输入：['JOIN',
#  ['R', 'government.government_position_held.office_holder'],
#  ['TC',
#   ['JOIN',
#    ['R', 'government.governmental_jurisdiction.governing_officials'],
#    ['JOIN',
#     'location.country.national_anthem',
#     ['JOIN', 'government.national_anthem_of_a_country.anthem', 'm.0gg95zf']]],
#   'government.government_position_held.from',
#   'now']]
# 输出：[['JOIN', 'government.national_anthem_of_a_country.anthem', 'm.0gg95zf'],
#  ['JOIN', 'location.country.national_anthem', '#0'],
#  ['JOIN',
#   ['R', 'government.governmental_jurisdiction.governing_officials'],
#   '#1'],
#  ['TC', '#2', 'government.government_position_held.from', 'now'],
#  ['JOIN', ['R', 'government.government_position_held.office_holder'], '#3']]
def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    # 总而言之enumerate就是枚举的意思，把元素一个个列举出来，第一个是什么，第二个是什么，所以他返回的是元素以及对应的索引。
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas

# 这段代码似乎是将Lisp形式的逻辑表达式转换为Lambda DCS（Dependency-based Compositional Semantics）形式的代码。
#     Lambda DCS是一种语义表示形式，通常用于自然语言处理（NLP）任务。
# 以下是代码的主要逻辑和解释：
# 1. **函数参数：**
#    - `expressions`: Lisp形式的逻辑表达式，可以是一个字符串或一个字符串列表。
# 2. **基本情况检查：**
#    - 如果输入的表达式不是列表，而是字符串，直接返回该字符串（基本情况）。
# 3. **递归转换：**
#    - 如果表达式是列表，检查其第一个元素。
#    - 如果第一个元素是 'AND'，则递归地调用 `lisp_to_lambda` 函数转换第二个和第三个元素，然后用 'AND' 连接它们。
#    - 如果第一个元素是 'JOIN'，则递归地调用 `lisp_to_lambda` 函数转换第二个和第三个元素，然后用 '*' 连接它们。
# 4. **备注：**
#    - 代码中只包含了对 'AND' 和 'JOIN' 的处理，这意味着其他可能的逻辑连接词或操作符的转换可能需要根据具体需求进行添加。
# 5. **实现状态：**
#    - 由于注释中提到 "I don't think this is ever gonna be implemented"，代码可能是一个未完成的尝试，而且缺少完整的实现。
# 如果您需要更全面的Lambda DCS转换，您可能需要继续实现其他逻辑连接词和操作符的处理，以及考虑如何转换成Lambda DCS形式的其他部分。
#  I don't think this is ever gonna be implemented
def lisp_to_lambda(expressions: Union[List[str], str]):  # from lisp-grammar formula to lambda DCS
    # expressions = lisp_to_nested_expression(source_formula)
    if not isinstance(expressions, list):
        return expressions
    if expressions[0] == 'AND':
        return lisp_to_lambda(expressions[1]) + ' AND ' + lisp_to_lambda(expressions[2])
    elif expressions[0] == 'JOIN':
        return lisp_to_lambda(expressions[1]) + '*' + lisp_to_lambda(expressions[2])


if __name__=='__main__':
    
    # gt_sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN government.government_position_held.office_position_or_title m.0j5wjnc) (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf)))) government.government_position_held.from NOW))'

    sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf))) government.government_position_held.from now))'    
    sparql = lisp_to_sparql(sexpr)
    print(sparql)
    # 使用`execute_query` 的函数，该函数用于执行 SPARQL 查询并返回结果。
    res = execute_query(sparql)
    print(res)



