
# 这段代码导入了一些常用的 Python 库，包括 `networkx`、`typing`、`defaultdict`、`Path`、`tqdm`、`re` 和 `json`。以下是对每个导入的库的简要解释：
# 1. **`networkx`**: 一个用于创建、分析和绘制复杂网络的库。
# 2. **`typing`**: 提供对类型提示的支持，用于在代码中添加类型信息。
#     - `List`: 表示列表类型。
#     - `Union`: 表示联合类型，即可以是多个类型之一。
# 3. **`defaultdict`**: 在字典的基础上，提供了默认值的功能。
# 4. **`Path`**: 用于处理文件路径的类。
# 5. **`tqdm`**: 用于在循环中显示进度条的库。
# 6. **`re`**: 正则表达式模块，用于处理字符串的匹配和替换。
# 7. **`json`**: 用于处理 JSON 数据的库。
# 这些库通常用于数据分析、网络分析、文件处理等任务。在此之前确保已经安装这些库，可以使用 `pip install networkx tqdm` 进行安装。
import networkx as nx
from typing import List, Union
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from executor.sparql_executor import execute_query
import re
import json


# 这段代码包含一些配置变量和加载本地文件的逻辑。以下是对这部分代码的解释：
# 1. **REVERSE 变量**:
#     - `REVERSE` 是一个布尔变量，如果设置为 `True`，则在语义实体匹配（semantic entity matching）时也考虑反向关系。
# 2. **文件路径和读取**:
#     - `path` 包含当前脚本所在目录的绝对路径。
# 3. **加载反向关系**:
#     - 从文件 `reverse_properties` 中加载反向关系，该文件包含每个关系及其对应的反向关系。
# 4. **加载关系及其领域-范围**:
#     - 从文件 `fb_roles` 中加载关系及其领域-范围信息。
# 5. **加载实体类型**:
#     - 从文件 `fb_types` 中加载实体类型信息。
# 6. **函数映射**:
#     - `function_map` 是一个字典，将字符串表示的函数映射到对应的操作符，例如将 `'le'` 映射到 `'<='`。这可能与后续的代码中使用的比较操作有关。
# 这些配置变量和加载逻辑的目的是准备数据，以便在后续的代码中使用。确保这些文件在指定的路径存在，并且文件内容的格式符合代码的预期。
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


# 这个函数将逻辑形式表示的Lisp字符串转换为嵌套列表的形式。具体来说，它将形如 "(count (division first))" 的Lisp字符串映射为 ['count', ['division', 'first']]。
# 函数使用了一个堆栈 (`stack`) 来跟踪嵌套表达式的层次结构，以及一个当前表达式 (`current_expression`) 来构建嵌套列表。
#   函数遍历Lisp字符串的标记，根据括号的位置来构建嵌套结构。
# 函数的主要步骤如下：
# 1. 初始化一个堆栈 (`stack`) 和当前表达式 (`current_expression`)。
# 2. 将输入的Lisp字符串分割为标记 (tokens)。
# 3. 对于每个标记，检查括号的位置，以构建嵌套结构。
# 4. 遇到左括号时，创建一个新的嵌套表达式，并将其添加到当前表达式中，并将当前表达式入栈。
# 5. 遇到右括号时，表示当前嵌套表达式结束，从堆栈中弹出当前表达式。
# 6. 返回最终的嵌套列表。
# 例如，对于输入字符串 "(count (division first))"，函数将返回 ['count', ['division', 'first']]。
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


# 这个函数根据输入的符号 (`symbol`) 判断其类型，并返回一个整数表示类型。具体的类型判断如下：
# 1. 如果符号包含 '^^'，则判断为类型 2。
# 2. 如果符号在预先加载的 `types` 集合中， 则判断为类型 3。
# 3. 如果符号在预先加载的 `relations` 集合中，则判断为类型 4。
# 4. 如果符号不为空 (非空字符串)，则判断为类型 1。
# 这个函数主要用于标识符号的类型，这在处理逻辑表达式或Lisp表达式时可能会用到。函数返回一个整数，表示符号的类型。
def get_symbol_type(symbol: str) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations:
        return 4
    elif symbol:
        return 1


# 这个函数的目的是比较两个逻辑形式 (`logical form`) 是否相同。函数接受两个逻辑形式的字符串 (`form1` 和 `form2`) 作为输入，并返回一个布尔值表示这两个逻辑形式是否相同。
# 函数的实现如下：
# 1. 如果 `form1` 或 `form2` 中包含 "@@UNKNOWN@@"，则直接返回 `False`，表示无法确定逻辑形式是否相同。
# 2. 尝试将 `form1` 转换为嵌套表达式，并构建相应的图 `G1`。如果转换或图构建过程中出现异常，返回 `False`。
# 3. 尝试将 `form2` 转换为嵌套表达式，并构建相应的图 `G2`。如果转换或图构建过程中出现异常，返回 `False`。
# 函数的实际比较逻辑可能在函数的后续部分，这里提供的代码片段是不完整的。在比较两个图 (`G1` 和 `G2`) 是否相同时，可能需要进一步检查节点和边的一致性。
# 需要注意的是，上述代码片段中使用了 `lisp_to_nested_expression` 函数将逻辑形式转换为嵌套表达式，
#     并调用了一个未提供的函数 `logical_form_to_graph` 来构建图。这些函数的具体实现可能包含在其他部分的代码中。
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


    # 这段代码定义了一个用于比较两个图节点是否匹配的函数 `node_match`。函数接受两个节点 `n1` 和 `n2` 作为输入，并根据节点的属性（id、type、function、tc）进行比较。
    #     如果这些属性都匹配，则认为节点匹配，返回 `True`；否则，返回 `False`。
    # 具体的比较规则如下：
    # 1. 检查节点的 `id` 和 `type` 是否相同，如果不同则节点不匹配。
    # 2. 检查节点的 `function` 和 `tc` 是否相同，如果不同则节点不匹配。
    # 3. 如果所有属性都匹配，则认为节点匹配。
    # 需要注意的是，该函数可能是用于图匹配的一部分，因为它检查节点的多个属性以确定它们是否匹配。在图匹配中，通常会定义类似的匹配函数，用于确定两个图的节点是否匹配。
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

    # 这段代码定义了一个函数 `multi_edge_match`，该函数用于比较两个图的边是否匹配。函数接受两个参数 `e1` 和 `e2`，分别表示两个图的边集合。该函数执行以下步骤：
    # 1. 检查两个图的边数是否相同，如果不同则直接返回 `False`，表示边不匹配。
    # 2. 从每个图的边集合中提取关系（relation）信息，存储在 `values1` 和 `values2` 中。
    # 3. 将两个关系集合排序，然后比较它们是否相同。如果相同，则返回 `True`，表示边匹配；否则，返回 `False`，表示边不匹配。
    # 最后，代码调用 `nx.is_isomorphic` 函数来检查两个图是否同构。
    #   函数的参数包括 `G1` 和 `G2` 表示两个图，以及 `node_match` 和 `edge_match` 分别表示节点匹配和边匹配的函数。
    # 在这里，`node_match` 使用之前定义的 `node_match` 函数，而 `edge_match` 使用当前定义的 `multi_edge_match` 函数。
    # 这段代码的目的是检查两个图是否同构，其中的节点匹配和边匹配都是通过相应的函数来进行的。
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


# 这段代码定义了一个函数 `logical_form_to_graph`，该函数将逻辑形式（logical form）表示的表达式转换为一个 `networkx` 的多重图（`nx.MultiGraph`）。
#     函数接受一个参数 `expression`，它是一个列表，表示逻辑形式的嵌套表达式。
# 函数首先调用 `_get_graph` 函数，该函数根据传入的逻辑形式表达式构建一个图，并将该图存储在变量 `G` 中。
#     然后，函数为图中的最后一个节点添加一个属性 `question_node`，该属性值为 1，表示这个节点是一个问题节点。
# 最后，函数返回构建好的图 `G`。
# 这段代码的目的是将逻辑形式的表达式转换为一个图，其中最后一个节点被标记为问题节点。
def logical_form_to_graph(expression: List) -> nx.MultiGraph:
    G = _get_graph(expression)
    G.nodes[len(G.nodes())]['question_node'] = 1
    return G


# 这部分代码定义了一个名为 `_get_graph` 的函数，该函数用于根据逻辑形式（logical form）的嵌套表达式构建一个多重图（`nx.MultiGraph`）。
#     该函数接受一个参数 `expression`，该参数是一个列表，表示逻辑形式的嵌套表达式。
# 函数首先检查 `expression` 是否是字符串，如果是字符串，表示这是一个单一的符号（entity、literal、class、relation等），然后根据符号类型将相应的节点添加到图中。
#     这里使用了 `get_symbol_type` 函数来确定符号的类型。
# 如果 `expression` 不是字符串，而是一个列表，那么函数会根据列表的第一个元素来判断逻辑操作的类型。
#   目前支持的逻辑操作类型包括 `'R'`、`'JOIN'`、`'le'`、`'ge'`、`'lt'`、`'gt'`、`'AND'`、`'COUNT'` 和 `'ARG'` 等。
# - 对于 `'R'`，表示取反操作，函数会递归调用 `_get_graph` 处理 `'R'` 操作的子表达式，并进行一些图的重构。
# - 对于 `JOIN`、`'le'`、`'ge'`、`'lt'`、`'gt'`、`'AND'`、`'COUNT'` 和 `'ARG'`，函数也会递归调用 `_get_graph` 处理相应的子表达式，并根据不同类型进行图的合并和重构。
# 最后，函数返回构建好的多重图 `G`。
# 请注意，这部分代码的实现比较复杂，具体的图操作和合并逻辑与具体的逻辑表达式结构相关。如果需要更详细的解释，建议深入研究 `networkx` 图库和逻辑形式的表示方式。
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

# 这部分代码定义了一个名为 `graph_to_logical_form` 的函数，该函数接受一个多重图 `G` 和一个起始节点 `start`，以及一个布尔值 `count`，用于生成表示逻辑形式的字符串。
# 函数的核心是调用 `none_function` 函数，该函数根据图的结构和节点信息生成逻辑形式的字符串。
#     如果 `count` 为真，则函数返回带有 `(COUNT ...)` 表达式的字符串，否则返回正常的逻辑形式字符串。
# 这里的 `none_function` 函数的具体实现需要查看你提供的完整代码，因为逻辑形式的表示方式和图的结构关系紧密，需要了解具体的节点属性和图的组织方式。
def graph_to_logical_form(G, start, count: bool = False):
    if count:
        return '(COUNT ' + none_function(G, start) + ')'
    else:
        return none_function(G, start)

# 这部分代码包含两个函数：`get_end_num` 和 `set_visited`。
# 1. `get_end_num(G, s)` 函数用于获取以节点 `s` 为起点的边的目标节点数量。
#     它遍历由节点 `s` 出发的边，统计每个目标节点的数量，并返回一个字典，其中键是目标节点，值是该目标节点的数量。
# 2. `set_visited(G, s, e, relation)` 函数将节点 `s` 到节点 `e` 的特定关系为 `relation` 的边标记为已访问。
#     该函数首先调用 `get_end_num` 获取以节点 `s` 为起点的边的目标节点数量，然后遍历这些边，找到目标节点为 `e` 且关系为 `relation` 的边，并将其标记为已访问。
#     另外，你还提供了一个名为 `binary_nesting` 的函数。该函数用于构建二元函数的嵌套表达式，可以指定路径上的类型信息。
#     函数接受三个参数：函数名 `function`、元素列表 `elements` 和类型列表 `types_along_path`。
#     根据这些参数，函数生成相应的嵌套表达式。如果元素列表的长度小于 2，则会输出错误信息。函数通过递归调用自身来构建嵌套表达式。
# 这些函数的具体用途和实现方式取决于程序的上下文和整体逻辑。
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

# `count_function` 函数返回一个表示 COUNT 操作的逻辑表达式。它通过调用 `none_function` 函数，传递图 `G` 和起始节点 `start`，得到表示节点 `start` 子图的逻辑表达式。
#     然后，在该表达式外部包裹一个 `(COUNT ...)` 结构，表示对子图的计数操作。
# 这个函数的作用是生成 COUNT 操作的逻辑表达式，用于表示某个子图的计数。
def count_function(G, start):
    return '(COUNT ' + none_function(G, start) + ')'



# 这段代码定义了一个名为 `none_function` 的函数，它似乎用于处理图结构中的节点，根据特定的规则生成语法树中的子表达式。以下是该函数的主要功能：
# 1. **处理带有 `arg_node` 参数的情况：** 如果传入了 `arg_node` 参数，
#     函数会从图 `G` 中的 `start` 节点到 `arg_node` 节点的路径上提取关系，并将这些关系用于生成子表达式。
#     接着，删除路径上的边，直到遇到第一个出度大于2的节点。最后，根据生成的关系子表达式和递归调用 `none_function`，构建包含 `arg` 函数的子表达式。
# 2. **处理不带 `arg_node` 参数的情况：** 如果没有传入 `arg_node` 参数，函数会处理从 `start` 节点出发的情况。
#     如果 `start` 节点的类型不是 'class'，则返回该节点的标识符。否则，获取与 `start` 相连的节点的数量，然后处理这些节点的关系，构建一个包含多个子表达式的复合表达式。
#     - 如果 `type_constraint` 为 True 且 `start` 节点包含问题信息，则将该节点的标识符添加到表达式中。
#     - 遍历与 `start` 相连的节点，根据节点的功能（function）生成关系子表达式，并递归调用 `none_function` 处理子节点。
#     - 如果节点的功能包含 `<` 或 `>`，则生成相应的比较子表达式，否则生成包含 'JOIN' 的关系子表达式。
# 3. **最终的返回值：** 如果生成的子表达式为空，则返回 `start` 节点的标识符。如果生成的子表达式只有一个，则直接返回该子表达式。否则，生成包含多个子表达式的复合表达式。
# 总体而言，该函数似乎用于从图结构中生成一种特定的逻辑表达式，其中涉及了节点之间的关系和递归处理。
#     函数中有一些与图结构相关的函数调用，例如 `get_end_num`、`set_visited` 等，这些函数的具体实现可能涉及到对图结构的处理。
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


# 这段代码定义了一个名为 `get_lisp_from_graph_query` 的函数，该函数接受一个图查询 `graph_query` 作为输入，并使用该图查询构建一个有序的有向图（DiGraph）。
#     然后，根据这个图生成一个 Lisp 风格的逻辑表达式。
# 具体步骤如下：
# 1. 创建一个有向图 `G`，节点的属性包括 `id`、`type`、`question`、`function` 和 `cla`。
# 2. 遍历输入的图查询中的节点，将节点的属性添加到图中。
# 3. 遍历图查询中的边，将边的属性添加到图中，并添加一个反向的边。
# 4. 如果聚合函数为 `'count'`，则调用 `count_function` 函数生成相应的 Lisp 表达式。否则，调用 `none_function` 函数生成 Lisp 表达式。
# 函数中的 `count_function` 和 `none_function` 函数可能在前面的代码中有定义，用于生成 Lisp 表达式。
# 最终，该函数返回生成的 Lisp 表达式。
# 需要注意的是，具体的 Lisp 表达式的生成过程可能涉及到 `count_function` 和 `none_function` 的实现，这两个函数的实现在代码中并未提供。
def get_lisp_from_graph_query(graph_query):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    for node in graph_query['nodes']:
        # G.add_node(node['nid'], id=node['id'].replace('.', '/'), type=node['node_type'], question=node['question_node'], function=node['function'])
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


# 这段代码定义了一个名为 `lisp_to_sparql` 的函数，用于将 Lisp 风格的逻辑表达式转换为 SPARQL 查询语句。以下是函数的主要功能和处理步骤：
# 1. 初始化一些变量，包括 `clauses`（用于存储 SPARQL 查询的子句）、`order_clauses`（用于存储排序相关的子句）、
#     `entities`（用于收集实体以进行过滤）、`identical_variables_r`（用于替换变量）等。
# 2. 将输入的 Lisp 程序转换为嵌套表达式，以便更容易处理。
# 3. 检查是否存在 SUPERLATIVE（ARGMAX 或 ARGMIN）表达式。如果存在，设置 `superlative` 为 True，表示这是一个最值表达式。
# 4. 对 SUPERLATIVE 表达式进行处理，去除其关系链中的 JOIN 操作符，以确保 ARGMX 和 ARGMIN 在第二个参数中的关系数决定其操作数数量。
# 5. 调用 `_linearize_lisp_expression` 函数对 Lisp 表达式进行线性化，得到一系列子程序。
# 6. 获取问题变量的索引（`question_var`），这通常是线性化结果中的最后一个子程序的索引。
# 7. 设置一个变量 `count` 为 False，这个变量似乎在后续的代码中用于标记是否需要进行 COUNT 操作。
# 这段代码的后续部分可能涉及到对不同类型的 Lisp 表达式的处理，以及根据这些表达式生成相应的 SPARQL 查询。
#     由于代码的截断，无法提供后续的详细内容。如果您对具体的 SPARQL 查询生成过程或其他部分有疑问，请提出，我将尽力提供帮助。
def lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
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

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1 # get the question_var (last sub_formula_id)
    count = False


    # 这段代码的主要功能是将从生成的 Lisp 程序中得到的查询图转换为 SPARQL 查询语句。以下是代码的主要步骤和功能：
    # 1. **`get_root` 函数：** 该函数用于获取等价变量链表中给定变量的根节点。通过循环遍历链表，找到根节点。
    # 2. **迭代处理 `sub_programs` 列表：** 对于每个子表达式，根据不同的子表达式类型执行相应的处理。
    #     - **JOIN 类型处理：** 处理 JOIN 类型的子表达式，根据 JOIN 子表达式中的信息生成相应的 SPARQL 子句，并将实体添加到集合 `entities` 中。
    #     - **AND 类型处理：** 处理 AND 类型的子表达式，用于处理等价变量的关系，更新等价变量链表。
    #     - **比较操作符（le, lt, ge, gt）处理：** 生成 SPARQL 子句，包括比较的变量和值，并根据数据类型添加合适的过滤条件。
    #     - **TC（Time Constraint） 类型处理：** 处理时间约束，生成与时间有关的 SPARQL 子句。
    #     - **ARGMIN 和 ARGMAX 类型处理：** 处理超级拉丁比尔函数，生成 SPARQL 子句，并添加排序和限制条件。
    #     - **COUNT 类型处理：** 处理 COUNT 类型，用于处理问题节点的计数。
    # 3. **Merge identical variables：** 合并等价变量，使用 `identical_variables_r` 中的信息将相同的变量替换为一个标准的变量。
    # 4. **Question variable 处理：** 将问题变量替换为一个标准的变量。
    # 5. **Superlative 类型处理：** 处理超级拉丁比尔函数类型，将 SPARQL 子句合并到 `arg_clauses` 中。
    # 最终，生成的 SPARQL 查询语句由 `clauses` 和 `arg_clauses` 组成。
    def get_root(var: int):
        while var in identical_variables_r:
            if var == identical_variables_r[var]:
                break
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
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


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
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



    # 这段代码的主要功能是根据生成的 SPARQL 子句中的变量和实体信息添加一些过滤条件。具体来说，它执行以下操作：
    # 1. **获取 SPARQL 子句中的所有变量：** 使用正则表达式 `re.findall` 从每个 SPARQL 子句中提取所有的变量。
    # 2. **过滤变量：** 对于提取到的变量列表，将其中重复的变量过滤掉，并排除特定的变量（如 `?x` 和以 `?sk` 开头的变量）。
    # 3. **添加变量过滤条件：** 根据过滤得到的变量列表，为每个变量添加一个 `FILTER` 子句，确保查询结果中不包含这些变量。
    # 4. **添加实体过滤条件：** 如果存在实体信息，为每个实体添加一个 `FILTER` 子句，确保查询结果中不包含这些实体。
    # 5. **添加字符串匹配过滤条件：** 对于 SPARQL 子句中出现的字符串，添加一个 `FILTER` 子句，确保查询结果中的字符串与给定的字符串匹配。
    # 这样，通过添加这些过滤条件，可以更精细地控制生成的 SPARQL 查询结果，以满足特定的要求和约束。
    # add variable filters
    filter_variables = []
    for clause in clauses:
        variables = re.findall(r"\?\w*",clause)
        if variables:
            for var in variables:
                var = var.strip()
                if var not in filter_variables and var != '?x' and not var.startswith('?sk'):
                    filter_variables.append(var)
    ifent = True           
    for var in filter_variables:
        clauses.append(f"FILTER (?x != {var})")
        ifent = False
    if ifent:
        for entity in entities:
            clauses.append(f'FILTER (?x != ns:{entity})')
        
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

    # 这部分代码主要是在生成的 SPARQL 查询语句的前后添加一些额外的信息，以构成完整的 SPARQL 查询。具体来说，它执行以下操作：
    # 1. **设置语言过滤条件：** 在查询的开头插入一条 `FILTER` 子句，以确保返回的结果中的文本是英文的或者没有语言标签。
    # 2. **插入 WHERE 子句：** 在查询的开头插入 `WHERE {`，标志着 SPARQL 查询的主体部分的开始。
    # 3. **设置 SELECT 子句：** 根据查询类型（COUNT、超级拉丁语或普通查询），在查询的开头插入相应的 `SELECT` 子句。
    # 4. **设置 RDF 前缀：** 插入 RDF 前缀声明，这是 SPARQL 查询的标准做法。
    # 5. **插入查询结束标记：** 在查询的末尾插入 `}`，标志着 SPARQL 查询的结束。
    # 6. **插入排序子句：** 如果存在排序子句，将其插入到 SPARQL 查询的最后。
    # 通过这些插入操作，确保生成的 SPARQL 查询是符合语法规范并包含所需信息的。
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
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


# 这段代码是一个递归函数，用于线性化嵌套的 Lisp 表达式。函数的目的是将嵌套的 Lisp 表达式转换为线性的结构，
#     并用 '#' 符号标记每个子公式。函数采用递归的方式遍历嵌套的 Lisp 表达式，将其中的每个子表达式都线性化，并返回一个包含所有线性化子表达式的列表。
# 具体步骤如下：
# 1. 函数的输入参数包括嵌套的 Lisp 表达式 `expression` 和一个用于生成唯一标识符的列表 `sub_formula_id`。
# 2. 函数通过迭代嵌套表达式中的元素，对于每个元素执行以下操作：
#    - 如果当前元素是列表且不是 'R'（关系），则递归调用 `_linearize_lisp_expression` 函数，将子表达式线性化，并将返回的线性化子表达式列表拼接到结果中。
#     同时，将当前元素替换为 '#' 后接一个唯一标识符，表示该子表达式的位置。
#    - 如果当前元素不是列表，保持不变。
# 3. 将当前线性化的子表达式加入到结果列表中。
# 4. 更新唯一标识符，确保每个子表达式都有唯一的标识符。
# 5. 返回包含所有线性化子表达式的列表。
# 最终，这个函数的目的是为了便于后续处理，将嵌套的 Lisp 表达式转换为线性的结构，并用唯一标识符标记每个子表达式的位置。
# linearize nested lisp exressions
def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas

# 这段代码是一个将 Lisp 表达式转换为 lambda DCS（Dependency-based Compositional Semantics）的函数。Lambda DCS 是一种语义表示形式，用于表示自然语言中的语义结构。
# 函数的输入是一个 Lisp 表达式，它通过递归的方式处理嵌套的表达式，并将其转换为 lambda DCS 表达式。当前函数支持两种基本的操作：'AND' 和 'JOIN'。
# - 如果当前表达式的操作是 'AND'，则递归地将左右两个子表达式转换为 lambda DCS，并用 'AND' 连接。
# - 如果当前表达式的操作是 'JOIN'，则递归地将左右两个子表达式转换为 lambda DCS，并用 '*' 连接。
# 这个函数目前只处理 'AND' 和 'JOIN' 两种操作，如果 Lisp 表达式中还包含其他操作，可能需要根据具体情况进行扩展。函数的返回值是一个 lambda DCS 表达式的字符串。
# 需要注意的是，这段代码只展示了处理 'AND' 和 'JOIN' 的部分，如果 Lisp 表达式中还包含其他操作或更复杂的结构，可能需要根据具体情况进行扩展和修改。
#  I don't think this is ever gonna be implemented
def lisp_to_lambda(expressions: Union[List[str], str]):  # from lisp-grammar formula to lambda DCS
    # expressions = lisp_to_nested_expression(source_formula)
    if not isinstance(expressions, list):
        return expressions
    if expressions[0] == 'AND':
        return lisp_to_lambda(expressions[1]) + ' AND ' + lisp_to_lambda(expressions[2])
    elif expressions[0] == 'JOIN':
        return lisp_to_lambda(expressions[1]) + '*' + lisp_to_lambda(expressions[2])


# 这段代码的目的是将一个Semantic Query Language（SEAL）表达式（`sexpr`）转换为SPARQL查询，然后执行它。
# 具体解释如下：
# 1. `sexpr` 变量包含一个SEAL表达式，这个表达式以S表达式的形式表示。SEAL是用于查询结构化数据的查询语言。
# 2. 代码假设存在两个函数：`lisp_to_sparql` 和 `execute_query`。
#    - `lisp_to_sparql(sexpr)`：该函数预期接受一个SEAL表达式（在本例中以S表达式表示）并将其转换为SPARQL查询。SPARQL是一种用于查询存储在RDF格式中的数据的查询语言。
#    - `execute_query(sparql)`：该函数预期接受一个SPARQL查询并执行它。查询执行的结果存储在变量 `res` 中。
# 3. 使用 `print(sparql)` 打印转换后的SPARQL查询。
# 4. 执行SPARQL查询，将结果使用 `print(res)` 打印出来。
# 然而，正如我之前提到的，`lisp_to_sparql` 和 `execute_query` 函数的实现在提供的代码片段中未给出。
#     要完全理解和运行此代码，您需要在代码库的其他地方实现这两个功能，或者从提供SEAL到SPARQL转换和SPARQL查询执行功能的库中获取它们。
if __name__=='__main__':
    
    # gt_sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN government.government_position_held.office_position_or_title m.0j5wjnc) (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf)))) government.government_position_held.from NOW))'

    sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf))) government.government_position_held.from now))'
    sparql = lisp_to_sparql(sexpr)
    print(sparql)
    res = execute_query(sparql)
    print(res)



