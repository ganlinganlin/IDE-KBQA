
# 这段代码导入了一些 Python 标准库和自定义模块。以下是每个导入语句的简要解释：
# 1. `import re`: 导入 Python 的正则表达式模块，用于支持正则表达式相关的操作。
# 2. `import os`: 导入 Python 的操作系统接口模块，用于与操作系统交互，比如文件和文件夹的路径处理等。
# 3. `from tqdm import tqdm`: 从 `tqdm` 模块中导入 `tqdm` 类，用于在终端显示循环的进度条。这对于处理大量数据或执行长时间任务时提供了可视化的进度信息。
# 4. `from components.utils import *`: 从自定义的 `components.utils` 模块中导入所有内容。这可能包括一些自定义的实用工具函数或类，用于在代码中执行特定任务。
# 5. `from components.expr_parser import parse_s_expr`: 从自定义的 `components.expr_parser` 模块中导入 `parse_s_expr` 函数。这个函数可能用于解析表达式。
# 6. `from executor.sparql_executor import execute_query, execute_query_with_odbc`:
#     从自定义的 `executor.sparql_executor` 模块中导入 `execute_query` 和 `execute_query_with_odbc` 函数。这些函数可能用于执行 SPARQL 查询。
# 这些导入语句表明代码可能涉及到正则表达式、操作系统交互、进度条显示、自定义工具函数、表达式解析以及 SPARQL 查询执行等功能。
import re
import os
from tqdm import tqdm
from components.utils import *
from components.expr_parser import parse_s_expr
from executor.sparql_executor import execute_query, execute_query_with_odbc
from executor.logic_form_util import lisp_to_sparql


# ParseError 是一个异常类，它继承自 Python 内置的 Exception 类。
#     这样的异常类通常用于在程序执行过程中发生错误或无法正常处理的情况下引发异常。在这里，ParseError 用于表示解析过程中的错误。
class ParseError(Exception):
    pass


class Parser:
    def __init__(self):
        pass

    # 这是一个用于解析SPARQL查询的Python函数。以下是对该函数的逐行解释：
    # 1. `lines = query.split('\n')`: 将输入的SPARQL查询字符串按行拆分成列表。
    # 2. `lines = [x for x in lines if x]`: 移除空行，得到一个不包含空行的列表。
    # 3. `assert lines[0] != '#MANUAL SPARQL'`: 断言确保查询的第一行不是以'#MANUAL SPARQL'开头，否则会触发`AssertionError`。
    def parse_query_webqsp(self, query, mid_list):
        """parse a sparql query into a s-expression

        @param query: sparql query
        @param mid_list: all mids appeared in the sparql query
        """
        # print('QUERY', query)
        lines = query.split('\n')
        lines = [x for x in lines if x]

        assert lines[0] != '#MANUAL SPARQL'

        # 4. `prefix_stmts = []`: 初始化一个空列表，用于存储查询中的PREFIX语句。
        # 5. `line_num = 0`: 初始化行号为0。
        # 6. `while True:`: 进入一个无限循环。
        # 7. `l = lines[line_num]`: 获取当前行。
        # 8. `if l.startswith('PREFIX'):`: 如果当前行以'PREFIX'开头，将该行添加到`prefix_stmts`列表中。
        # 9. `else: break`: 如果不是以'PREFIX'开头，则跳出循环。
        # 10. `line_num = line_num + 1`: 行号加1，继续下一行。
        prefix_stmts = []
        line_num = 0
        while True:
            l = lines[line_num]
            if l.startswith('PREFIX'):
                prefix_stmts.append(l)
            else:
                break
            line_num = line_num + 1

        # 11. `next_line = lines[line_num]`: 获取下一行。
        # 12. `assert next_line.startswith('SELECT DISTINCT ?x')`: 断言确保下一行以'SELECT DISTINCT ?x'开头，否则触发`AssertionError`。
        # 13. `line_num = line_num + 1`: 行号加1。
        # 14. `next_line = lines[line_num]`: 获取下一行。
        # 15. `assert next_line == 'WHERE {'`: 断言确保下一行是'WHERE {'，否则触发`AssertionError`。
        next_line = lines[line_num]
        assert next_line.startswith('SELECT DISTINCT ?x')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'WHERE {'

        # 16. 接下来是一些处理ORDER BY、LIMIT、OFFSET等特殊情况的逻辑。
        # 17. `assert lines[-1] in ['}', 'LIMIT 1']`: 断言确保最后一行是'}'或'LIMIT 1'，否则触发`AssertionError`。
        # 18. `lines = lines[line_num:]`: 将剩余的行赋给`lines`，截取了从`WHERE {`开始的部分。
        # 19. `filter_string_flag = not all(['FILTER (str' not in x for x in lines])`: 判断是否存在包含'FILTER (str'的行，根据结果设置`filter_string_flag`标志。
        if re.match(r'ORDER BY .*\?\w*.* LIMIT 1', lines[-1]):
            lines[-1] = lines[-1].replace('LIMIT 1', '').strip()
            lines.append('LIMIT 1')
        
        if re.match(r'LIMIT \d*', lines[-1]): # TODO LIMIT n 
            lines[-1]='LIMIT 1' # transform to LIMIT 1, temporally
        
        if lines[-1].startswith('OFFSET'): # TODO LITMIT 1 \n OFFSET 1 ('the second ...')
            lines.pop(-1) # transform to LIMIT 1, temporally

        assert lines[-1] in ['}', 'LIMIT 1']

        lines = lines[line_num:]

        filter_string_flag = not all(['FILTER (str' not in x for x in lines])

        # assert all(['FILTER (str' not in x for x in lines])


        # 20. 调用`self.normalize_body_lines`方法对主体行进行标准化处理，得到`body_lines`、`spec_condition`和`filter_lines`。
        # 21. `body_lines = body_lines[2:]`: 如果主体的第一行是以'FILTER'开头的，去掉前两行。
        # normalize body lines
        body_lines, spec_condition, filter_lines = self.normalize_body_lines(lines, filter_string_flag)
        body_lines = [x.strip() for x in body_lines]  # strip spaces
        # assert all([x.startswith('?') or x.startswith('ns') or x.startswith('FILTER') for x in body_lines])
        # we only parse query following this format
        if body_lines[0].startswith('FILTER'):
            predefined_filter0 = body_lines[0]
            predefined_filter1 = body_lines[1]

            # filter_0_line validation
            filter_0_valid = (predefined_filter0 == f'FILTER (?x != ?c)')
            if not filter_0_valid:
                for mid in mid_list:
                    filter_0_valid = filter_0_valid or (
                        predefined_filter0 == f'FILTER (?x != {mid})')

            assert filter_0_valid

            # filter_1_line validation
            assert predefined_filter1 == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
            # if predefined_filter0 != f'FILTER (?x != ns:{topic_mid})':
            #     print('QUERY', query)
            #     print('First Filter')
            # if predefined_filter1 != "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
            #     print('QUERY', query)
            #     print('Second Filter')
            # if any([not (x.startswith('?') or x.startswith('ns:')) for x in body_lines]):
            #     print('Unprincipled Filter')
            #     print('QUERY', query)
            body_lines = body_lines[2:]

        # body line form assertion
        # 22. `assert all([(x.startswith('?') or x.startswith('ns:')) for x in body_lines])`: 断言确保主体行的每一行要么以'?'或'ns:'开头，否则触发`AssertionError`。
        assert all([(x.startswith('?') or x.startswith('ns:'))
                   for x in body_lines])
        # print(body_lines)

        # 23. 调用`self.parse_naive_body`方法，解析主体行，得到变量依赖列表`var_dep_list`。
        # 24. 调用`self.dep_graph_to_s_expr`方法，将变量依赖列表转换为s表达式，最终返回该s表达式。
        # 这个函数的目的是将输入的SPARQL查询字符串解析为s表达式，并且包含了一些对特殊情况的处理逻辑。如果您有特定的问题或需要更详细的解释，请随时提问。
        var_dep_list = self.parse_naive_body(body_lines, filter_lines, '?x', spec_condition)
        s_expr = self.dep_graph_to_s_expr(var_dep_list, '?x', spec_condition)
        return s_expr


    # 这是一个名为`normalize_body_lines`的方法，该方法接收一个字符串列表`lines`和一个布尔参数`filter_string_flag`，并返回处理过的主体行、特殊条件和过滤行。
    # 函数的主要功能包括：
    # 1. 处理`FILTER (str`的情况：
    #    - 如果存在`FILTER (str`的行，将其存储在`filter_lines`中，然后从`lines`中移除这些行。
    #    - 如果不存在，直接将`lines`中的每一行去除首尾空格后存储在`lines`中。
    # 2. 获取比较行：
    #    - 通过正则表达式匹配获取包含比较操作的行（例如，`FILTER (?num > "2009-01-02"^^xsd:dateTime)`）。
    #    - 解析比较行，获取变量、操作符和比较值，存储为一个条件。
    # 3. 处理`FILTER(NOT EXISTS`的情况：
    #    - 查找包含`FILTER(NOT EXISTS`的行，标记起始和结束位置。
    #    - 如果存在，剔除这些行，处理多余的范围过滤器，存储为一个条件。
    # 4. 获取`LIMIT 1`和`argmin`/`argmax`行：
    #    - 如果最后一行是`LIMIT 1`，则解析倒数第二行，获取`argmin`或`argmax`条件。
    # 5. 处理范围过滤器：
    #    - 如果存在范围过滤器，解析范围过滤器，获取关系、变量、起始时间和结束时间，存储为一个条件。
    # 最终，函数返回处理后的主体行`body_lines`、特殊条件`spec_condition`和过滤行`filter_lines`。
    # 这个方法的目的是对SPARQL查询的主体部分进行标准化处理，提取其中的条件和过滤器。
    def normalize_body_lines(self, lines, filter_string_flag=False):
        """return normalized body lines of sparql, specially return filter lines starting with `FILTER (str(`        

        @param lines: sparql lines list
        @param filter_string_flag: flag indicates existence of filter lines


        @return: (body_lines,
                    spec_condition,
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]
                    filter_lines
                  )
        """

        spec_condition = []

        # 1. get literal filter_lines
        # ?x ns:base.biblioness.bibs_location.loc_type ?sk0 .
        # FILTER (str(?sk0) = "Country")
        if filter_string_flag:
            filter_lines = [x.strip() for x in lines if 'FILTER (str' in x]
            lines = [x.strip() for x in lines if 'FILTER (str' not in x]
        else:
            lines = [x.strip() for x in lines]
            filter_lines = None
        
        # 2. get compare lines
        # 2.1 FILTER (?num > "2009-01-02"^^xsd:dateTime) .
        # 2.2 FILTER (xsd:integer(?num) < 33351310952) . 
        if re.match(r'FILTER \(\?\w* (>|<|>=|<=) .*',lines[-2]) \
            or re.match(r'FILTER \(xsd:integer\(\?\w*\) (>|<|>=|<=) .*',lines[-2]):
            
            compare_line = lines.pop(-2)
            compare_var = re.findall(r'\?\w*',compare_line)[0]
            compare_operator = re.findall(r'(>|>=|<|<=)',compare_line)[0]
            operator_mapper = {'<':'lt','<=':'le','>':'gt',">=":"ge"}
            if "^^xsd:dateTime" in compare_line: # dateTime
                compare_value = re.findall(r'".*"\^\^xsd:dateTime',compare_line)[0]
            else: # number
                compare_value = compare_line.replace(") .","").split(" ")[-1]

            compare_value = compare_value.replace('"','') # remove \" in compare value
            # print(variable,compare_operator,compare_value)
            compare_condition = ['COMPARATIVE', operator_mapper[compare_operator],compare_var,compare_value]
            spec_condition.append(compare_condition)
            
        # 3. get range lines, move to the end of where clause
        # WHERE {
            # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
            # ?y ns:government.government_position_held.office_holder ?x .
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
            # EXISTS {?y ns:government.government_position_held.from ?sk1 .
            # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
            # EXISTS {?y ns:government.government_position_held.to ?sk3 .
            # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
            # }
        start_line = -1
        right_parantheses_line = -1
        not_exists_num = 0
        for i, line in enumerate(lines):
            if line.startswith("FILTER(NOT EXISTS"):
                not_exists_num +=1
                if start_line == -1:
                    start_line = i
            # if line.startswith("FILTER(") and "2015-08-10" in line and start_line != -1:
            #     meaningless_time_flag = True
            if line == '}':
                right_parantheses_line = i

        if start_line != -1:
            
            if not_exists_num==4: # redundant range filters
                end_line = start_line+12
            else:
                end_line = start_line+6
            
            assert end_line <= right_parantheses_line
            
            if end_line==start_line+12: # discard redundant range filters
                lines = lines[:start_line]+lines[end_line:right_parantheses_line] + \
                        lines[start_line:end_line-6]+lines[right_parantheses_line:]
            else:
                lines = lines[:start_line]+lines[end_line:right_parantheses_line] + \
                        lines[start_line:end_line]+lines[right_parantheses_line:]
                    
           
        # 4. get SUPERLATIVE lines
        body_lines = []
        if lines[-1] == 'LIMIT 1':
            # spec_condition = argmax
            # who did jackie robinson first play for?
            # WHERE {
            # ns:m.0443c ns:sports.pro_athlete.teams ?y .
            # ?y ns:sports.sports_team_roster.team ?x .
            # ?y ns:sports.sports_team_roster.from ?sk0 .
            # }
            # ORDER BY DESC(xsd:datetime(?sk0))
            # LIMIT 1
            order_line = lines[-2]
            direction = 'argmax' if 'DESC(' in order_line else 'argmin'
            compare_var = re.findall(r'\?\w*', order_line)[0]
            # assert ('?sk0' in order_line) # variable in order_line
            assert(compare_var in order_line)

            _tmp_body_lines = lines[1:-3]
            
            hit = False
            for l in _tmp_body_lines:
                if compare_var in l :
                    if 'FILTER' in l: # the return var is also the argmax var, not covered by S-Expression
                        assert 1==2 # raise AssertionError
                    # self.parse_assert(l.endswith('?sk0 .') and not hit)
                    self.parse_assert(l.endswith(compare_var+" .")
                                      and not hit)  # appear only once
                    hit = True
                    arg_var, arg_r = l.split(' ')[0], l.split(' ')[1]
                    arg_r = arg_r[3:]  # rm ns:
                else:
                    body_lines.append(l)

            superlative_cond = ['SUPERLATIVE',direction,arg_var,arg_r]
            spec_condition.append(superlative_cond)
        
            # if not lines[-4].startswith('FILTER(NOT EXISTS {?'):
            #     if filter_string_flag:
            #         return body_lines, [direction, arg_var, arg_r], filter_lines
            #     else:
            #         return body_lines, [direction, arg_var, arg_r], None
            # else:
            #     # contains range constraints FILTER
            #     pass

        # 4. process range lines
        if body_lines: # already processed by superlative extraction
            lines = body_lines
            range_line_num = -6
        else:
            range_line_num = -7
        if len(lines)>= abs(range_line_num) and lines[range_line_num].startswith('FILTER(NOT EXISTS {?'):
            # WHERE {
            # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
            # ?y ns:government.government_position_held.office_holder ?x .
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
            # EXISTS {?y ns:government.government_position_held.from ?sk1 .
            # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
            # EXISTS {?y ns:government.government_position_held.to ?sk3 .
            # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
            # }
            if not body_lines:
                body_lines = lines[1:-7]
                range_lines = lines[-7:-1]
            else:
                body_lines = lines[:-6]
                range_lines = lines[-6:]
            range_prompt = range_lines[0]
            range_prompt = range_prompt[range_prompt.index(
                '{') + 1:range_prompt.index('}')]
            range_var = range_prompt.split(' ')[0]
            range_relation = range_prompt.split(' ')[1]
            # range_relation = '.'.join(
            #     range_relation.split('.')[:2]) + '.time_macro'
            range_relation = range_relation[3:]  # rm ns:
            range_start_time = re.findall(f'".*"\^\^',range_lines[2])[0].split("^^")[0].strip('"')
            if range_start_time =='2015-08-10':
                range_start_time = 'NOW'
            range_start = range_lines[2].split(' ')[2]
            range_start = range_start[1:]
            range_start = range_start[:range_start.index('"')]
            
            range_end = range_lines[5].split(' ')[2]
            range_end = range_end[1:]
            range_end = range_end[:range_end.index('"')]

            # assert range_start[:4] == range_end[:4]
            # to fit parsable
            # range_year = range_start[:4] + \
            #     '^^http://www.w3.org/2001/XMLSchema#dateTime' if range_start_time != 'NOW' else 'NOW'
            range_year = range_start[:4] if range_start_time != 'NOW' else 'NOW'
            range_start_cond = ['RANGE', range_relation, range_var, range_year]
            spec_condition.append(range_start_cond)
            
            # if filter_string_flag:
            #     return body_lines, ['range', range_var, range_relation, range_year], filter_lines
            # else:
            #     return body_lines, ['range', range_var, range_relation, range_year], None
        
        # body_lines not extracted yet
        if not body_lines: 
            body_lines = lines[1:-1]
            # if filter_string_flag:
            #     return body_lines, None, filter_lines
            # else:
            #     return body_lines, None, None
        
        return body_lines, spec_condition, filter_lines


# 这是一个名为`dep_graph_to_s_expr`的方法，用于将依赖图转换为s表达式。以下是对该方法的逐行解释：
# 1. `self.parse_assert(var_dep_list[0][0] == ret_var)`: 断言确保变量依赖列表的第一个元素的第一个元素（即列表中的第一个变量）等于返回变量 `ret_var`。
# 2. `var_dep_list.reverse()`: 反转变量依赖列表，以便按照依赖关系的顺序进行处理。
# 3. `parsed_dict = {}`: 初始化一个字典，用于存储已解析的变量及其对应的表达式。
# 4. `spec_var_map = {cond[2]:i for i,cond in enumerate(spec_condition)} if spec_condition else None`:
#     如果存在特殊条件 `spec_condition`，则创建一个字典 `spec_var_map`，将特殊条件中的变量映射到其在 `spec_condition` 中的索引。
# 5. `for var_name, dep_relations in var_dep_list:`: 遍历变量依赖列表。
#    a. `clause = self.triplet_to_clause(var_name,  dep_relations[0], parsed_dict)`:
#     调用 `triplet_to_clause` 方法，将第一个三元组转换为表达式，并存储在 `clause` 中。
#    b. `for tri in dep_relations[1:]:`: 遍历变量的其他依赖关系。
#       i. `n_clause = self.triplet_to_clause(var_name, tri, parsed_dict)`: 调用 `triplet_to_clause` 方法，将每个三元组转换为表达式。
#       ii. `clause = 'AND ({}) ({})'.format(n_clause, clause)`: 使用 'AND' 运算符将新的表达式与先前的表达式连接。
#    c. 如果变量名在 `spec_var_map` 中，表示该变量具有特殊条件：
#       i. `cond = spec_condition[spec_var_map[var_name]]`: 获取该变量对应的特殊条件。
#       ii. 根据特殊条件的类型进行处理：
#          - 如果是超级拉丁条件（`'SUPERLATIVE'`）：
#            - `relation = cond[3]`: 获取超级拉丁条件中的关系。
#            - `clause = '{} ({}) {}'.format(cond[1].upper(), clause, relation)`: 根据条件类型将超级拉丁条件添加到表达式中。
#          - 如果是范围条件（`'RANGE'`）：
#            - `relation, time_point = cond[1], cond[3]`: 获取范围条件中的关系和时间点。
#            - `clause = 'TC ({}) {} {}'.format(clause, relation, time_point)`: 根据条件类型将范围条件添加到表达式中。
#          - 如果是比较条件（`'COMPARATIVE'`）：
#            - `op, value = cond[1], cond[3]`: 获取比较条件中的操作符和值。
#            - `rel = cond[4]`: 获取比较条件中的关系。
#            - `n_clause = '{} {} {}'.format(op, rel, value)`: 构造比较条件的子句。
#            - `clause = 'AND ({}) ({})'.format(n_clause, clause)`: 将比较条件添加到表达式中。
#    d. `parsed_dict[var_name] = clause`: 将变量及其对应的表达式存储在 `parsed_dict` 中。
# 6. `res = '(' + parsed_dict[ret_var] + ')'`: 获取返回变量对应的表达式，并添加括号。
# 7. `res = res.replace('xsd:','http://www.w3.org/2001/XMLSchema#')`: 将表达式中的命名空间 `xsd:` 替换为完整的命名空间 `http://www.w3.org/2001/XMLSchema#`。
# 8. 返回最终的s表达式 `res`。
# 该方法的目的是将依赖图转换为s表达式，同时处理了特殊条件，包括超级拉丁条件、范围条件和比较条件。
    def dep_graph_to_s_expr(self, var_dep_list, ret_var, spec_condition=None):
        """Convert dependancy graph to s_expression
        @param var_dep_list: varialbe dependancy list
        @param ret_var: return var
        @param spec_condition: special condition

        @return s_expression
        """
        self.parse_assert(var_dep_list[0][0] == ret_var)
        var_dep_list.reverse() # reverse the var_dep_list
        parsed_dict = {}  # dict for parsed variables

        # spec_condition,
        #             # [
        #             #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
        #             #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value],
        #             #     ['RANGE', range_relation, range_var, range_year],
        #             # ]

        # specical condition var map {spec_var:idx in spec_condition}
        spec_var_map = {cond[2]:i for i,cond in enumerate(spec_condition)} if spec_condition else None
        # spec_var = spec_condition[1] if spec_condition is not None else None

        for var_name, dep_relations in var_dep_list:
            # expr = ''
            dep_relations[0]
            clause = self.triplet_to_clause(
                var_name,  dep_relations[0], parsed_dict)
            for tri in dep_relations[1:]:
                n_clause = self.triplet_to_clause(var_name, tri, parsed_dict)
                clause = 'AND ({}) ({})'.format(n_clause, clause)
            # if var_name == spec_var:
            if spec_var_map and var_name in spec_var_map: # spec_condition
                cond = spec_condition[spec_var_map[var_name]]
                # if cond[0] == 'argmax' or cond[0] == 'argmin': # superlative
                if cond[0]=='SUPERLATIVE':
                    #relation = spec_condition[2]
                    relation = cond[3]
                    clause = '{} ({}) {}'.format(
                        cond[1].upper(), clause, relation)
                elif cond[0] == 'RANGE':
                    relation, time_point = cond[1], cond[3]
                    clause = 'TC ({}) {} {}'.format(clause, relation, time_point)
                    # n_clause = 'TC {} {}'.format(relation, time_point)
                    # clause = 'AND ({}) ({})'.format(n_clause, clause)
                elif cond[0] == 'COMPARATIVE':
                    op = cond[1]
                    value = cond[3]
                    rel = cond[4]
                    n_clause = '{} {} {}'.format(op, rel, value)
                    clause = 'AND ({}) ({})'.format(n_clause, clause)
                    # pass
            parsed_dict[var_name] = clause
        
        res = '(' + parsed_dict[ret_var] + ')'
        res = res.replace('xsd:','http://www.w3.org/2001/XMLSchema#')
        return res


# 这是一个名为 `triplet_to_clause` 的方法，用于将三元组转换为S表达式子句。以下是对该方法的逐行解释：
# 1. `if triplet[0] == tgt_var:`: 如果三元组的第一个元素等于目标变量 `tgt_var`。
#    a. `this = triplet[0]`: 将目标变量设置为三元组的第一个元素。
#    b. `other = triplet[-1]`: 将与目标变量连接的变量设置为三元组的最后一个元素。
#    c. `if other in parsed_dict: other = '(' + parsed_dict[other] + ')'`: 如果连接的变量已经被解析，将其替换为已解析的表达式。
#    d. 返回形如 `'JOIN {} {}'.format(triplet[1], other)` 的S表达式子句，表示目标变量与连接变量之间的关系。
# 2. `elif triplet[-1] == tgt_var:`: 如果三元组的最后一个元素等于目标变量 `tgt_var`。
#    a. `this = triplet[-1]`: 将目标变量设置为三元组的最后一个元素。
#    b. `other = triplet[0]`: 将与目标变量连接的变量设置为三元组的第一个元素。
#    c. `if other in parsed_dict: other = '(' + parsed_dict[other] + ')'`: 如果连接的变量已经被解析，将其替换为已解析的表达式。
#    d. 返回形如 `'JOIN (R {}) {}'.format(triplet[1], other)` 的S表达式子句，表示目标变量与连接变量之间的逆向关系。
# 3. `else: raise ParseError()`: 如果目标变量既不是三元组的第一个元素，也不是最后一个元素，则抛出 `ParseError` 异常。
# 该方法的目的是根据目标变量与三元组的关系，将三元组转换为S表达式子句。如果目标变量是三元组的起始变量，则使用 `'JOIN {} {}'.format(triplet[1], other)` 形式的S表达式，
#     表示目标变量与连接变量之间的关系。如果目标变量是三元组的终止变量，则使用 `'JOIN (R {}) {}'.format(triplet[1], other)` 形式的S表达式，
#     表示目标变量与连接变量之间的逆向关系。如果目标变量既不是起始变量也不是终止变量，则抛出异常。
    def triplet_to_clause(self, tgt_var, triplet, parsed_dict):
        """Convert a triplet to S_expression clause
        @param tgt_var: target variable
        @param triplet: triplet in sparql
        @param parsed_dict: dict for variables already parsed
        """
        if triplet[0] == tgt_var:
            this = triplet[0]
            other = triplet[-1]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN {} {}'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            this = triplet[-1]
            other = triplet[0]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN (R {}) {}'.format(triplet[1], other)
        else:
            raise ParseError()


    # 这是一个名为 `parse_assert` 的方法，用于进行解析时的断言。方法接收一个表达式 `eval`，如果该表达式的值为假，则抛出 `ParseError` 异常。
    # 具体步骤如下：
    # 1. `if not eval:`: 如果表达式的值为假。
    # 2. `raise ParseError()`: 抛出 `ParseError` 异常，表示解析失败。
    # 该方法的作用是在解析过程中进行断言，如果某个条件不符合预期，就会抛出异常，用于捕获和处理解析错误。
    def parse_assert(self, eval):
        if not eval:
            raise ParseError()

    # 这是一个名为 `parse_naive_body` 的方法，用于解析SPARQL查询的主体部分。以下是对该方法的逐行解释：
    # 1. `assert all([x[-1] == '.' for x in body_lines])`: 断言所有的主体行都以句点结尾，确保语法正确。
    # 2. `if filter_lines: assert all(['FILTER (str' in x for x in filter_lines])`:
    #   如果存在过滤器行，则断言所有过滤器行都以 `'FILTER (str'` 开头，确保过滤器语法正确。
    # 3. `triplets = [x.replace('"','') if "^^xsd:" in x else x for x in body_lines]`: 如果三元组中包含 `"^^xsd:"`，则去掉双引号。
    # 4. `triplets = [x.split(' ') for x in triplets]`: 将每个三元组按空格拆分为列表。
    # 5. `triplets = [x[:2] + [" ".join(x[2:-1]), x[-1]] if len(x)>4 else x for x in triplets]`: 如果三元组长度大于4，将第三个元素及其之后的元素合并为一个字符串。
    # 6. `triplets = [x[:-1] if x[-1] == '.' else x for x in triplets]`: 如果三元组以句点结尾，则去掉句点。
    # 这些步骤主要是对主体行进行预处理，以确保其格式符合期望，便于后续解析。
    # 接下来的步骤可能包含解析三元组和处理特殊条件的逻辑，但在提供的代码片段中未提供完整的实现。如果您有关于接下来代码的具体问题或需要更多详细信息，请告诉我，我将尽力提供帮助。
    def parse_naive_body(self, body_lines, filter_lines, ret_var, spec_condition=None):
        """Parse body lines
        @param body_lines: list of sparql body lines
        @param filter_lines: lines that start with `FILTER (str(?`
        @param ret_var: return var, default `?x`
        @param spec_condition: spec_condition like
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]

        @return: variable dependancy list
        """
        # ret_variable
        # body_lines
        assert all([x[-1] == '.' for x in body_lines])
        # filter lines assertion
        if filter_lines:
            assert all(['FILTER (str' in x for x in filter_lines])


        triplets = [x.replace('"','') if "^^xsd:" in x else x for x in body_lines]
        triplets = [x.split(' ') for x in triplets]  # split by '
                
        triplets = [x[:2] + [" ".join(x[2:-1]), x[-1]] if len(x)>4 else x for x in triplets] # avoid error splitting like "2100 Woodward Avenue"@en
        triplets = [x[:-1] if x[-1] == '.' else x for x in triplets]  # remove '.'
        
        

        # 这是 `parse_naive_body` 方法的继续部分。在这里，代码处理了主体行的剩余部分，主要包括解析三元组的变量依赖关系以及处理特殊条件（`spec_condition`）的逻辑。
        # 以下是代码的逐行解释：
        # 1. `triplets = [[x[3:] if x.startswith('ns:') else x for x in tri] for tri in triplets]`: 移除命名空间前缀 `ns:`。
        # 2. `triplets_pool = triplets`: 将三元组保存在 `triplets_pool` 中备用。
        # 3. `var_dep_list = []`: 初始化变量依赖列表。
        # 4. `successors = []`: 初始化后继变量列表。
        # 5. 处理返回变量（`ret_var`）：
        #    - `dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, filter_lines, ret_var, successors)`:
        #     调用 `resolve_dependancy` 方法解析返回变量的依赖关系。
        #    - `var_dep_list.append((ret_var, dep_triplets))`: 将返回变量和其依赖关系添加到变量依赖列表中。
        # 6. 处理所有后继变量：
        #    - `while len(successors):`: 当存在后继变量时，进行循环处理。
        #    - `tgt_var = successors[0]`: 取出当前处理的后继变量。
        #    - `dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, filter_lines, tgt_var, successors)`:
        #     调用 `resolve_dependancy` 方法解析后继变量的依赖关系。
        #    - `if len(dep_triplets) == 0:`: 如果依赖关系为空，可能是一个2-hop约束，通过检查 `spec_condition` 进行处理。
        #    - `else:`: 如果有依赖关系，将变量和其依赖关系添加到变量依赖列表中。
        # 7. `if(len(triplets_pool) != 0): print(triplets_pool)`: 检查是否所有的三元组都被处理，如果不是，则输出剩余的三元组。
        # 8. `self.parse_assert(len(triplets_pool) == 0)`: 断言所有的三元组都已经被处理，否则抛出异常。
        # 9. `return var_dep_list`: 返回变量依赖列表。
        # 这部分代码的核心工作是通过解析三元组构建变量依赖关系图，同时处理了特殊条件（`spec_condition`）的逻辑。
        #     如果您有关于这部分代码的具体问题或需要更多详细信息，请告诉我，我将尽力提供帮助。
        # remove ns
        triplets = [[x[3:] if x.startswith(
            'ns:') else x for x in tri] for tri in triplets]
        # dependancy graph
        triplets_pool = triplets
        # while True:
        # varaible dependancy list, in the form like [(?x,[['?x','ns:aaa.aaa.aaa','?y'],['ns:m.xx','ns:bbb.bbb.bbb','?x''])]
        var_dep_list = []
        successors = []

        # firstly solve the return variable
        dep_triplets, triplets_pool = self.resolve_dependancy(
            triplets_pool, filter_lines, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))
        # vars_pool = []
        # go over un resolved vars
        # for tri in triplets_pool:
        #     if tri[0].startswith('?') and tri[0] not in vars_pool and tri[0] != ret_var:
        #         vars_pool.append(tri[0])
        #     if tri[-1].startswith('?') and tri[-1] not in vars_pool and tri[-1] != ret_var:
        #         vars_pool.append(tri[-1])

        # for tgt_var in vars_pool:
        #     dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, tgt_var)
        #     self.parse_assert(len(dep_triplets) > 0)
        #     var_dep_list.append((tgt_var, dep_triplets))

        # handle all the successor variables
        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = self.resolve_dependancy(
                triplets_pool, filter_lines, tgt_var, successors)

            # if (len(dep_triplets)==0):
            #     # no triplet for tgt_var
            #     # ?x ns:xxx ?c
            #     # ?c ns:xxx ?num
            #     # ORDER BY ?num LIMIT 1
            #     print(dep_triplets)

            # assert len(dep_triplets) > 0 # at least one dependancy triplets
            if len(dep_triplets) == 0:
                # zero dep_triples, can be a 2-hop constraint
                # e.g.
                # 'ns:m.0d0x8 ns:government.political_district.representatives ?y .'
                # '?y ns:government.government_position_held.office_holder ?x .'
                # '?y ns:government.government_position_held.governmental_body ns:m.07t58 .'
                # '?x ns:government.politician.government_positions_held ?c .'
                
                if spec_condition and any([tgt_var in x for x in spec_condition]):
                    cond = []
                    for x in spec_condition:
                        if tgt_var in x:
                            cond = x
                            break
                    
                    repeat = True
                    while repeat:        
                        # tgt_var is a var in spec_condition
                        for (var, triplets) in var_dep_list:
                            if any([tgt_var in trip for trip in triplets]):
                                head_var = var  # find the real constrained var
                                _temp_triplets = triplets[:]
                                triplets.clear()
                                for trip in _temp_triplets:
                                    if tgt_var not in trip:
                                        triplets.append(trip)
                                    else:
                                        # find the constraint relation
                                        cons_rel = trip[1]
                                        if trip[0] == head_var:
                                            reversed_direction = False
                                        else:
                                            reversed_direction = True
                                        cons_rel = f'(R {cons_rel})' if reversed_direction else cons_rel

                                # modify spec_condition
                                # spec_condition[1] = head_var
                                if cond[0]=='COMPARATIVE':
                                    cond[2] = head_var
                                    if len(cond)<5:
                                        cond.append(cons_rel)
                                    else:
                                        cond[4] = "(JOIN " + cons_rel+" "+ cond[4]+")"
                                else: # SUPERLATIVE
                                    cond[2] = head_var
                                    cond[3] = "(JOIN "+ cons_rel+" "+cond[3]+")"
                                tgt_var = head_var
                        
                        # check whether need to repeat
                        remove_idx=-1
                        for i,(var,triplets) in enumerate(var_dep_list):
                            if var == head_var:
                                if len(triplets)==0:
                                    repeat = True
                                    remove_idx = i
                                else:
                                    repeat = False
                                break
                        
                        if remove_idx>=0:
                            var_dep_list.pop(remove_idx)
                        else:
                            repeat=False
         
                else:
                    # uncovered situation
                    assert 1 == 2
            else:
                """dep_triplets not None"""
                self.parse_assert(len(dep_triplets) > 0)  # at least one dependancy triplets
                var_dep_list.append((tgt_var, dep_triplets))

        if(len(triplets_pool) != 0):
            print(triplets_pool)

        self.parse_assert(len(triplets_pool) == 0)
        return var_dep_list


    # 这是 `resolve_dependancy` 方法的实现。该方法用于解析变量的依赖关系，将目标变量的依赖三元组和剩余的三元组分别返回。以下是代码的逐行解释：
    # 1. `dep = []` 和 `left = []`: 初始化两个列表，用于保存目标变量的依赖三元组和剩余的三元组。
    # 2. `if not triplets:`: 如果 `triplets` 为空，说明目标变量受到了过滤器的约束，这时需要检查过滤器中的约束条件。
    #    - `for tri in triplets:`: 遍历每个三元组。
    #       - `if tri[0] == target_var:`: 如果三元组的头是目标变量，将其添加到依赖三元组中。
    #          - `if tri[-1].startswith('?') and tri[-1] not in successors:`: 如果三元组的尾部是变量，并且该变量不在 `successors` 中，说明它是目标变量的后继变量。
    #             - `successor_var = tri[-1]`: 将该变量添加到 `successors` 中。
    #             - `if filter_lines:`: 如果存在过滤器，需要检查过滤器中的条件。
    #                - `new_filter_lines = []`: 初始化一个新的过滤器列表。
    #                - `for line in filter_lines:`: 遍历过滤器中的每一行。
    #                   - `if successor_var in line:`: 如果后继变量在当前行中，说明找到了对应的过滤器条件。
    #                      - `found_filter_variable = True`: 将标志位置为 `True`。
    #                      - `line = line.replace('FILTER (str(', '').replace(')', '')`: 去除过滤器条件中的无关部分。
    #                      - `tuple_list = line.split('=')`: 将过滤器条件按等号分割。
    #                      - `var = tuple_list[0].strip()`: 获取变量部分，并去除空格。
    #                      - `value = tuple_list[1].strip()`: 获取值部分，并去除空格。
    #                      - `assert successor_var == var`: 断言后继变量和过滤器中的变量相同。
    #                      - `if value.isalpha(): tri[-1] = value+'@en'`: 如果值是字母，则添加 '@en' 后缀。
    #                      - `else: tri[-1] = value`: 否则，直接将值赋给三元组的尾部。
    #                - `new_filter_lines.append(line)`: 将未处理的过滤器条件添加到新的过滤器列表中。
    #                - `if not found_filter_variable:`: 如果没有找到对应的过滤器条件，说明后继变量不受过滤器约束。
    #                   - `successors.append(successor_var)`: 将后继变量添加到 `successors` 中。
    #                - `filter_lines = new_filter_lines`: 更新过滤器列表。
    #             - `else:`: 如果没有过滤器条件，直接将后继变量添加到 `successors` 中。
    #       - `elif tri[-1] == target_var:`: 如果三元组的尾部是目标变量，将其添加到依赖三元组中。
    #          - `if tri[0].startswith('?') and tri[0] not in successors:`: 如果三元组的头是变量，并且该变量不在 `successors` 中，说明它是目标变量的后继变量。
    #             - `successors.append(tri[0])`: 将该变量添加到 `successors` 中。
    #       - `else:`: 如果既不是头部也不是尾部，将该三元组添加到剩余的三元组列表中。
    # 3. `return dep, left`: 返回目标变量的依赖三元组列表和剩余的三元组列表。
    # 该方法的主要作用是根据给定的三元组信息解析目标变量的依赖关系，同时将后继变量添加到 `successors` 列表中。如果您有进一步的问题或需要更多解释，请随时提问。
    def resolve_dependancy(self, triplets, filter_lines, target_var, successors):
        """resolve dependancy of variables
        @param triplets: all sparql triplet lines
        @param filter_lines: filter lines that start with `Filter (str(`
        @param target_var: target variable
        @param successors: successor variables of target variable

        @return: dependancy triplets of target_var, left triplets (independant of target_var)
        """
        dep = []
        left = []
        if not triplets:  # empty triplets, target_var constrained by filter

            # ns:m.0f9wd ns:influence.influence_node.influenced ?x .
            # ?x ns:government.politician.government_positions_held ?c .
            # ?c ns:government.government_position_held.from ?num .
            # ORDER BY ?num LIMIT 1
            pass
        else:
            for tri in triplets:
                if tri[0] == target_var:  # head is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # tail is variable
                    if tri[-1].startswith('?') and tri[-1] not in successors:
                        successor_var = tri[-1]
                        if filter_lines:  # check filter variable `?sk0`
                            new_filter_lines = []
                            found_filter_variable = False
                            for line in filter_lines:
                                if successor_var in line:
                                    found_filter_variable = True
                                    line = line.replace(
                                        'FILTER (str(', '').replace(')', '')
                                    tuple_list = line.split('=')
                                    var = tuple_list[0].strip()
                                    value = tuple_list[1].strip()

                                    assert successor_var == var
                                    if value.isalpha():
                                        tri[-1] = value+'@en'
                                    else:
                                        tri[-1] = value
                                    # tri[-1] = value+'@en'
                                else:
                                    new_filter_lines.append(line)

                            # remove corresponding filter_lines
                            if not found_filter_variable:  # no filter variable found
                                # add to successor variable
                                successors.append(successor_var)

                            filter_lines = new_filter_lines

                        else:
                            # add to successor variable
                            successors.append(successor_var)
                elif tri[-1] == target_var:  # tail is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # head is variable
                    if tri[0].startswith('?') and tri[0] not in successors:
                        successors.append(tri[0])  # add to successor variable
                else:
                    left.append(tri)  # left triplets
        return dep, left


# 这是一个将WebQSP解析实例转换为S表达式的函数。以下是代码的逐行解释：
# 1. `def convert_parse_instance(parse):`: 定义一个函数，接受WebQSP解析实例作为输入。
# 2. `sparql = parse['Sparql']`: 从解析实例中提取Sparql查询字符串。
# 3. `try:`: 尝试执行以下操作。
#    - `s_expr = parser.parse_query(sparql, parse['TopicEntityMid'])`:
#     调用名为 `parser` 的对象的 `parse_query` 方法，将Sparql查询字符串和解析实例中的 `TopicEntityMid` 作为参数传递。此方法用于将Sparql查询转换为S表达式。
#    - `except AssertionError:`: 如果发生 `AssertionError` 异常，表示解析失败，将 `s_expr` 设置为字符串 'null'。
# 4. `parse['SExpr'] = s_expr`: 将生成的S表达式存储在解析实例的字典中，使用键 `'SExpr'`。
# 5. `return parse, s_expr != 'null'`: 返回解析实例的字典和一个布尔值，该值指示解析是否成功。如果 `s_expr` 不等于字符串 'null'，则布尔值为 `True`，否则为 `False`。
# 该函数主要作用是调用解析器来尝试将Sparql查询转换为S表达式，并将结果存储在解析实例中。如果解析成功，返回 `True`，否则返回 `False`。
def convert_parse_instance(parse):
    """convert a webqsp parse instance to a s_expr"""
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        s_expr = parser.parse_query(sparql, parse['TopicEntityMid'])
        # print('---GOOD------')
        # print(sparql)
        # print(s_expr)
    except AssertionError:
        s_expr = 'null'
    # print(parse[''])
    parse['SExpr'] = s_expr
    return parse, s_expr != 'null'


# 这是两个函数的代码片段，涉及WebQSP S表达式的处理。以下是逐行解释：
# 1. `def webq_s_expr_to_sparql_query(s_expr):`
#    - `ast = parse_s_expr(s_expr)`: 调用 `parse_s_expr` 函数来将S表达式解析成抽象语法树（AST）。
# 这个函数的目的是将WebQSP的S表达式转换为Sparql查询的抽象语法树（AST）。
# 2. `def execute_webq_s_expr(s_expr):`
#    - `try:`: 尝试执行以下操作。
#       - `sparql_query = lisp_to_sparql(s_expr)`: 调用 `lisp_to_sparql` 函数将S表达式转换为Sparql查询字符串，并将结果存储在 `sparql_query` 中。
#       - `print(f'Transformed sparql:\n{sparql_query}')`: 打印转换后的Sparql查询。
#       - `denotation = execute_query(sparql_query)`: 调用 `execute_query` 函数执行Sparql查询，并将结果存储在 `denotation` 中。
#    - `except:`: 如果发生异常，将 `denotation` 设置为空列表 `[]`。
#    - `return denotation`: 返回Sparql查询的执行结果。
# 这个函数的目的是执行一个WebQSP的S表达式，将其转换为Sparql查询，执行查询，并返回查询结果。
# 如果您有关于这两个函数的具体问题或需要进一步的解释，请告诉我，我将乐意提供帮助。
def webq_s_expr_to_sparql_query(s_expr):
    ast = parse_s_expr(s_expr)


def execute_webq_s_expr(s_expr):
    try:
        sparql_query = lisp_to_sparql(s_expr)
        print(f'Transformed sparql:\n{sparql_query}')
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    return denotation


# 这段代码主要涉及对WebQSP数据集进行增强，通过生成S表达式，并进行Sparql查询的执行，以及对执行结果的检查。
# 以下是主要步骤的解释：
# 1. **加载数据集：**
#    从指定路径加载WebQSP数据集。
# 2. **循环处理每个数据实例：**
#    对数据集中的每个问题进行循环处理。
# 3. **生成S表达式并执行Sparql查询：**
#    对每个问题的解析结果，生成S表达式，并通过`convert_webqsp_sparql_instance`函数获取生成结果。
# 4. **检查执行结果的准确性：**
#    如果指定了`check_execute_accuracy`，则会检查Sparql查询的执行结果是否正确，并将结果存储在`SExpr_execute_right`字段中。
# 5. **输出和保存结果：**
#    打印并保存生成的S表达式结果。
# 该代码的目的是通过生成S表达式，增强原始的WebQSP数据集，并检查Sparql查询的执行结果准确性。如果有特定的问题或需要更详细的解释，请提出。
def augment_with_s_expr_webqsp(split, check_execute_accuracy=False):
    """augment original webqsp datasets with s-expression"""
    #dataset = load_json(f'data/origin/ComplexWebQuestions_{split}.json')
    dataset = load_json(f'data/WebQSP/origin/WebQSP.{split}.json')
    dataset = dataset['Questions']

    total_num = 0
    hit_num = 0
    execute_hit_num = 0
    failed_instances = []
    for i,data in tqdm(enumerate(dataset), total=len(dataset)):
        
        # sparql = data['sparql']  # sparql string
        parses = data['Parses']
        for parse in parses:
            total_num += 1
            sparql = parse['Sparql']
        
            instance, flag_success = convert_webqsp_sparql_instance(sparql, parse)
            if flag_success:
                hit_num += 1
                if check_execute_accuracy:
                    execute_right_flag = False
                    try:
                        execute_ans = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        execute_ans = [res.replace("http://rdf.freebase.com/ns/",'') for res in execute_ans]
                        if 'Answers' in parse:
                            gold_ans = [ans['AnswerArgument'] for ans in parse['Answers']]
                        else:
                            gold_ans = execute_query_with_odbc(parse['Sparql'])
                            gold_ans = [res.replace("http://rdf.freebase.com/ns/",'') for res in gold_ans]
                        # if split=='test':
                        #     gold_ans = execute_query(parse['Sparql'])
                        # else:
                        #     gold_ans = [x['answer_id'] for x in data['answers']]

                        if set(execute_ans) == set(gold_ans):
                            execute_hit_num +=1
                            execute_right_flag = True
                            # print(f'{i}: SExpr generation:{flag_success}, Execute right:{execute_right_flag}')
                        else:
                            temp = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        instance['SExpr_execute_right'] = execute_right_flag
                    except Exception:
                        temp = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        # instance['SExpr_executed_succeed']=False
                        instance['SExpr_execute_right'] = execute_right_flag
                    if not execute_right_flag:
                        pass
                        # print(f'ID:{instance["ID"]},\nExpected Ansewr:{gold_ans},\nGot Answer:{execute_ans}')
            else:
                # if check_execute_accuracy:
                #     instance['SExpr_execute_right'] = False
                failed_instances.append(instance)
    # print(hit_num, total_num, hit_num/total_num, len(dataset))
        if (i+1)%100==0:
            print(f'In the First {i+1} questions, S-Expression Gen rate [{split}]: {hit_num}, {total_num}, {hit_num/total_num}, {i+1}')
            if check_execute_accuracy:            
                    print(f'In the First {i+1} questions, Execute right rate [{split}]: {execute_hit_num}, {total_num}, {execute_hit_num/total_num}, {i+1}', )

    print(f'S-Expression Gen rate [{split}]: {hit_num}, {total_num}, {hit_num/total_num}, {len(dataset)}')
    print(f'Execute right rate [{split}]: {execute_hit_num}, {total_num}, {execute_hit_num/total_num}, {len(dataset)}', )
    

    sexpr_dir = 'data/WebQSP/sexpr'
    if not os.path.exists(sexpr_dir):
        os.makedirs(sexpr_dir)

    print(f'Writing S_Expression Results into {sexpr_dir}/WebQSP.{split}.expr.json')

    dump_json(dataset, f'{sexpr_dir}/WebQSP.{split}.expr.json', indent=4)
    # dump_json(failed_instances, f'data/WEBQSP/sexpr/WebQSP.{split}.failed.json', indent=4)


# 这个函数的目的是将WebQSP数据集中的Sparql查询转换为S表达式，并将结果存储在原始数据中。下面是该函数的主要步骤：
# 1. **获取Topic Entity 的 MID 列表：**
#    使用提供的 Topic Entity 的 MID 创建 MID 列表。
# 2. **解析 Sparql 查询为 S 表达式：**
#    调用 `parser.parse_query_webqsp` 函数将 Sparql 查询解析为 S 表达式。如果解析失败，会抛出 `AssertionError` 异常，
#     并在不包含 `#MANUAL SPARQL` 的情况下打印错误信息。如果解析失败，将 `s_expr` 设置为 'null'。
# 3. **将 S 表达式结果存储到原始数据中：**
#    将生成的 S 表达式存储在原始数据的 'SExpr' 字段中。
# 4. **返回结果：**
#    返回包含原始数据和一个指示是否成功生成 S 表达式的标志的元组。
# 这个函数是 WebQSP 数据集增强过程的一部分，将 Sparql 转换为 S 表达式以提高问题的可解释性和查询的执行性能。
def convert_webqsp_sparql_instance(sparql, origin_data):
    """convert a webqsp sparql to a s_expr"""
    # mid_list = []
    # pattern_str = r'ns:m\.0\w*'
    # pattern = re.compile(pattern_str)
    # mid_list = list(set([mid.strip()
    #                 for mid in re.findall(pattern_str, sparql)]))
    
    mid_list = [f'ns:{origin_data["TopicEntityMid"]}']
    
    # for debug
    # if origin_data['TopicEntityMid'] in ['m.05bz_j','m.0166b']:
    #     print('for debug')

    try:
        s_expr = parser.parse_query_webqsp(sparql, mid_list)
    except AssertionError:
        if '#MANUAL SPARQL' not in sparql:
            print(f'Error processing sparql: {sparql}')
        s_expr = 'null'

    origin_data['SExpr'] = s_expr
    return origin_data, s_expr != 'null'


# 这个函数似乎是用于从查询中找到时间宏模板的一部分。以下是该函数的主要步骤：
# 1. **提取查询的前缀语句：**
#    提取查询中的前缀语句，这些语句以 "PREFIX" 开头。
# 2. **验证查询的开头和结束：**
#    对查询的开头和结尾进行验证，确保满足特定的格式。
# 3. **提取查询主体部分：**
#    提取查询主体部分。
# 4. **调用 `check_time_macro_from_body_lines` 函数：**
#    调用另一个函数来检查查询主体部分是否包含时间宏模板。
# 总体而言，这个函数看起来是一个用于处理 WebQSP 数据集中查询的特定部分的辅助函数，以便找到时间宏模板。
def find_macro_template_from_query(query, topic_mid):
    # print('QUERY', query)
    lines = query.split('\n')
    lines = [x for x in lines if x]

    assert lines[0] != '#MANUAL SPARQL'

    prefix_stmts = []
    line_num = 0
    while True:
        l = lines[line_num]
        if l.startswith('PREFIX'):
            prefix_stmts.append(l)
        else:
            break
        line_num = line_num + 1

    next_line = lines[line_num]
    assert next_line.startswith('SELECT DISTINCT ?x')
    line_num = line_num + 1
    next_line = lines[line_num]
    assert next_line == 'WHERE {'
    assert lines[-1] in ['}', 'LIMIT 1']

    lines = lines[line_num:]
    assert all(['FILTER (str' not in x for x in lines])
    # normalize body lines
    # return_val = check_time_macro_from_body_lines(lines)
    # if return_val:

    # relation_prefix, suffix_pair = c
    return check_time_macro_from_body_lines(lines)


# 这个函数用于检查查询主体部分是否包含时间宏模板。具体步骤如下：
# 1. **检查主体部分的最后四行是否符合时间宏的模板：**
#    检查主体部分的倒数第四行是否以 `'FILTER(NOT EXISTS {?'` 开头。这是时间宏的一部分。
# 2. **提取时间宏的相关信息：**
#    提取时间宏的起始和结束关系的信息。
# 3. **验证起始和结束关系是否匹配：**
#    验证时间宏的起始和结束关系是否匹配，确保它们的前两个部分相同。
# 4. **提取前缀、起始后缀和结束后缀：**
#    提取时间宏的前缀、起始关系的后缀和结束关系的后缀。
# 5. **返回时间宏的相关信息：**
#    返回时间宏的前缀、起始关系的后缀和结束关系的后缀。
# 如果查询主体部分符合时间宏的模板，则函数返回包含时间宏信息的元组；否则返回 `None`。
def check_time_macro_from_body_lines(lines):
    # check if xxx
    if lines[-4].startswith('FILTER(NOT EXISTS {?'):
        # WHERE {
        # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
        # ?y ns:government.government_position_held.office_holder ?x .
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
        # EXISTS {?y ns:government.government_position_held.from ?sk1 .
        # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
        # EXISTS {?y ns:government.government_position_held.to ?sk3 .
        # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
        # }
        body_lines = lines[1:-7]
        range_lines = lines[-7:-1]
        range_prompt_start = range_lines[0]
        range_prompt_start = range_prompt_start[range_prompt_start.index(
            '{') + 1:range_prompt_start.index('}')]
        range_relation_start = range_prompt_start.split(' ')[1]

        # range_relation = '.'.join(range_relation.split('.')[:2]) + '.time_macro'
        # range_relation = range_relation[3:] # rm ns:

        range_prompt_end = range_lines[3]
        range_prompt_end = range_prompt_end[range_prompt_end.index(
            '{') + 1:range_prompt_end.index('}')]
        range_relation_end = range_prompt_end.split(' ')[1]

        assert range_relation_start.split(
            '.')[:2] == range_relation_end.split('.')[:2]
        start_suffix = range_relation_start.split('.')[-1]
        end_suffix = range_relation_end.split('.')[-1]
        prefix = '.'.join(range_relation_start.split('.')[:2])[3:]
        return prefix, start_suffix, end_suffix
    else:
        return None


# 这个函数用于从WebQSP数据集中的一个解析实例中提取宏模板信息。具体步骤如下：
# 1. **获取Sparql查询和主题实体信息：**
#    从解析实例中获取Sparql查询字符串。
# 2. **调用 `find_macro_template_from_query` 函数：**
#    调用之前定义的 `find_macro_template_from_query` 函数，以提取Sparql查询中的宏模板信息。
# 3. **异常处理：**
#    如果在提取过程中出现断言错误（AssertionError），则返回 `None`。
# 该函数的目的是在给定的解析实例中查找和提取Sparql查询中的宏模板信息。如果找到宏模板，将返回包含宏模板信息的元组；否则返回 `None`。
def extract_macro_template_from_instance(parse):
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        return find_macro_template_from_query(sparql, parse['TopicEntityMid'])
    except AssertionError:
        return None


# `parse_webqsp_sparql` 函数的目的是将 WebQSP 数据集中的 Sparql 查询解析为 s-expressions，并通过 `augment_with_s_expr_webqsp` 函数进行数据增强。
#     这个函数有一个布尔参数 `check_execute_accuracy`，用于指示是否检查执行准确性。
# 具体步骤如下：
# 1. **调用 `augment_with_s_expr_webqsp` 函数：**
#    分别对训练集和测试集调用 `augment_with_s_expr_webqsp` 函数，进行数据增强。这个函数会将原始 WebQSP 数据集中的 Sparql 查询转换为 s-expressions，并将转换后的结果写入文件。
# 2. **数据增强：**
#    在数据增强的过程中，会对每个解析实例进行处理，将 Sparql 查询转换为 s-expressions。如果 `check_execute_accuracy` 参数为 `True`，还会检查执行准确性。
# 这个函数的目的是预处理 WebQSP 数据集，以便后续的模型训练或其他任务。
def parse_webqsp_sparql(check_execute_accuracy=False):
    """Parse WebQSP sparqls into s-expressions"""
    augment_with_s_expr_webqsp('train',check_execute_accuracy)
    # augment_with_s_expr_webqsp('dev',check_execute_accuracy)
    augment_with_s_expr_webqsp('test',check_execute_accuracy)    
    


if __name__ == '__main__':
    
    parser = Parser()
    """
    Since WebQSP may provide multiple `Parses` for each question
    Execution accuracy of generated S-Expression will be verified.
    It will later be used as an filtering condition in step (5).1
    """
    parse_webqsp_sparql(check_execute_accuracy=False)
