
# 这段代码导入了一些 Python 模块和库，让我们逐个进行解释：
# 1. **`from collections import defaultdict`**:
#    - 引入了 Python 内建的 `collections` 模块中的 `defaultdict` 类。`defaultdict` 是一个字典的子类，它允许在创建字典时为每个键设置一个默认值。
# 2. **`from typing import List, Tuple`**:
#    - 从 `typing` 模块中导入了 `List` 和 `Tuple` 类型。这用于提供对类型提示的支持，以增加代码的可读性和可维护性。
# 3. **`from SPARQLWrapper import SPARQLWrapper, JSON`**:
#    - 导入了 `SPARQLWrapper` 模块中的 `SPARQLWrapper` 和 `JSON`。`SPARQLWrapper` 是一个用于与SPARQL端点进行交互的库，`JSON` 用于指定在 SPARQL 查询中使用 JSON 格式。
# 4. **`import json`**:
#    - 导入了 Python 内建的 `json` 模块，用于处理 JSON 数据。
# 5. **`import urllib`**:
#    - 导入了 Python 内建的 `urllib` 模块，用于处理 URL 相关的操作。
# 6. **`from pathlib import Path`**:
#    - 导入了 Python 3.4 引入的 `pathlib` 模块，提供了一种处理文件路径的面向对象的方式。
# 7. **`from tqdm import tqdm`**:
#    - 导入了 `tqdm` 模块，用于在循环中显示进度条，提升用户体验。
# 8. **`from config import FREEBASE_SPARQL_WRAPPER_URL, FREEBASE_ODBC_PORT`**:
#    - 导入了自定义的 `config` 模块中的 `FREEBASE_SPARQL_WRAPPER_URL` 和 `FREEBASE_ODBC_PORT`。
#       这可能是用于设置 Freebase SPARQL 端点 URL 和 ODBC 端口的常量或配置信息。
# 这些导入语句表明这段代码可能与 SPARQL 查询和 Freebase 数据相关。 `SPARQLWrapper` 可能被用于与 SPARQL 端点进行通信，而其他模块则可能用于处理和操作 JSON 数据、URL 等。
from collections import defaultdict
from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm
from config import FREEBASE_SPARQL_WRAPPER_URL, FREEBASE_ODBC_PORT


# 这段代码执行以下操作：
# 1. **SPARQLWrapper 初始化:**
#    - 创建一个 `SPARQLWrapper` 对象，设置其 SPARQL 端点 URL 为 `FREEBASE_SPARQL_WRAPPER_URL`。
#     此对象用于与 SPARQL 端点进行通信。`setReturnFormat(JSON)` 指定返回结果的格式为 JSON。
# 2. **文件路径和读取:**
#    - 获取当前脚本所在目录的绝对路径，并构建一个路径字符串。
#    - 打开位于相对路径 `'../ontology/fb_roles'` 的文件，读取其中的内容。
# 3. **处理文件内容:**
#    - 对打开的文件内容进行迭代，逐行处理。
#    - 每一行使用 `split()` 方法进行拆分，生成一个字段列表。
#    - 提取字段列表中的第二个元素（`fields[1]`），并将其添加到一个 `roles` 集合中。
# 这段代码的目的是从文件中读取 Freebase 角色（roles）的信息，这些角色可能与 Freebase 的本体（ontology）相关。读取的信息可能包括角色的标识符或其他属性。
sparql = SPARQLWrapper(FREEBASE_SPARQL_WRAPPER_URL)
sparql.setReturnFormat(JSON)

path = str(Path(__file__).parent.absolute())

with open(path + '/../ontology/fb_roles', 'r') as f:
    contents = f.readlines()

roles = set()
for line in contents:
    fields = line.split()
    roles.add(fields[1])


# 这段代码涉及连接 Freebase 数据库的 ODBC（Open Database Connectivity）。让我们逐步解释：
# 1. **全局变量 `odbc_conn` 的初始化:**
#    - 定义了一个全局变量 `odbc_conn`，用于存储 ODBC 连接对象。初始值为 `None`。
# 2. **`initialize_odbc_connection` 函数:**
#    - 定义了一个函数 `initialize_odbc_connection` 用于初始化 ODBC 连接。
# 3. **连接初始化:**
#    - 在函数中，使用 `pyodbc.connect` 建立 ODBC 连接。
#    - 连接字符串由以下组成：
#      - `f'DRIVER={path}/../lib/virtodbc.so;Host=localhost:{FREEBASE_ODBC_PORT};UID=dba;PWD=dba'`
#        - `DRIVER`: 指定 ODBC 驱动程序。
#        - `Host`: 指定主机名和端口号。
#        - `UID` 和 `PWD`: 指定用户名和密码。
#    - 设置字符编码和解码方案以处理 UTF-8 编码的数据。
#    - 将连接对象赋给全局变量 `odbc_conn`。
# 4. **连接超时设置:**
#    - 设置连接的超时时间为 1 秒。
# 5. **打印连接成功信息:**
#    - 输出信息表示 Freebase Virtuoso ODBC 连接成功建立。
# 这段代码的作用是初始化并建立与 Freebase 数据库的 ODBC 连接，以便在后续的操作中能够与数据库进行交互。连接参数包括驱动程序路径、主机名和端口号、用户名和密码等。
#     连接成功后，连接对象被存储在全局变量 `odbc_conn` 中。
# connection for freebase
odbc_conn = None
def initialize_odbc_connection():
    global odbc_conn
    odbc_conn = pyodbc.connect(
        f'DRIVER={path}/../lib/virtodbc.so;Host=localhost:{FREEBASE_ODBC_PORT};UID=dba;PWD=dba'
    )
    odbc_conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    odbc_conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    odbc_conn.setencoding(encoding='utf8')
    odbc_conn.timeout = 1
    print('Freebase Virtuoso ODBC connected')

# 这段代码定义了一个名为 `execute_query` 的函数，该函数用于执行 SPARQL 查询并返回结果。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def execute_query(query: str) -> List[str]:`
#    - 函数接受一个字符串参数 `query`，表示要执行的 SPARQL 查询。
#    - 函数返回一个字符串列表 (`List[str]`)，包含查询结果。
# 2. **设置 SPARQL 查询:**
#    - `sparql.setQuery(query)`
#    - 使用先前创建的 `SPARQLWrapper` 对象 `sparql` 设置要执行的 SPARQL 查询。
# 3. **执行查询:**
#    - `results = sparql.query().convert()`
#    - 使用 `SPARQLWrapper` 对象执行查询，并将结果转换为 Python 对象。
# 4. **异常处理:**
#    - `except urllib.error.URLError:`
#    - 捕获 `urllib.error.URLError` 异常。在发生异常时，输出当前查询 (`query`) 并注释掉 `exit(0)` 行。
#     `exit(0)` 是退出程序的语句，但在这里被注释掉，可能是为了允许程序继续执行。
# 5. **处理查询结果:**
#    - 迭代查询结果的字典形式。
#    - 对于每个结果，检查字典中的变量数量是否为 1，确保只选择了一个变量。
#    - 提取结果中的变量值，将其添加到返回列表 `rtn` 中。
#    - 在添加到列表之前，对变量值进行一些处理，例如去除命名空间前缀和日期时间信息。
# 6. **返回结果列表:**
#    - `return rtn`
#    - 返回包含处理过的查询结果的字符串列表。
# 该函数的主要作用是执行 SPARQL 查询并处理返回的结果，将结果中的变量值提取并经过一些处理后返回。
#     异常处理部分表明如果发生 `URLError` 异常，将输出当前查询，并继续执行，而不是退出程序。
def execute_query(query: str) -> List[str]:
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        # exit(0)
    rtn = []
    for result in results['results']['bindings']:
        assert len(result) == 1  # only select one variable
        for var in result:
            rtn.append(result[var]['value'].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

    return rtn


# 这段代码定义了一个名为 `execute_query_with_odbc` 的函数，该函数使用 ODBC 连接执行 SPARQL 查询并返回结果。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def execute_query_with_odbc(query: str) -> List[str]:`
#    - 函数接受一个字符串参数 `query`，表示要执行的 SPARQL 查询。
#    - 函数返回一个字符串集合 (`Set[str]`)，包含查询结果。
def execute_query_with_odbc(query:str) -> List[str]:
    # 2. **全局变量检查和连接初始化:**
    #    - `global odbc_conn`
    #    - 检查全局变量 `odbc_conn` 是否为 `None`。如果是，则调用 `initialize_odbc_connection` 函数进行 ODBC 连接的初始化。
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()

    # 3. **构建查询:**
    #    - `query2 = "SPARQL " + query`
    #    - 在执行之前，将原始查询字符串添加了 `"SPARQL "` 前缀，构建新的查询字符串 `query2`。
    # print('successfully connnected to Freebase ODBC')
    result_set = set()
    query2 = "SPARQL "+query

    # 4. **查询执行:**
    #    - 使用 `odbc_conn.cursor()` 创建游标对象，执行查询，并使用 `fetchmany(10000)` 获取查询结果的前 10000 行。
    #    - 查询结果存储在 `rows` 中。
    # 5. **异常处理:**
    #    - `except Exception:`
    #    - 捕获所有异常，如果发生异常，输出错误信息并退出程序。这里注释掉了 `exit(0)`，允许程序继续执行。
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query2}")
        exit(0)

    # 6. **处理结果:**
    #    - 将查询结果中的第一列的值添加到一个集合 (`result_set`) 中。
    # 7. **返回结果集:**
    #    - `return result_set`
    #    - 返回包含查询结果的字符串集合。
    # 该函数的主要作用是使用 ODBC 连接执行 SPARQL 查询并处理结果，将结果中的第一列值添加到集合中。异常处理部分表明如果发生异常，将输出错误信息并继续执行，而不是退出程序。
    for row in rows:
        result_set.add(row[0])

    return result_set


# 这段代码定义了一个名为 `get_types_with_odbc` 的函数，该函数用于获取给定实体的类型信息。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def get_types_with_odbc(entity: str) -> List[str]:`
#    - 函数接受一个字符串参数 `entity`，表示要查询类型的实体。
#    - 函数返回一个字符串列表 (`List[str]`)，包含实体的类型信息。
# 2. **全局变量检查和连接初始化:**
#    - `global odbc_conn`
#    - 检查全局变量 `odbc_conn` 是否为 `None`。如果是，则调用 `initialize_odbc_connection` 函数进行 ODBC 连接的初始化。
# 3. **构建查询:**
#    - 构建了一个 SPARQL 查询，查询给定实体的类型信息。
#    - 查询使用了 Freebase 的 RDF 命名空间和相关前缀。
# 4. **查询执行:**
#    - 使用 `odbc_conn.cursor()` 创建游标对象，执行查询，并使用 `fetchmany(10000)` 获取查询结果的前 10000 行。
#    - 查询结果存储在 `rows` 中。
# 5. **异常处理:**
#    - `except Exception:`
#    - 捕获所有异常，如果发生异常，输出错误信息并将 `rows` 设置为空列表。这里注释掉了 `exit(0)`，允许程序继续执行。
# 6. **处理结果:**
#    - 将查询结果中的第一列的值添加到一个集合 (`types`) 中。
# 7. **返回结果集:**
#    - 如果集合中包含了类型信息，将其转换为列表并返回；否则返回空列表。
# 该函数的主要作用是使用 ODBC 连接执行 SPARQL 查询以获取给定实体的类型信息。异常处理部分表明如果发生异常，将输出错误信息并将结果列表设置为空列表，而不是退出程序。
def get_types_with_odbc(entity: str)  -> List[str]:

    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    types = set()

    query = ("""SPARQL
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query1}")
        rows=[]
        # exit(0)
    

    for row in rows:
        types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
    
    if len(types)==0:
        return []
    else:
        return list(types)


# 这段代码定义了一个名为 `get_in_relations` 的函数，该函数用于获取与给定实体有关的入边关系（in-relations）。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def get_in_relations(entity: str):`
#    - 函数接受一个字符串参数 `entity`，表示要查询入边关系的实体。
#    - 函数返回一个字符串集合 (`Set[str]`)，包含与给定实体相关的入边关系。
# 2. **构建查询:**
#    - 构建了一个 SPARQL 查询，查询与给定实体相关的入边关系。
#    - 查询使用了 Freebase 的 RDF 命名空间和相关前缀。
# 3. **查询执行:**
#    - 使用 `sparql.setQuery(query1)` 设置查询。
#    - 使用 `sparql.query().convert()` 执行查询，并将结果转换为 Python 对象。
# 4. **异常处理:**
#    - `except urllib.error.URLError:`
#    - 捕获 `urllib.error.URLError` 异常。在发生异常时，输出当前查询 (`query1`) 并退出程序。
# 5. **处理结果:**
#    - 迭代查询结果的字典形式。
#    - 对于每个结果，提取字典中 `'value'` 键对应的值，将其添加到一个集合 (`in_relations`) 中。
# 6. **返回结果集:**
#    - `return in_relations`
#    - 返回包含与给定实体相关的入边关系的字符串集合。
# 该函数的主要作用是使用 SPARQL 查询获取给定实体的入边关系，并返回结果集。异常处理部分表明如果发生 `URLError` 异常，将输出当前查询并退出程序。
def get_in_relations(entity: str):
    in_relations = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations


# 这段代码定义了一个名为 `get_out_relations` 的函数，该函数用于获取与给定实体有关的出边关系（out-relations）。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def get_out_relations(entity: str):`
#    - 函数接受一个字符串参数 `entity`，表示要查询出边关系的实体。
#    - 函数返回一个字符串集合 (`Set[str]`)，包含与给定实体相关的出边关系。
# 2. **构建查询:**
#    - 构建了一个 SPARQL 查询，查询与给定实体相关的出边关系。
#    - 查询使用了 Freebase 的 RDF 命名空间和相关前缀。
# 3. **查询执行:**
#    - 使用 `sparql.setQuery(query2)` 设置查询。
#    - 使用 `sparql.query().convert()` 执行查询，并将结果转换为 Python 对象。
# 4. **异常处理:**
#    - `except urllib.error.URLError:`
#    - 捕获 `urllib.error.URLError` 异常。在发生异常时，输出当前查询 (`query2`) 并退出程序。
# 5. **处理结果:**
#    - 迭代查询结果的字典形式。
#    - 对于每个结果，提取字典中 `'value'` 键对应的值，将其添加到一个集合 (`out_relations`) 中。
# 6. **返回结果集:**
#    - `return out_relations`
#    - 返回包含与给定实体相关的出边关系的字符串集合。
# 该函数的主要作用是使用 SPARQL 查询获取给定实体的出边关系，并返回结果集。异常处理部分表明如果发生 `URLError` 异常，将输出当前查询并退出程序。
def get_out_relations(entity: str):
    out_relations = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations

# 这段代码定义了一个名为 `query_two_hop_relations_gmt` 的函数，该函数执行带有两跳关系的 SPARQL 查询，针对给定的实体列表。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def query_two_hop_relations_gmt(entities_path, output_file):`
#    - 函数接受两个参数：
#      - `entities_path`: 包含要查询的实体列表的 JSON 文件路径。
#      - `output_file`: 存储查询结果的 JSON 文件路径。
# 2. **全局变量检查和连接初始化:**
#    - `global odbc_conn`
#    - 检查全局变量 `odbc_conn` 是否为 `None`。如果是，则调用 `initialize_odbc_connection` 函数进行 ODBC 连接的初始化。
# 3. **初始化结果字典:**
#    - `res_dict = defaultdict(list)`
#    - 创建一个默认值为列表的字典，用于存储查询结果。
# 4. **加载实体列表:**
#    - `entities = load_json(entities_path)`
#    - 从指定路径加载实体列表。
# 5. **迭代实体列表:**
#    - 使用 `tqdm` 迭代实体列表，对每个实体执行查询。
# 6. **构建查询:**
#    - 构建了一个 SPARQL 查询，该查询寻找与给定实体有两跳关系的实体。
# 7. **查询执行:**
#    - 使用 ODBC 连接执行 SPARQL 查询，并获取查询结果的前 10000 行。
# 8. **处理结果:**
#    - 将查询结果中的实体添加到一个集合 (`res`) 中。
# 9. **更新结果字典:**
#    - 将实体和对应的关系集合添加到 `res_dict` 字典中。
# 10. **异常处理:**
#    - `except Exception:`
#    - 捕获所有异常，如果发生异常，将 `rows` 设置为空列表。
# 11. **将结果保存到文件:**
#    - 使用 `dump_json(res_dict, output_file)` 将结果字典保存到指定的 JSON 文件中。
# 该函数的主要作用是执行带有两跳关系的 SPARQL 查询，对给定实体列表进行查询，并将结果保存到指定的 JSON 文件中。
#     异常处理部分表明如果发生异常，将输出错误信息并将结果列表设置为空列表。
def query_two_hop_relations_gmt(entities_path, output_file):
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    res_dict = defaultdict(list)
    entities = load_json(entities_path)
    for entity in tqdm(entities, total=len(entities)):
        query = """
        SPARQL SELECT DISTINCT ?x0 as ?r0 ?y as ?r1 where {{
            {{ ?x1 ?x0 {} . ?x2 ?y ?x1 }}
            UNION
            {{ ?x1 ?x0 {} . ?x1 ?y ?x2 }}
            UNION
            {{ {} ?x0 ?x1 . ?x2 ?y ?x1 }}
            UNION
            {{ {} ?x0 ?x1 . ?x1 ?y ?x2 }}
            FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
            FILTER (?y != rdf:type && ?y != rdfs:label)
            FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
            FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
            FILTER( !regex(?x0,"wikipedia","i"))
            FILTER( !regex(?y,"wikipedia","i"))
            FILTER( !regex(?x0,"type.object","i"))
            FILTER( !regex(?y,"type.object","i"))
            FILTER( !regex(?x0,"common.topic","i"))
            FILTER( !regex(?y,"common.topic","i"))
            FILTER( !regex(?x0,"_id","i"))
            FILTER( !regex(?y,"_id","i"))
            FILTER( !regex(?x0,"#type","i"))
            FILTER( !regex(?y,"#type","i"))
            FILTER( !regex(?x0,"#label","i"))
            FILTER( !regex(?y,"#label","i"))
            FILTER( !regex(?x0,"/ns/freebase","i"))
            FILTER( !regex(?y,"/ns/freebase","i"))
            FILTER( !regex(?x0, "ns/common."))
            FILTER( !regex(?y, "ns/common."))
            FILTER( !regex(?x0, "ns/type."))
            FILTER( !regex(?y, "ns/type."))
            FILTER( !regex(?x0, "ns/kg."))
            FILTER( !regex(?y, "ns/kg."))
            FILTER( !regex(?x0, "ns/user."))
            FILTER( !regex(?y, "ns/user."))
            FILTER( !regex(?x0, "ns/base."))
            FILTER( !regex(?y, "ns/base."))
            FILTER( !regex(?x0, "ns/dataworld."))
            FILTER( !regex(?y, "ns/dataworld."))
            FILTER regex(?x0, "http://rdf.freebase.com/ns/")
            FILTER regex(?y, "http://rdf.freebase.com/ns/")
        }} 
        
        LIMIT 300
        """.format('ns:'+entity, 'ns:'+entity, 'ns:'+entity, 'ns:'+entity)
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
            res = set()
            for row in rows:
                if row[0].startswith("http://rdf.freebase.com/ns/"):
                    res.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
                if row[1].startswith("http://rdf.freebase.com/ns/"):
                    res.add(row[1].replace('http://rdf.freebase.com/ns/', ''))
            res_dict[entity] = list(res)
            
        except Exception:
            # print(f"Query Execution Failed:{query1}")
            rows=[]
    
    # return list(res)
    dump_json(res_dict, output_file)



# 这段代码定义了一个名为 `get_2hop_relations_with_odbc` 的函数，该函数执行带有两跳关系的 SPARQL 查询，获取与给定实体相关的入边、出边和路径。以下是函数的逐步解释：
# 1. **函数签名:**
#    - `def get_2hop_relations_with_odbc(entity: str):`
#    - 函数接受一个参数：
#      - `entity`: 要查询的实体。
# 2. **初始化变量:**
#    - `in_relations = set()`
#    - `out_relations = set()`
#    - `paths = []`
#    - 初始化三个集合，用于存储入边、出边和路径。
# 3. **全局变量检查和连接初始化:**
#    - `global odbc_conn`
#    - 检查全局变量 `odbc_conn` 是否为 `None`。如果是，则调用 `initialize_odbc_connection` 函数进行 ODBC 连接的初始化。
def get_2hop_relations_with_odbc(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()

    # 4. **构建查询1:**
    #    - `query1`: 查询给定实体的入边关系。将查询结果中的实体添加到 `in_relations` 集合中。
    # 5. **构建查询2:**
    #    - `query2`: 查询给定实体的出边关系。将查询结果中的实体添加到 `out_relations` 集合中。
    # 6. **构建查询3:**
    #    - `query3`: 查询给定实体的路径，路径中的实体都是入边关系。将路径中的实体组成元组，添加到 `paths` 列表中。
    # 7. **构建查询4:**
    #    - `query4`: 查询给定实体的路径，路径中的实体都是出边关系。将路径中的实体组成元组，添加到 `paths` 列表中。
    query1 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)
    # print(query1)
    # 8. **执行查询1:**
    #    - 使用 ODBC 连接执行查询1，并获取查询结果的前 10000 行。
    # 9. **处理查询1结果:**
    #    - 将查询结果中的实体添加到 `in_relations` 集合中。
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query1)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query1}")
        rows=[]
        # exit(0)


    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1))
        

    query2 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                  """
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)
    # 10. **执行查询2:**
    #    - 使用 ODBC 连接执行查询2，并获取查询结果的前 10000 行。
    # 11. **处理查询2结果:**
    #    - 将查询结果中的实体添加到 `out_relations` 集合中。

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query2}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1 + '#R'))

    
    query3 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)
    # 12. **执行查询3:**
    #    - 使用 ODBC 连接执行查询3，并获取查询结果的前 10000 行。
    # 13. **处理查询3结果:**
    #    - 将查询结果中的路径实体组成元组，添加到 `paths` 列表中。

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query3)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query3}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1))


    query4 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                """
                FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                FILTER (?y != rdf:type && ?y != rdfs:label)
                FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                FILTER( !regex(?x0,"wikipedia","i"))
                FILTER( !regex(?y,"wikipedia","i"))
                FILTER( !regex(?x0,"type.object","i"))
                FILTER( !regex(?y,"type.object","i"))
                FILTER( !regex(?x0,"common.topic","i"))
                FILTER( !regex(?y,"common.topic","i"))
                FILTER( !regex(?x0,"_id","i"))
                FILTER( !regex(?y,"_id","i"))
                FILTER( !regex(?x0,"#type","i"))
                FILTER( !regex(?y,"#type","i"))
                FILTER( !regex(?x0,"#label","i"))
                FILTER( !regex(?y,"#label","i"))
                FILTER( !regex(?x0,"/ns/freebase","i"))
                FILTER( !regex(?y,"/ns/freebase","i"))
                FILTER( !regex(?x0, "ns/common."))
                FILTER( !regex(?y, "ns/common."))
                FILTER( !regex(?x0, "ns/type."))
                FILTER( !regex(?y, "ns/type."))
                FILTER( !regex(?x0, "ns/kg."))
                FILTER( !regex(?y, "ns/kg."))
                FILTER( !regex(?x0, "ns/user."))
                FILTER( !regex(?y, "ns/user."))
                FILTER( !regex(?x0, "ns/dataworld."))
                FILTER( !regex(?y, "ns/dataworld."))
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                FILTER regex(?y, "http://rdf.freebase.com/ns/")
                }
                LIMIT 1000
                """)
    # 14. **执行查询4:**
    #    - 使用 ODBC 连接执行查询4，并获取查询结果的前 10000 行。
    # 15. **处理查询4结果:**
    #    - 将查询结果中的路径实体组成元组，添加到 `paths` 列表中。
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query4)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query4}")
        rows = []
        # exit(0)

    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1 + '#R'))
    # 16. **返回结果:**
    #    - 返回 `in_relations`、`out_relations` 和 `paths`。
    # 该函数的主要作用是执行带有两跳关系的 SPARQL 查询，获取与给定实体相关的入边、出边和路径。最终，返回这些关系的集合和路径的列表。
    return in_relations, out_relations, paths



# 这是一个用于从Freebase知识图谱中获取与指定实体相关的二跳关系的Python代码。我将逐步解释代码的主要部分。
# 1. **函数签名**:
#     ```python
#     def get_2hop_relations_with_odbc_wo_filter(entity: str):
#     ```
#     - 函数名: `get_2hop_relations_with_odbc_wo_filter`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
# 2. **初始化和连接**:
#     ```python
#     global odbc_conn
#     if odbc_conn == None:
#         initialize_odbc_connection()
#     ```
#     - 这里使用了一个全局变量 `odbc_conn`，并检查是否已经初始化。如果尚未初始化，调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询1**:
#     ```python
#     query1 = """SPARQL PREFIX rdf: ... LIMIT 1000 """
#     ```
#     - 查询与给定实体 `entity` 相关的二跳关系，包括入边和出边。
# 4. **执行查询1**:
#     ```python
#     with odbc_conn.cursor() as cursor:
#         cursor.execute(query1)
#         rows = cursor.fetchmany(10000)
#     ```
#     - 使用ODBC连接执行SPARQL查询1，并获取查询结果的前10000行。
# 5. **处理查询1的结果**:
#     ```python
#     for row in rows:
#         # 处理每一行的结果，提取关系并添加到相应的集合中
#     ```
#     - 提取每行结果中的关系，并将它们添加到 `in_relations` 和 `out_relations` 集合中。
# 6. **SPARQL查询2、3、4**:
#     - 与查询1类似，分别处理不同的情况：查询与入边、出边相关的关系，以及它们的组合。
# 7. **执行查询2、3、4**:
#     - 与查询1相似，分别执行三个查询。
# 8. **返回结果**:
#     ```python
#     return in_relations, out_relations, paths
#     ```
#     - 函数返回三个值：
#         - `in_relations`：包含所有入边关系的集合。
#         - `out_relations`：包含所有出边关系的集合。
#         - `paths`：包含所有满足条件的路径的列表，每个路径表示为一个包含两个关系的元组。
# 请注意，此代码使用SPARQL语言来查询Freebase知识图谱，并且具有一些过滤条件以排除不必要的关系。此外，由于知识图谱可能很庞大，代码使用了限制条件 (`LIMIT 1000`) 以确保查询的效率。
def get_2hop_relations_with_odbc_wo_filter(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()


    query1 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)
    # print(query1)
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query1)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query1}")
        rows=[]
        # exit(0)


    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1))
        

    query2 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                  """
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query2}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1 + '#R'))

    
    query3 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query3)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query3}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1))


    query4 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                """
                FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                FILTER (?y != rdf:type && ?y != rdfs:label)
                FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                FILTER( !regex(?x0,"wikipedia","i"))
                FILTER( !regex(?y,"wikipedia","i"))
                FILTER( !regex(?x0,"_id","i"))
                FILTER( !regex(?y,"_id","i"))
                FILTER( !regex(?x0,"#type","i"))
                FILTER( !regex(?y,"#type","i"))
                FILTER( !regex(?x0,"#label","i"))
                FILTER( !regex(?y,"#label","i"))
                FILTER( !regex(?x0,"/ns/freebase","i"))
                FILTER( !regex(?y,"/ns/freebase","i"))
                FILTER( !regex(?x0, "ns/kg."))
                FILTER( !regex(?y, "ns/kg."))
                FILTER( !regex(?x0, "ns/dataworld."))
                FILTER( !regex(?y, "ns/dataworld."))
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                FILTER regex(?y, "http://rdf.freebase.com/ns/")
                }
                LIMIT 1000
                """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query4)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query4}")
        rows = []
        # exit(0)

    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1 + '#R'))

    return in_relations, out_relations, paths



# 这段代码是一个用于从Freebase知识图谱中获取实体标签的函数。我将逐步解释代码的主要部分。
# 1. **函数签名**:
#     - 函数名: `get_label`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
#     - 返回类型: `str`，表示实体的标签。
# 2. **SPARQL查询**:
#     - 查询实体的标签，使用了两层SELECT嵌套。
#     - 在内层SELECT中，通过 `rdfs:label` 关系获取实体的标签。
#     - 使用 `FILTER` 语句过滤只选择英语标签。
#     - 外层SELECT将结果别名为 `label`。
# 3. **执行查询**:
#     - 使用SPARQL查询语句设置查询。
#     - 使用 `sparql.query().convert()` 执行查询并获取结果。
# 4. **处理查询结果**:
#     - 遍历查询结果中的每个标签，将其添加到 `rtn` 列表中。
# 5. **返回结果**:
#     - 如果 `rtn` 列表不为空，返回第一个标签作为结果；否则，返回 `None`。
# 请注意，这里的代码使用了一个外部的 `sparql` 对象，但在提供的代码中没有看到它的初始化或导入语句。在使用这个函数之前，你需要确保 `sparql` 对象已经被正确初始化和配置。
def get_label(entity: str) -> str:
    """Get the label of an entity in Freebase"""
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             ':' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        label = result['label']['value']
        rtn.append(label)
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None


# 这段代码是一个简单的 Python 函数，用于测试 `pyodbc` 库与 Virtuoso RDF 数据库的连接。以下是代码的解释：
# 1. **导入库**:
#     - 导入 `pyodbc` 库，这是一个用于连接数据库的库。
# 2. **函数定义**:
#     - 定义了一个名为 `pyodbc_test` 的函数。
# 3. **数据库连接**:
#     - 使用 `pyodbc.connect()` 方法创建与 Virtuoso RDF 数据库的连接。
#     - 连接字符串包括驱动程序（`DRIVER`）、主机地址（`Host`）、端口号（`FREEBASE_ODBC_PORT`）、用户名（`UID`）、密码（`PWD`）等信息。
# 4. **设置编码**:
#     - 使用 `setdecoding` 和 `setencoding` 方法设置字符编码，确保正确处理 UTF-8 编码的数据。
# 5. **执行查询**:
#     - 使用 `cursor` 对象执行 SPARQL 查询。
#     - 查询中使用了类似 SQL 的语法，从 Virtuoso 数据库中选择子类关系。
# 6. **处理查询结果**:
#     - 遍历查询结果的每一行，将其转换为字符串并打印出来。
# 7. **提交事务和关闭连接**:
#     - 提交事务并关闭数据库连接。
# 请确保在使用此代码之前，已经定义了 `path`、`FREEBASE_ODBC_PORT` 等变量，并且 Virtuoso 数据库已经正确配置和运行。
import pyodbc
def pyodbc_test():
    conn = pyodbc.connect(f'DRIVER={path}/../lib/virtodbc.so;Host=localhost:{FREEBASE_ODBC_PORT};UID=dba;PWD=dba')
    print(conn)
    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    conn.setencoding(encoding='utf8')
    
    with conn.cursor() as cursor:
        cursor.execute("SPARQL SELECT ?subject ?object WHERE { ?subject rdfs:subClassOf ?object }")
        # rows = cursor.fetchall()
        rows = cursor.fetchmany(10000)
    
    for row in rows:
        row = str(row)
        print(row)
    conn.commit()
    conn.close()


# 这段代码定义了一个用于从 Freebase 知识图谱中获取实体标签的函数 `get_label_with_odbc`。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_label_with_odbc`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
#     - 返回类型: `str`，表示实体的标签。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询**:
#     - 查询实体的标签，使用了两层 SELECT 嵌套。
#     - 在内层 SELECT 中，通过 `rdfs:label` 关系获取实体的标签。
#     - 使用 `FILTER` 语句过滤只选择英语标签。
# 4. **执行查询**:
#     - 使用 `odbc_conn` 连接执行 SPARQL 查询，并获取查询结果的前10000行。
# 5. **处理查询结果**:
#     - 遍历查询结果中的每个标签，将其添加到 `rtn` 列表中。
# 6. **返回结果**:
#     - 如果 `rtn` 列表不为空，返回第一个标签作为结果；否则，返回 `None`。
# 请确保在使用此代码之前，`odbc_conn` 已经被正确初始化，且 `initialize_odbc_connection` 函数已经定义。
def get_label_with_odbc(entity: str) -> str:
    """Get the label of an entity in Freebase"""

    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
        
    query = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns: <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             'ns:' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)

    # query = query.replace("\n"," ")
    # print(query)
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query}")
        exit(0)
    
    
    rtn = []
    for row in rows:
        # print(type(row))
        rtn.append(row[0])
    
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None


# 这段代码定义了一个函数 `get_in_relations_with_odbc`，该函数从 Freebase 知识图谱中获取与指定实体相关的入边关系。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_in_relations_with_odbc`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
#     - 返回类型: `set`，包含实体的入边关系。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询**:
#     - 查询与指定实体 `entity` 相关的入边关系，使用了两层 SELECT 嵌套。
#     - 在内层 SELECT 中，通过 `:` 关系获取入边关系。
#     - 使用 `FILTER` 语句过滤只选择包含指定命名空间的关系。
# 4. **执行查询**:
#     - 使用 `odbc_conn` 连接执行 SPARQL 查询，并获取查询结果的前10000行。
#
# 5. **处理查询结果**:
#     - 遍历查询结果中的每个入边关系，将其添加到 `in_relations` 集合中。
# 6. **返回结果**:
#     - 返回包含实体入边关系的集合。
# 请确保在使用此代码之前，`odbc_conn` 已经被正确初始化，且 `initialize_odbc_connection` 函数已经定义。
def get_in_relations_with_odbc(entity: str) -> str:
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    in_relations = set()

    query1 = ("""SPARQL
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)


    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query1)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query1}")
        exit(0)
    

    for row in rows:
        in_relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations



# 这段代码定义了一个函数 `get_out_relations_with_odbc`，该函数从 Freebase 知识图谱中获取与指定实体相关的出边关系。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_out_relations_with_odbc`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
#     - 返回类型: `set`，包含实体的出边关系。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询**:
#     - 查询与指定实体 `entity` 相关的出边关系，使用了两层 SELECT 嵌套。
#     - 在内层 SELECT 中，通过 `:` 关系获取出边关系。
#     - 使用 `FILTER` 语句过滤只选择包含指定命名空间的关系。
# 4. **执行查询**:
#     - 使用 `odbc_conn` 连接执行 SPARQL 查询，并获取查询结果的前10000行。
# 5. **处理查询结果**:
#     - 遍历查询结果中的每个出边关系，将其添加到 `out_relations` 集合中。
# 6. **返回结果**:
#     - 返回包含实体出边关系的集合。
# 请确保在使用此代码之前，`odbc_conn` 已经被正确初始化，且 `initialize_odbc_connection` 函数已经定义。
def get_out_relations_with_odbc(entity: str) -> str:
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    out_relations = set()

    query2 = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)
    

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query2}")
        exit(0)
    

    for row in rows:
        out_relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations



# 这段代码定义了一个函数 `get_1hop_relations_with_odbc`，该函数从 Freebase 知识图谱中获取与指定实体直接相连的一跳关系。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_1hop_relations_with_odbc`
#     - 参数: `entity`，表示要查询的实体的名称，是一个字符串。
#     - 返回类型: `set`，包含实体的一跳关系。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询**:
#     - 查询与指定实体 `entity` 直接相连的一跳关系，使用了两层 SELECT 嵌套。
#     - 在内层 SELECT 中，通过 `:` 关系获取一跳关系。
#     - 使用 `UNION` 运算符将正向和反向的关系合并。
#     - 使用 `FILTER` 语句过滤只选择包含指定命名空间的关系。
# 4. **执行查询**:
#     - 使用 `odbc_conn` 连接执行 SPARQL 查询，并获取查询结果的前10000行。
# 5. **处理查询结果**:
#     - 遍历查询结果中的每个一跳关系，将其添加到 `relations` 集合中。
# 6. **返回结果**:
#     - 返回包含实体一跳关系的集合。
# 请确保在使用此代码之前，`odbc_conn` 已经被正确初始化，且 `initialize_odbc_connection` 函数已经定义。
def get_1hop_relations_with_odbc(entity):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    relations = set()

    query = ("""SPARQL
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '{ ?x1 ?x0 ' + ':' + entity + ' }'
              + ' UNION '
              + '{ :' + entity + ' ?x0 ?x1 ' + '}'
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)


    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query}")
        exit(0)
    

    for row in rows:
        relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return relations


# 这段代码定义了一个函数 `get_freebase_mid_from_wikiID`，该函数从 Freebase 知识图谱中获取与给定维基百科ID（`wikiID`）相关的 Freebase M_ID。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_freebase_mid_from_wikiID`
#     - 参数: `wikiID`，表示要查询的维基百科ID，是一个整数。
#     - 返回类型: `str`，包含与给定维基百科ID相关的 Freebase M_ID。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **SPARQL查询**:
#     - 查询与给定维基百科ID `wikiID` 相关的 Freebase M_ID，使用了两层 SELECT 嵌套。
#     - 在内层 SELECT 中，通过 `<http://rdf.freebase.com/key/wikipedia.en_id>` 关系获取 Freebase M_ID。
#     - 使用 `FILTER` 语句过滤只选择包含指定命名空间的 M_ID。
# 4. **执行查询**:
#     - 使用 `odbc_conn` 连接执行 SPARQL 查询，并获取查询结果的前10000行。
# 5. **处理查询结果**:
#     - 遍历查询结果中的每个 Freebase M_ID，将其添加到 `mid` 集合中。
# 6. **返回结果**:
#     - 如果 `mid` 集合不为空，返回集合中的第一个 Freebase M_ID；否则，返回空字符串。
# 请确保在使用此代码之前，`odbc_conn` 已经被正确初始化，且 `initialize_odbc_connection` 函数已经定义。
def get_freebase_mid_from_wikiID(wikiID: int):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    mid = set()

    query2 = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              '?x0 <http://rdf.freebase.com/key/wikipedia.en_id> ' + f'"{wikiID}"'
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)
    

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query2}")
        exit(0)
    

    for row in rows:
        mid.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
    
    if len(mid)==0:
        return ''
    else:
        return list(mid)[0]


# 这里定义了两个用于读取和写入JSON文件的辅助函数。
# 1. **`load_json` 函数**:
#     - 该函数用于从JSON文件中加载数据。
#     - 参数 `fname` 是文件名，`mode` 是打开文件的模式，默认为只读模式 (`"r"`)。
#     - 如果模式中包含 "b"（二进制模式），则将编码设置为 `None`。
#     - 使用 `open` 函数打开文件，并使用 `json.load` 从文件中加载JSON数据。
# 2. **`dump_json` 函数**:
#     - 该函数用于将数据写入JSON文件。
#     - 参数 `obj` 是要写入文件的对象，`fname` 是文件名，`indent` 是缩进量，默认为4。
#     - `mode` 是打开文件的模式，默认为写入模式 (`"w"`)。
#     - 如果模式中包含 "b"（二进制模式），则将编码设置为 `None`。
#     - 使用 `open` 函数打开文件，并使用 `json.dump` 将对象写入文件，可以设置缩进和是否使用ASCII编码。
# 这两个函数可以方便地读取和写入JSON格式的数据。确保在使用它们之前，导入了 `json` 模块。例如，可以使用以下导入语句：
def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


# 这段代码定义了一个函数 `get_entity_labels`，其目的是从 Freebase 知识图谱中获取一组实体的标签（labels）并将结果保存到一个JSON文件中。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_entity_labels`
#     - 参数:
#         - `src_path`: 包含实体列表的JSON文件的路径。
#         - `tgt_path`: 保存实体标签的JSON文件的路径。
# 2. **加载实体列表**:
#     - 使用先前定义的 `load_json` 函数从 `src_path` 文件中加载实体列表。
# 3. **查询实体标签**:
#     - 使用 `tqdm` 模块显示查询进度。
#     - 对每个实体调用 `get_label_with_odbc` 函数获取其标签，将结果存储在字典 `res` 中。
# 4. **保存结果**:
#     - 使用先前定义的 `dump_json` 函数将实体标签的结果写入 `tgt_path` 文件。
# 这个函数的作用是批量查询一组实体的标签并将结果保存到文件中。确保在使用此代码之前，`get_label_with_odbc` 函数已经定义，并且输入的实体列表和文件路径是正确的。
def get_entity_labels(src_path, tgt_path):
    entities_list = load_json(src_path)
    res = dict()
    # for entity in entities_list:
    for entity in tqdm(entities_list, total=len(entities_list),desc=f'Querying entity labels'):
        label = get_label_with_odbc(entity)
        res[entity] = label
    dump_json(res, tgt_path)


# 这段代码定义了一个函数 `query_relation_domain_range_label_odbc`，其目的是通过 Freebase 知识图谱查询一组关系（relations）的域、值域和标签，并将结果保存到一个JSON文件中。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `query_relation_domain_range_label_odbc`
#     - 参数:
#         - `input_path`: 包含关系列表的JSON文件的路径。
#         - `output_path`: 保存关系的域、值域和标签的JSON文件的路径。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **加载关系列表**:
#     - 使用先前定义的 `load_json` 函数从 `input_path` 文件中加载关系列表。
# 4. **查询关系的域、值域和标签**:
#     - 对于每个关系，构建一个 SPARQL 描述查询，查询关系的详细信息。
#     - 使用 `tqdm` 模块显示查询进度。
#     - 遍历查询结果，提取关系的域、值域和标签信息，并将结果存储在 `res_dict` 字典中。
# 5. **保存结果**:
#     - 使用先前定义的 `dump_json` 函数将关系的域、值域和标签信息写入 `output_path` 文件。
# 这个函数的作用是批量查询一组关系的域、值域和标签信息，并将结果保存到文件中。
#     确保在使用此代码之前，`initialize_odbc_connection` 函数已经定义，并且输入的关系列表和文件路径是正确的。
def query_relation_domain_range_label_odbc(input_path, output_path):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    relations = load_json(input_path)
    
    res_dict = dict()
    for relation in tqdm(relations):
        query = """
        SPARQL DESCRIBE {}
        """.format('ns:' + relation)
        
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
        except Exception:
            # print(f"Query Execution Failed:{query}")
            exit(0)
        
        res_dict[relation] = dict()
        for row in rows:
            if '#domain' in row[1]:
                res_dict[relation]["domain"] = row[2].replace('http://rdf.freebase.com/ns/', '')
            elif '#range' in row[1]:
                res_dict[relation]["range"] = row[2].replace('http://rdf.freebase.com/ns/', '')
            elif '#label' in row[1]:
                res_dict[relation]["label"] = row[2].replace('http://rdf.freebase.com/ns/', '')
    
    dump_json(res_dict, output_path)


# 这段代码定义了一个函数 `freebase_query_entity_type_with_odbc`，其目的是通过 Freebase 知识图谱查询一组实体的类型（type），并将结果保存到一个JSON文件中。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `freebase_query_entity_type_with_odbc`
#     - 参数:
#         - `entities_path`: 包含实体列表的JSON文件的路径。
#         - `output_path`: 保存实体类型的JSON文件的路径。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **初始化结果字典**:
#     - 使用 `defaultdict` 创建一个字典，用于存储每个实体对应的类型列表。
# 4. **加载实体列表并查询类型**:
#     - 对于每个实体，构建一个 SPARQL 描述查询，查询实体的详细信息。
#     - 使用 `tqdm` 模块显示查询进度。
#     - 遍历查询结果，提取实体的类型信息，并将结果存储在 `res_dict` 字典中。
# 5. **保存结果**:
#     - 使用先前定义的 `dump_json` 函数将实体类型的结果写入 `output_path` 文件。
# 这个函数的作用是批量查询一组实体的类型信息，并将结果保存到文件中。确保在使用此代码之前，`initialize_odbc_connection` 函数已经定义，并且输入的实体列表和文件路径是正确的。
def freebase_query_entity_type_with_odbc(entities_path, output_path):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    res_dict = defaultdict(list)
    entities = load_json(entities_path)
    count = 0
    for entity in entities:
        query = """
        SPARQL DESCRIBE {}
        """.format('ns:' + entity)
        print('count: {}'.format(count))
        count += 1
        
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
            for row in rows:
                if row[1] == 'http://rdf.freebase.com/ns/kg.object_profile.prominent_type':
                    if row[2].startswith('http://rdf.freebase.com/ns/'):
                        # res_dict[entity].append(row[2])
                        res_dict[entity].append(row[2].replace('http://rdf.freebase.com/ns/', ''))
        except Exception:
            # print(f"Query Execution Failed:{query1}")
            rows=[]
            # exit(0)
    
    dump_json(output_path, res_dict)


# 这段代码定义了一个函数 `get_freebase_relations_with_odbc`，用于获取 Freebase 知识图谱中的所有关系，并可选地限制结果数量。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `get_freebase_relations_with_odbc`
#     - 参数:
#         - `data_path`: 保存结果的JSON文件的路径。
#         - `limit`: 可选参数，限制结果的数量。默认为100，如果设置为负值或0，则表示不限制数量。
# 2. **构建连接**:
#     - 检查全局变量 `odbc_conn` 是否已经初始化，如果没有则调用 `initialize_odbc_connection()` 函数进行初始化。
# 3. **构建查询语句**:
#     - 根据传入的 `limit` 参数构建 SPARQL 查询语句，查询所有不同的关系及其出现的频次。
# 4. **执行查询**:
#     - 使用 ODBC 连接执行构建好的 SPARQL 查询语句。
# 5. **处理查询结果**:
#     - 遍历查询结果，将关系和频次的组合以列表形式存储在 `rtn` 列表中。
# 6. **保存结果**:
#     - 使用先前定义的 `dump_json` 函数将结果写入 `data_path` 文件。
# 这个函数的作用是获取 Freebase 知识图谱中的所有关系及其出现的频次，并将结果保存到文件中。确保在使用此代码之前，`initialize_odbc_connection` 函数已经定义，并且输入的文件路径是正确的。
"""
copied from `relation_retrieval/sparql_executor.py`
"""
def get_freebase_relations_with_odbc(data_path, limit=100):
    """Get all relations of Freebase"""
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    # {{ }}: to escape
    if limit > 0:
        query = """
        SPARQL SELECT DISTINCT ?p (COUNT(?p) as ?freq) WHERE {{
            ?subject ?p ?object
        }}
        LIMIT {}
        """.format(limit)
    else:
        query = """
        SPARQL SELECT DISTINCT ?p (COUNT(?p) as ?freq) WHERE {{
            ?subject ?p ?object
        }}
        """
    print('query: {}'.format(query))
    
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            # rows = cursor.fetchall()
            rows = cursor.fetchmany(10000)
    except Exception:
        # print(f"Query Execution Failed:{query1}")
        rows=[]
        exit(0)
    
    rtn = []
    for row in rows:
        rtn.append([row[0], int(row[1])])
    
    if len(rtn) != 0:
        dump_json(rtn, data_path)


# 这段代码定义了一个名为 `freebase_relations_post_process` 的函数，用于对 Freebase 知识图谱中的关系进行后处理。以下是代码的解释：
# 1. **函数签名**:
#     - 函数名: `freebase_relations_post_process`
#     - 参数:
#         - `input_path`: 包含关系数据的JSON文件的路径。
#         - `output_path`: 保存处理后关系的JSON文件的路径。
# 2. **加载输入数据**:
#     - 使用 `load_json` 函数加载输入文件中的关系数据。
# 3. **数据后处理**:
#     - 打印输入数据的长度。
#     - 从输入数据中提取关系，确保关系以 "http://rdf.freebase.com/ns/" 开头。
#     - 移除关系中的前缀 "http://rdf.freebase.com/ns/"。
#     - 将关系列表转换为集合，以去除重复项。
#     - 打印处理后的关系数据的长度。
# 4. **保存处理后的结果**:
#     - 使用先前定义的 `dump_json` 函数将处理后的关系数据写入输出文件。
# 这个函数的作用是对 Freebase 知识图谱中的关系数据进行简单的后处理，包括提取正确的前缀和去重。确保在使用此代码之前，输入的文件路径是正确的。
def freebase_relations_post_process(input_path, output_path):
    input_data = load_json(input_path)
    print(f'input length: {len(input_data)}')
    output_data = [item[0] for item in input_data]
    output_data = [item for item in output_data if item.startswith("http://rdf.freebase.com/ns/")]
    output_data = [item.replace('http://rdf.freebase.com/ns/', '') for item in output_data]
    output_data = list(set(output_data))
    print(f'output length: {len(output_data)}')
    dump_json(output_data, output_path)


if __name__=='__main__':
    
    # pyodbc_test()
    
    # print(get_label('m.04tfqf'))
    # print(get_label_with_odbc('m.0rczx'))
    # print(get_in_relations_with_odbc('m.04tfqf'))
    # print(get_out_relations_with_odbc('m.04tfqf'))

    # print(get_label('m.0fjp3'))
    # print(get_label_with_odbc('m.0fjp3'))
    # print(get_label('m.0z33s'))
    # print(get_2hop_relations_with_odbc('m.0rv97'))
    # print(get_1hop_relations_with_odbc('m.09fcm'))

    # print(get_wikipage_id_from_dbpedia_uri("http://dbpedia.org/resource/China"))

    
    # sparql = """
    # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    # PREFIX ns: <http://rdf.freebase.com/ns/> 

    # SELECT count(DISTINCT ?p)  WHERE {
    # ?s ?p ?o .
    # }
    # """
    # print(execute_query_with_odbc(sparql))
    
    # print(get_label_with_odbc('m.0y80cnb'))

    # print(get_freebase_mid_from_wikiID(39027))


    # in_rel = get_in_relations('m.04tfqf')
    # print(type(in_rel))
    # print(in_rel)
    # for split in ['train', 'dev', 'test']:
    #     execuate_reduced_sparql(split)

    # get_entity_labels()
    # print(get_2hop_relations('m.01n4w'))
    # in_relations, out_relations, paths = get_2hop_relations('m.03krjv')
    # print(len(in_relations))
    # print(len(out_relations))
    # in_relations, out_relations, paths = get_2hop_relations_with_odbc_wo_filter('m.04904')
    # print(in_relations, out_relations)

    # query_two_hop_relations_gmt(
    #     'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/unique_entity_ids.json',
    #     'data/WebQSP/relation_retrieval/cross-encoder/rng_kbqa_linking_results/WebQSP.2hopRelations.rng.elq.candEntities.json'
    # )

    # query_two_hop_relations_gmt(
    #     'data/CWQ/entity_retrieval/disamb_entities/unique_entities.json',
    #     'data/CWQ/relation_retrieval/bi-encoder/CWQ.2hopRelations.candEntities.json'
    # )

    """common_data related"""
    
    # get_freebase_relations_with_odbc('../data/common_data_0822/freebase_relations.json', limit=0)
    # freebase_relations_post_process(
    #     '../data/common_data_0822/freebase_relations.json',
    #     '../data/common_data_0822/freebase_relations_filtered.json'
    # )
    # query_relation_domain_range_label_odbc(
    #     '../data/common_data_0822/freebase_relations_filtered.json',
    #     '../data/common_data_0822/fb_relations_domain_range_label.json'
    # )