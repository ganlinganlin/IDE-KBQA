
# 这段代码导入了一些常用的Python库和深度学习框架的相关模块。以下是对导入的模块的简要解释：
# 1. `random`: Python 内置的随机数生成模块，用于生成伪随机数。
# 2. `torch`: PyTorch 深度学习框架的主要模块，提供张量（tensor）操作、梯度计算等功能。
# 3. `numpy as np`: NumPy 是用于科学计算的 Python 库，`as np` 是为了方便使用别名 `np`。
# 4. `transformers`: 这是 Hugging Face 公司提供的一个用于自然语言处理任务的库，包括预训练的模型和相应的工具。在这里，它导入了其中的两个子模块：
#    - `AutoTokenizer`: 用于自动选择合适的分词器（tokenizer），根据预训练模型的类型。
#    - `AutoConfig`: 用于自动选择合适的配置文件，根据预训练模型的类型。
# 这些库和模块通常在深度学习和自然语言处理的任务中使用，以简化代码、提高效率，并提供许多预训练的模型和工具供使用。
import random

import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
)

# 这段代码定义了一些常量，它们可能是与实体链接（entity linking）和 Freebase 数据库相关的配置信息。以下是对这些常量的简要解释：
# 1. `ELQ_SERVICE_URL = "http://localhost:5688/entity_linking"`: 这个常量定义了一个 URL，指向一个实体链接服务。
#     实体链接是自然语言处理中的任务，旨在将文本中的实体链接到知识库中的相应实体。这个服务可能用于执行实体链接的任务。
# 2. `FREEBASE_SPARQL_WRAPPER_URL = "http://localhost:8890/sparql"`: 这个常量定义了一个 URL，指向 Freebase 数据库的 SPARQL 包装器。
#     SPARQL 是一种查询语言，用于检索 RDF 数据库中的信息。这个 URL 可能用于执行与 Freebase 数据库相关的 SPARQL 查询。
# 3. `FREEBASE_ODBC_PORT = "13001"`: 这个常量定义了 Freebase 数据库的 ODBC（Open Database Connectivity）端口号。ODBC 是一种用于访问数据库的标准接口。
#     这个端口号可能用于建立与 Freebase 数据库的连接。
# 这些常量提供了与实体链接和 Freebase 数据库交互所需的配置信息，包括服务的 URL 和端口号。请注意，这里的地址 "localhost" 表示这些服务运行在本地机器上。
ELQ_SERVICE_URL = "http://localhost:5688/entity_linking"
FREEBASE_SPARQL_WRAPPER_URL = "http://localhost:8890/sparql"
FREEBASE_ODBC_PORT = "13001"

# 这段代码包含了两个函数：
# 1. `set_seed(args)`: 该函数用于设置随机种子，以确保在进行随机操作时能够复现结果。接受一个 `args` 对象作为参数，其中可能包含一个 `seed` 属性和 `n_gpu` 属性。
#     它使用 `random.seed`、`np.random.seed` 和 `torch.manual_seed` 来设置 Python 内置的随机数生成器、NumPy 库和 PyTorch 库的随机种子。
#     如果有 GPU（`args.n_gpu > 0`），还调用了 `torch.cuda.manual_seed_all` 来设置 CUDA 库的随机种子。
# 2. `to_list(tensor)`: 该函数用于将 PyTorch 的张量（tensor）转换为 Python 列表。它通过 `detach` 方法脱离梯度，然后使用 `cpu` 方法将张量移动到 CPU 上，
#     最后使用 `tolist` 方法将其转换为 Python 列表。
# 这两个函数通常在深度学习任务中使用。`set_seed` 用于确保实验的可重复性，而 `to_list` 用于将 PyTorch 的张量转换为 Python 可处理的数据结构。
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# 这段代码定义了一个函数 `register_args(parser)`，该函数用于向一个命令行参数解析器（parser）中添加一系列参数。
#     这些参数通常用于配置模型训练和评估的各种选项。下面是对一些重要参数的简要解释：
# - **Required Parameters**:
#   - `--dataset`: 操作的数据集。
#   - `--model_type`: 模型类型。
#   - `--model_name_or_path`: 预训练模型或来自 huggingface.co/models 的模型标识符的路径。
#   - `--output_dir`: 模型检查点和预测输出的目录。
# - **Other Parameters**:
#   - `--data_dir`: 输入数据目录，应包含任务的 .json 文件。如果未指定数据目录或训练/预测文件，则将使用 tensorflow_datasets。
#   - `--train_file` 和 `--predict_file`: 输入训练文件和评估文件。
#   - `--config_name` 和 `--tokenizer_name`: 预训练配置文件和分词器的名称或路径。
#   - `--cache_dir`: 存储从 s3 下载的预训练模型的位置。
#   - `--max_seq_length`: WordPiece 分词后的最大输入序列长度。
#   - `--do_train`, `--do_eval`, `--do_predict`: 控制是否进行训练、评估、预测。
#   - `--per_gpu_train_batch_size` 和 `--per_gpu_eval_batch_size`: 训练和评估时每个 GPU/CPU 的批处理大小。
#   - `--learning_rate`, `--num_train_epochs`, `--max_steps`: 训练时的学习率、训练轮数和最大步数。
#   - `--gradient_accumulation_steps`: 累积梯度的步数。
#   - `--warmup_steps` 和 `--warmup_ratio`: 学习率线性预热的步数和比率。
#   - `--num_contrast_sample`: 一个批次中的样本数量。
#   - `--no_cuda`: 是否不使用 CUDA。
#   - `--overwrite_output_dir` 和 `--overwrite_cache`: 是否覆盖输出目录和缓存。
#   - `--seed`: 随机种子。
#   - `--threads`: 用于将示例转换为特征的线程数。
# 以及其他一些参数，用于训练课程（`training_curriculum`）、文本化选择（`linear_method`）、日志记录器（`logger`）等。
# 这些参数允许用户在命令行中配置模型训练和评估的各个方面。
def register_args(parser):
    # Required parameters
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="dataset to operate on",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",
        default=96,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup ratio.")
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm bar"
    )
    parser.add_argument("--num_contrast_sample", type=int, default=20, help="number of samples in a batch.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    # train curriculum
    parser.add_argument("--training_curriculum", default="random",type=str, choices=["random", "bootstrap", "mixbootstrap"])
    parser.add_argument("--bootstrapping_start", default=None, type=int, help="when to start bootstrapping sampling")
    parser.add_argument("--bootstrapping_ticks", default=None, type=str, help="when to update scores for bootstrapping in addition to the startpoint")

    # textualizing choices
    parser.add_argument("--linear_method", default="vanilla",type=str, choices=["vanilla", "naive_text", "reduct_text"])

    # logger
    parser.add_argument("--logger",default=None, help="logger")

# 这段代码定义了一个函数 `validate_args(args)`，该函数用于在加载数据之前验证一些参数的合法性。
#     主要的验证逻辑涉及到参数 `training_curriculum`、`bootstrapping_start` 和 `bootstrapping_ticks`。以下是对代码的简要解释：
# - 如果 `training_curriculum` 为 "random"，则 `bootstrapping_update_epochs` 参数应该为空列表。
# - 否则，`bootstrapping_start` 参数应该不为 `None` 且大于 0。
# - 如果 `bootstrapping_ticks` 为 `None`，则 `bootstrapping_update_epochs` 会被设置为只包含 `bootstrapping_start` 的列表。
# - 否则，`bootstrapping_ticks` 中的额外更新时期将被解析为整数，并与 `bootstrapping_start` 一起放入 `bootstrapping_update_epochs` 列表中。
# 这个函数的目的是确保一些特定参数在合理的范围内，以避免在加载数据之前出现错误。
def validate_args(args):
    # validate before loading data
    if args.training_curriculum == "random":
        args.bootstrapping_update_epochs = []
    else:
        assert args.bootstrapping_start is not None
        assert args.bootstrapping_start > 0

        if args.bootstrapping_ticks is None:
            bootstrapping_update_epochs = [args.bootstrapping_start]
        else:
            additional_update_epochs = [int(x) for x in args.bootstrapping_ticks.split(',')]
            bootstrapping_update_epochs = [args.bootstrapping_start] + additional_update_epochs
        args.bootstrapping_update_epochs = bootstrapping_update_epochs

# 这段代码定义了一个函数 `load_untrained_model(args)`，该函数用于加载未经训练的模型。以下是对代码的简要解释：
# 1. `args.model_type = args.model_type.lower()`: 将 `args.model_type` 转换为小写，确保一致性。
# 2. `config = AutoConfig.from_pretrained(...)`: 使用 `AutoConfig` 类从预训练模型或配置文件中加载配置信息。如果提供了 `args.config_name`，则使用它，
#     否则使用 `args.model_name_or_path`。同时，可以指定缓存目录。
# 3. `tokenizer = AutoTokenizer.from_pretrained(...)`: 使用 `AutoTokenizer` 类从预训练模型或分词器文件中加载分词器。
#     如果提供了 `args.tokenizer_name`，则使用它，否则使用 `args.model_name_or_path`。可以选择是否进行小写处理，并指定缓存目录。
# 4. `model_class = MODEL_TYPE_DICT[args.model_type]`: 从一个名为 `MODEL_TYPE_DICT` 的字典中获取模型类。这需要确保 `args.model_type` 在字典中有对应的条目。
# 5. `model = model_class.from_pretrained(...)`: 使用获取的模型类从预训练模型中加载模型。根据是否是 TensorFlow 模型进行适当的设置，并使用提供的配置信息和缓存目录。
# 最终，函数返回加载得到的配置对象 (`config`)、分词器对象 (`tokenizer`) 和模型对象 (`model`)。这样，可以使用这些对象进行后续的训练或评估。
def load_untrained_model(args):
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_class = MODEL_TYPE_DICT[args.model_type]
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    return config, tokenizer, model

# 这个函数 `get_model_class(args)` 接受一个 `args` 对象作为参数，并从一个名为 `MODEL_TYPE_DICT` 的字典中获取与 `args.model_type` 相对应的模型类。
#     这个字典似乎是在其他地方定义的，其中包含了模型类型的映射关系。函数返回对应的模型类。
# 例如，如果 `args.model_type` 是字符串 "bert"，那么这个函数可能返回与 BERT 模型相关的类。字典的定义可能是这样的：
# ```python
# MODEL_TYPE_DICT = {
#     "bert": BertModel,
#     "gpt": GPTModel,
#     # 其他模型类型的映射
# }
# ```
# 这样，通过调用 `get_model_class(args)` 函数，可以根据参数中指定的模型类型获取相应的模型类，然后用于加载模型或进行其他操作。
def get_model_class(args):
    return MODEL_TYPE_DICT[args.model_type]

