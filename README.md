

#  IDE-KBQA: Intent Detection Enhanced Model for Knowledge Base Question Answering with Large Language Models


##  1. Overview 

[//]: # (![]&#40;./figs/F1.drawio.png&#41;)
[//]: # (![]&#40;./figs/F2.drawio.png&#41;)


##  2. General Setup 

[//]: # (<h2>General Setup</h2>)
###  2.1 Environment Setup

- Create a environment
  ```
  conda create -n idekbqa python=3.8.8
<<<<<<< HEAD
  conda activate idekbqa
=======
<<<<<<< HEAD
  conda activate idekbqa
=======
  conda activate idekbqa 
  sudo apt-get update
  sudo apt-get install -y unixodbc-dev # 安装 Microsoft ODBC Driver for SQL Server (Linux)
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install -r requirement.txt
  ```
- Download the flash-attention from [flash_attn-2.3.5+cu117torch1.13cxx11abiTRUE-cp38-cp38-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases?page=2) to `princeton-nlp/`
  ```
  cd princeton-nlp
  pip install flash_attn-2.3.5+cu117torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  mv nltk_data ../../
  ```

###  2.2 Freebase KG Setup ( Below steps are according to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup) )

- Download the Freebase Virtuoso DB file from [Dropbox](https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip) or [Baidu Netdisk](https://pan.baidu.com/s/19BWHgcP534mhmBn5cA6LZg?pwd=29ic) to `Freebase-Setup/` (WARNING: 53G+ disk space is needed)
- Download the OpenLink Virtuoso 7.2.5 from [OpenLink Virtuoso 7.2.5](https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz/download) to `Freebase-Setup/`
  
  ```
  cd Freebase-Setup
  tar -zxvf virtuoso_db.zip
  tar -zxvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
  ```

- Managing the Virtuoso service ( A server with at least 100 GB RAM is recommended )

  To start the Virtuoso service:
    ```
    python3 virtuoso.py start 3001 -d virtuoso_db
    ```
  and to stop a currently running service at the same port:
    ```
    python3 virtuoso.py stop 3001
    ```

- Download FACC1 mentions for Entity Retrieval

  Download the mention information (including processed FACC1 mentions and all entity alias in Freebase) from [FACC1](https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye) to `data/common_data/facc1/`
  ```
  IDE-KBQA/
  └── data/
      ├── common_data/                  
          ├── facc1/   
              ├── entity_list_file_freebase_complete_all_mention
              └── surface_map_file_freebase_complete_all_mention                                           
  ```

### 2.3 KBQA Datasets (WebQSP, CWQ)

- [WebQSP](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) dataset has been downloaded under `data/WebQSP/origin`

  ```
  IDE-KBQA/
  └── data/
      ├── WebQSP                  
          ├── origin                    
              ├── WebQSP.train.json                    
              └── WebQSP.test.json                                       
  ```

- [CWQ](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) dataset has been downloaded under `data/CWQ/origin`

  ```
  IDE-KBQA/
  └── data/
      ├── CWQ                 
          ├── origin                    
              ├── ComplexWebQuestions_train.json                   
              ├── ComplexWebQuestions_dev.json      
              └── ComplexWebQuestions_test.json                              
  ```

### 2.4 Large Language Models (LLM)

- Download the intent ranking model from [text2vec-base-multilingual](https://huggingface.co/shibing624/text2vec-base-multilingual) to `IE/reward-model-datasets/shibing624/text2vec-base-multilingual/`
- Download the intent ranking model from [ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh) to `IE/nghuyong/ernie-3.0-base-zh/`
- Download the LLM from [LLaMa2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf) to `meta-llama/Llama-2-13b-hf/`


## 3. Data Processing

### 3.1 **Parse SPARQL queries to S-expressions**

- WebQSP
  ```
  python DP01_parse_sparql_webqsp.py
  ``` 
  Run python DP_parse_sparql_webqsp.py and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train].json`. 

- CWQ
  ```
  python DP01_parse_sparql_cwq.py
  ```
  Run DP_parse_sparql_cwq.py and the augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train].json`.

### 3.2 **Prepare data for training and evaluation**

- Start the Virtuoso service
- WebQSP
  ```
  python DP02_data_process.py --action merge_all --dataset WebQSP --split test
  ```
  ```
  python DP02_data_process.py --action merge_all --dataset WebQSP --split train
  ``` 
  The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.
  ```
  python DP02_data_process.py --action get_type_label_map --dataset WebQSP --split train
  ```
  The label map file will be saved as `data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json`.

- CWQ
  ```
  python DP02_data_process.py --action merge_all --dataset CWQ --split test 
  ```
  ```
  python DP02_data_process.py --action merge_all --dataset CWQ --split train
  ```
  The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train].json`.
  ```
  python DP02_data_process.py --action get_type_label_map --dataset CWQ --split train
  ```
  The label map file will be saved as `data/CWQ/generation/label_maps/CWQ_train_type_label_map.json`.

### 3.3 **Prepare data for LLM**

- WebQSP
  ```
  python DP03_process_NQ.py --dataset_type WebQSP
  ```
  The merged data file will be saved as `LLMs/data_sexpr/WebQSP_Freebase_NQ_test[train]/examples.json`.
  ```
  python DP03_process_NQ_ID.py --dataset_type WebQSP
  ```
  The merged data file will be saved as `LLMs/data_id/WebQSP_Freebase_NQ_test[train]/examples.json`.

- CWQ
  ```
  python DP03_process_NQ.py --dataset_type CWQ
  ```
  The merged data file will be saved as `LLMs/data_sexpr/CWQ_Freebase_NQ_test[train]/examples.json`.
  ```
  python DP03_process_NQ_ID.py --dataset_type CWQ
  ```
  The merged data file will be saved as `LLMs/data_id/CWQ_Freebase_NQ_test[train]/examples.json`.


## 4. Fine-tuning LLM, Intent Detection Enhanced Model
The following is an example of LLaMa2-13b fine-tuning and retrieval (num_beam = 10) on WebQSP and LLaMa2-13b fine-tuning and retrieval (num_beam = 8) on CWQ, respectively.

### 4.1 Train and test LLM for Logical Form Generation

- WebQSP
  - Train LLMs for Logical Form Generation (The checkpoint data will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/checkpoint/`)
  - Beam-setting LLMs for Logical Form Generation (The generated_predictions.jsonl will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/`)
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/`)
=======
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/`)
=======
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam8_WebQSP_Freebase_NQ_test/`)
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  ```
  CUDA_VISIBLE_DEVICES=0 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data_sexpr --dataset WebQSP_Freebase_NQ_train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0  --quantization_bit 8 --plot_loss  --fp16 >> Sexpr_train_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  ```
  CUDA_VISIBLE_DEVICES=0 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data_sexpr --dataset WebQSP_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --quantization_bit 8 --num_beams 8 >> Sexpr_predbeam8_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  ```
  python FT_generator.py --data_file_name Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam8_WebQSP_Freebase_NQ_test/generated_predictions.jsonl
  ```

- CWQ
  - Train LLMs for Logical Form Generation (The checkpoint data will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/checkpoint/`)
  - Beam-setting LLMs for Logical Form Generation (The generated_predictions.jsonl will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/`)
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/`)
=======
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/`)
=======
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/`)
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  ```
  CUDA_VISIBLE_DEVICES=1 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data_sexpr --dataset CWQ_Freebase_NQ_train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --quantization_bit 8 --plot_loss --fp16 >> Sexpr_train_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```
  ```
  CUDA_VISIBLE_DEVICES=1 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data_sexpr --dataset CWQ_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/checkpoint --quantization_bit 8 --num_beams 8 >> Sexpr_predbeam8_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```
  ```
  python FT_generator.py --data_file_name Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/generated_predictions.jsonl
  ```

### 4.2 IDE: Intent Detection Enhanced Model

#### 4.2.1 ID: Train and test LLM for intent detection

- WebQSP
  - Train LLMs for Intent Detection (The checkpoint data will be saved as `Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/checkpoint/`)
  - Beam-setting LLMs for Intent Detection (The generated_predictions.jsonl will be saved as `Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/`)
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/`)
=======
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/`)
=======
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam12_WebQSP_Freebase_NQ_test/`)
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  ```
  CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data_id --dataset WebQSP_Freebase_NQ_train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0 --quantization_bit 8 --plot_loss  --fp16 >> ID_train_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  ```
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> bf0602a (first commit)
  CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/beam_output_eva_id.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data_id --dataset WebQSP_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --quantization_bit 8 --num_beams 10 >> ID_predbeam10_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  ```
  python FT_generator.py --data_file_name Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam10_WebQSP_Freebase_NQ_test/generated_predictions.jsonl
<<<<<<< HEAD
=======
=======
  CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/beam_output_eva_id.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data_id --dataset WebQSP_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --quantization_bit 8 --num_beams 12 >> ID_predbeam12_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  ```
  python FT_generator.py --data_file_name Reading/intent-detection/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam12_WebQSP_Freebase_NQ_test/generated_predictions.jsonl
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  ```

- CWQ
  - Train LLMs for Intent Detection (The checkpoint data will be saved as `Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/checkpoint/`)
  - Beam-setting LLMs for Intent Detection (The generated_predictions.jsonl will be saved as `Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/`)
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/`)
=======
<<<<<<< HEAD
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/`)
=======
  - Data processing (The beam_test_gen_statistics.json and beam_test_top_k_predictions.json will be saved as `Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/`)
>>>>>>> 145cb81 (first commit)
>>>>>>> bf0602a (first commit)
  ```
  CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data_id --dataset CWQ_Freebase_NQ_train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --quantization_bit 8 --plot_loss  --fp16 >> ID_train_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```
  ```
  CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/beam_output_eva_id.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data_id --dataset CWQ_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/checkpoint --quantization_bit 8 --num_beams 8 >> ID_predbeam8_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```
  ```
  python FT_generator.py --data_file_name Reading/intent-detection/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam8_CWQ_Freebase_NQ_test/generated_predictions.jsonl
  ```

#### 4.2.2 IE: Train and test reward model for intent enhancement

- WebQSP

  - Prepare the datasets for reward model (The reward model datasets will be saved as `IE/data/reward_datasets/sentiment_analysis_WebQSP/dev[train].tsv`.)
  ```
  cd IE/reward-model-datasets
  python WebQSP_rm_datasets.py
  ```

  - Train and test reward model for intent enhancement
  ```
  cd ../
  sh WebQSP_rm_train.sh
  nohup python WebQSP_rm_intent_enhancement.py >> nohup_WebQSP_rm_intent_enhancement.txt 2>&1 &
  cd ../
  python FT_generator_ID.py --data_file_name IE/logs/reward_model/sentiment_analysis_WebQSP/generated_predictions_WebQSP.jsonl
  ```

  - Merge the enhanced intent into the Logical Form (The new Logical Form file will be saved as `Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam_new/generated_predictions.jsonl`.)
  ```
  cd IE
  python WebQSP_intent_merging.py
  cd ../
  python FT_generator.py --data_file_name Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam_new/generated_predictions.jsonl
  ```

- CWQ

  - Prepare the datasets for reward model (The reward model datasets will be saved as `IE/data/reward_datasets/sentiment_analysis_CWQ/dev[train].tsv`.)
  ```
  cd IE/reward-model-datasets
  python CWQ_rm_datasets.py
  ```

  - Train and test reward model for intent enhancement
  ```
  cd ../
  sh CWQ_rm_train.sh
  nohup python CWQ_rm_intent_enhancement.py >> nohup_CWQ_rm_intent_enhancement.txt 2>&1 &
  cd ../
  python FT_generator_ID.py --data_file_name IE/logs/reward_model/sentiment_analysis_CWQ/generated_predictions_CWQ.jsonl
  ```

  - Merge the enhanced intent into the Logical Form (The new Logical Form file will be saved as `Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/generated_predictions.jsonl`.)
  ```
  cd IE
  python CWQ_intent_merging.py
  cd ../
  python FT_generator.py --data_file_name Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/generated_predictions.jsonl
  ```


## 5. Retrieval & Evaluation

### 5.1 **Evaluate KBQA result with Retrieval**

- WebQSP 

  - Evaluate KBQA result with entity-retrieval and relation-retrieval
  ```
  CUDA_VISIBLE_DEVICES=0 nohup python -u RE_webqsp.py --dataset WebQSP --pred_file Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam_new/beam_test_top_k_predictions.json >> RE_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```
  
  - Evaluate KBQA result with golden-entities and relation-retrieval
  ```
  CUDA_VISIBLE_DEVICES=1 nohup python -u RE_webqsp.py --dataset WebQSP --pred_file Reading/logical-form-generation/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam_new/beam_test_top_k_predictions.json --golden_ent >> RE_goldent_LLaMA2-13b_WebQSP_QLoRA_epoch100.txt 2>&1 &
  ```

- CWQ

  - Evaluate KBQA result with entity-retrieval and relation-retrieval
  ```
  CUDA_VISIBLE_DEVICES=0 nohup python -u RE_cwq.py --dataset CWQ --pred_file Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/beam_test_top_k_predictions.json >> RE_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```
  
  - Evaluate KBQA result with golden-entities and relation-retrieval
  ```
  CUDA_VISIBLE_DEVICES=1 nohup python -u RE_cwq.py --dataset CWQ --pred_file Reading/logical-form-generation/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam_new/beam_test_top_k_predictions.json --golden_ent >> RE_goldent_LLaMA2-13b_CWQ_QLoRA_epoch10.txt 2>&1 &
  ```


## 6. Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning), [SimCSE](https://github.com/princeton-nlp/SimCSE), [GMT-KBQA](https://github.com/HXX97/GMT-KBQA), [ChatKBQA](https://github.com/LHRLAB/ChatKBQA), [LLaMA-Factory](https://github.com/HXX97/GMT-KBQA) and [DECAF](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.