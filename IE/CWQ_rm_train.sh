nohup python train_reward_model.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/reward_datasets/sentiment_analysis_CWQ/train.tsv" \
    --dev_path "data/reward_datasets/sentiment_analysis_CWQ/dev.tsv" \
    --save_dir "checkpoints/reward_model/sentiment_analysis_CWQ" \
    --img_log_dir "logs/reward_model/sentiment_analysis_CWQ" \
    --img_log_name "ERNIE Reward Model" \
    --batch_size 3 \
    --max_seq_len 1024 \
    --learning_rate 8e-6 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 5 \
    --device "cuda:1" \
    >> CWQ_rm_train.txt 2>&1 &
