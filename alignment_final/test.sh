checkpoint=./bert-pretrain
init_checkpoint=$checkpoint/$(awk -F '\"' 'NR==1{print $2}' $checkpoint/checkpoint)
output_dir=./tmp/output
CUDA_VISIBLE_DEVICES=2 python3 run_alignment.py\
                               --bert_config_file=$checkpoint/bert_config.json\
                               --vocab_file=$checkpoint/vocab.txt\
                               --init_checkpoint=$init_checkpoint\
                               --output_dir=$output_dir\
                               --train_file=./tmp/data/train\
                               --eval_file=./tmp/data/test\
                               --predict_file=./tmp/data/test\
                               --do_train=false\
                               --do_predict=true\
                               --do_eval=false\
                               --do_export=false\
                               --max_seq_length=40\
                               --train_batch_size=512\
                               --predict_batch_size=64\
                               --eval_batch_size=64\
                               --learning_rate=5e-5\
                               --warmup_proportion=0.1\
                               --num_train_epochs=30\
                               --save_checkpoints_steps=1000\
                               
                               #--bert_config_file=./tmp/data/bert_config.json\
