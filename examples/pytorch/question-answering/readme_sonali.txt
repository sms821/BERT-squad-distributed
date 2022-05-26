// run with trainer single node multi-gpu
python -u run_qa.py --model_name_or_path bert-large-uncased-whole-word-masking --dataset_name squad --do_train --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir bert_large --overwrite_output_dir

// run with trainer distributed training (single node in this case)
python -u -m torch.distributed.launch --nproc_per_node=4 run_qa.py --model_name_or_path bert-large-uncased-whole-word-masking --dataset_name squad --do_train --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir bert_large --overwrite_output_dir

# run with trainer single node single-gpu
python -u run_qa_no_trainer.py --model_name_or_path bert-large-uncased-whole-word-masking --dataset_name squad --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir bert_large
