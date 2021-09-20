python train.py \
	--from_pretrained ./pretrained_model \
	--max_seq_len 128 \
	--batch_size 16 \
	--epoch 10 \
	--lr 5e-6 \
	--data_dir ./data \
	--save_model_dir saved_model \
	--max_steps $((8544*10/16))
