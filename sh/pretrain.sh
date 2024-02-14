python pretrain.py --lm SciBERT --epochs 30 --text_lr 1e-5 --mol_lr 1e-5 --model AMOLE --dataset TanimotoSTM --data_path ./data/PubChemSTM --target_T 0.1 --T 0.1 --p_aug 0.5 --num_cand 50 --batch_size 45 --alpha 1.0 --device 0
