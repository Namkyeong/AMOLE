import numpy as np
import torch
import copy

'''
This code is borrowed from MoleculeSTM for BERT Training
Paper: https://arxiv.org/abs/2212.10789
Code: https://github.com/chao1224/MoleculeSTM
'''

def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = np.vstack([o[0] for o in tokens_outputs])
    masks = np.vstack([o[1] for o in tokens_outputs])
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    masks = torch.Tensor(masks).bool().to(device)
    return tokens_ids, masks


def preprocess_each_sentence_kd(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()
    
    # Mask the knowledge after [SEP] Token (Token id: 103)
    knowledge_mask = copy.deepcopy(attention_mask)
    knowledge_mask[np.where(input_ids == 103)[0][0] + 1:] = 0

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    knowledge_masks = padarray(knowledge_mask, max_seq_len)
    return [sentence_tokens_ids, knowledge_masks, sentence_masks]


def prepare_text_tokens_kd(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence_kd(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = np.vstack([o[0] for o in tokens_outputs])
    knowledge_masks = np.vstack([o[1] for o in tokens_outputs])
    sentence_masks = np.vstack([o[2] for o in tokens_outputs])
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    knowledge_masks = torch.Tensor(knowledge_masks).bool().to(device)
    sentence_masks = torch.Tensor(sentence_masks).bool().to(device)
    return tokens_ids, knowledge_masks, sentence_masks