import sys
sys.path.append('.')

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

# Call Trainer class
from trainer import trainer

from utils.util import freeze_network, cycle_index
from utils.bert import prepare_text_tokens_kd
from utils.chem import calc_tanimoto
from utils import argument


class AMOLE_Trainer(trainer):

    def __init__(self, args):
        trainer.__init__(self, args)

        ##### Build Language Model and Graph Neural Networks #####
        self.build_LM(args)
        self.build_GNN(args)
        
        self.text2latent = nn.Linear(self.text_dim, args.SSL_emb_dim).to(self.device)
        self.mol2latent = nn.Linear(self.molecule_dim, args.SSL_emb_dim).to(self.device)

        # Freeze Teacher Network
        self.ttext2latent = copy.deepcopy(self.text2latent)
        self.ttext_model = copy.deepcopy(self.text_model)
        freeze_network(self.ttext2latent)
        freeze_network(self.ttext_model)
        self.distill_loss = nn.MSELoss()

        ##### Freeze model parameters #####
        if args.representation_frozen:
            freeze_network(self.text_model)
            freeze_network(self.molecule_model)
            model_param_group = [
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        else:
            model_param_group = [
                {"params": self.text_model.parameters(), "lr": args.text_lr},
                {"params": self.molecule_model.parameters(), "lr": args.mol_lr},
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        
        self.optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
        self.optimal_loss = 1e10        
    

    def get_text_repr_kd(self, text):
        """
        Get representation of molecular textual description with Language Models
        """
        text_tokens_ids, knowledge_masks, sentence_masks = prepare_text_tokens_kd(
            device=self.device, description=text, tokenizer=self.text_tokenizer, max_seq_len=self.args.max_seq_len)
        
        # Get Representation of Original Sentence
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=knowledge_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)

        # Get Representation of Sentence including External Knowledge
        knowledge_output = self.ttext_model(input_ids=text_tokens_ids, attention_mask=sentence_masks)
        knowledge_repr = knowledge_output["pooler_output"]
        knowledge_repr = self.ttext2latent(knowledge_repr)
        
        return text_repr, knowledge_repr


    def update_teacher_model(self):
        # Update text model
        for s_params, t_params in zip(self.text_model.parameters(), self.ttext_model.parameters()):
            t_params.data = s_params.data
        # Update text2latent model
        for s_params, t_params in zip(self.text2latent.parameters(), self.ttext2latent.parameters()):
            t_params.data = s_params.data


    def train(self):

        for epoch in range(1, self.args.epochs + 1):
            
            start_time = time.time()

            accum_loss, accum_distill_loss, accum_acc = 0, 0, 0

            for bc, samples in enumerate(tqdm(self.dataloader)):

                self.optimizer.zero_grad()

                description = samples[0]
                molecule = samples[1]
                rand_molecule = samples[2]
                aux_description = samples[3]
                
                ##### Forward Pass: Language Model #####
                description_repr = self.get_text_repr(description)

                ##### Forward Pass: Molecule Model #####
                molecule_repr = self.get_molecule_repr(rand_molecule)

                ##### Get Tanimoto Similarity between Molecules #####
                tanimoto_mat = calc_tanimoto(molecule.fp.to(self.device), rand_molecule.fp)

                loss_01 = self.calc_S2P_Loss(description_repr, molecule_repr, tanimoto_mat)
                loss_02 = self.calc_S2P_Loss(molecule_repr, description_repr, tanimoto_mat.T)

                ##### Auxiliary Forward Pass: Language Model #####
                aux_description_repr, aux_knowledge_repr = self.get_text_repr_kd(aux_description)

                distill_loss = self.distill_loss(aux_description_repr, aux_knowledge_repr)

                loss = (loss_01 + loss_02) / 2
                total_loss = loss + self.args.alpha * distill_loss
                
                total_loss.backward()
                self.optimizer.step()

                self.update_teacher_model()

                accum_loss += loss.item()
                accum_distill_loss += distill_loss.item()

            temp_loss = accum_loss
            if temp_loss < self.optimal_loss:
                self.optimal_loss = temp_loss
                self.save_model(epoch=epoch)
            print("[Epoch {}] CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(epoch, accum_loss, accum_acc, time.time() - start_time))


if __name__ == "__main__":
    
    args, unknown = argument.parse_args()

    from models import AMOLE_Trainer
    model_trainer = AMOLE_Trainer(args)

    model_trainer.train()