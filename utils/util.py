import os
import torch

def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def save(checkpoint_path, model_path, model, config):

    saved_file_path = os.path.join(checkpoint_path, model_path)
    
    ##### Create directory if it does not exist #####
    os.makedirs(saved_file_path, exist_ok=True) 
    
    saved_file_path = os.path.join(saved_file_path, "{}.pth".format(config))
    torch.save(model.state_dict(), saved_file_path)

    print("{} has been saved!".format(model_path))