#!/usr/bin/env python3
# coding=utf8
import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae
import pickle
import model
from data_manager import get_loader, make_spk_vector
from itertools import combinations

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_sp(feat_dir, num_mcep=36):
    feat_path = os.path.join(feat_dir, 'feats.p')
    with open(feat_path, 'rb') as f:
        sp, _, _, _, _ = pickle.load(f)
    return sp

def calc_parm_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def update_parm(opt_list, loss):
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--SI', type=int, default=0)
parser.add_argument('--I', type=int, default=0)
parser.add_argument('--LI', type=int, default=0)
parser.add_argument('--AC', type=int, default=0)
parser.add_argument('--SC', type=int, default=0)
parser.add_argument('--CC', type=int, default=0)
parser.add_argument('--GAN', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=0)

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

SPK_LIST = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2']
TOTAL_SPK_NUM = len(SPK_LIST)

SP_DICT_TRAIN = {
    spk_id:load_sp(os.path.join("data","train", spk_id)) 
    for spk_id in SPK_LIST
}

SP_DICT_DEV = dict()
for spk_id in SPK_LIST:
    sps = []
    for _, _, file_list in os.walk(os.path.join("data", "dev", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data", "dev", spk_id, file_id)
            coded_sp, f0, ap = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_DEV[spk_id] = sps

model_dir = args.model_dir
lr = 0.001
coef = {"rec": 1.0, "adv": 0.0, "kl": 0.1}

print(model_dir)
os.makedirs(model_dir + "/parm", exist_ok=True)

latent_dim = 8
is_MD = True if args.model_type == "MD" else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Enc = model.Encoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
Enc = Enc.to(device)
Enc_opt = optim.Adam(Enc.parameters(), lr=lr)
Enc_sch = optim.lr_scheduler.ExponentialLR(Enc_opt, 0.9)

print(calc_parm_num(Enc))

if is_MD:    
    Dec_group = dict()
    Dec_opt_group = dict()
    Dec_sch_group = dict()
    for spk_id in SPK_LIST:
        Dec_group[spk_id] = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
        Dec_group[spk_id] = Dec_group[spk_id].to(device) 
        Dec_opt_group[spk_id] = optim.Adam(Dec_group[spk_id].parameters(), lr=lr)
        Dec_sch_group[spk_id] = optim.lr_scheduler.ExponentialLR(Dec_opt_group[spk_id], 0.9)
else:
    Dec = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
    Dec = Dec.to(device)
    Dec_opt = optim.Adam(Dec.parameters(), lr=lr)
    Dec_sch = optim.lr_scheduler.ExponentialLR(Dec_opt, 0.9)

    print(Enc)
    print(Dec)

epochs = 100
print("Training Settings")
print("LR", lr)
print("Number of epochs", epochs)
print(".....................")
lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "total_loss"])

total_time = 0
min_dev_loss = float('inf')
min_epoch = 0
d_epoch = 1

batch_size = 8
n_frames = 128
for epoch in range(epochs + 1):
    print("EPOCH:", epoch)
    lm.init_stat()  

    start_time = time.time()
    Enc.train()
    if is_MD:
        for dec in Dec_group.values():
            dec.train()
    else:
        Dec.train()
    
    train_loader = get_loader(SP_DICT_TRAIN, batch_size, n_frames=n_frames, shuffle=True, is_MD=is_MD)

    for A_x, spk_idx in train_loader:
        if is_MD:
            spk_id = SPK_LIST[spk_idx]
            Dec = Dec_group[spk_id]
            Dec_opt = Dec_opt_group[spk_id]
            Dec_sch = Dec_sch_group[spk_id]
        
        batch_len = A_x.size()[0]
        A_y = make_spk_vector(spk_idx, TOTAL_SPK_NUM, batch_len, is_MD)
        
        z_mu, z_logvar, A_z = Enc(A_x, A_y)
     
