import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from typing import List
from collections import OrderedDict
import random
import scanpy as sc

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.autograd as autograd
from torch.autograd import Variable

from annoy import AnnoyIndex

########## DATA PREPROCESS ##########
from imap.imap import sub_data_preprocess
from imap.imap import data_preprocess

########## UTILITY FUNCTIONS ##########
from imap.imap import extract_data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    
def normalize(data: np.float32) -> np.float32:
    norm = data#(np.exp2(data)-1)
    return norm# / np.array([np.sqrt(np.sum(np.square(norm), axis=1))]).T

def acquire_pairs(X, Y, k, metric):
    X = normalize(X)
    Y = normalize(Y)

    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat>0))]
    return pairs

def create_pairs_dict(pairs):
    pairs_dict = {}
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict

########## CLASS SINGLE CELL DATASET ##########
class ScDataset(Dataset):
    def __init__(self, n_sample=3000):
        self.dataset = []
        self.cali_dataset = []
        self.variable = None
        self.anchor_index = 0
        self.query_index = 1
        self.pairs = None
        self.labels = None
        self.transform = None
        self.sample = None
        self.metric = 'euclidean'
        self.k1 = None
        self.k2 = None
        self.n_sample = n_sample


    def change_dataset(self, index: int=1):
        self.query_index = index


    def acquire_anchor(self, index: int=0):
        self.anchor_index = index


    def calculate_mnn_pairs(self):
        tmp = np.arange(len(self.dataset[self.anchor_index]))
        np.random.shuffle(tmp)
        self.sample = self.cali_dataset[self.anchor_index][tmp[:self.n_sample]]
        ####
        tmp2 = np.arange(len(self.dataset[self.query_index]))
        np.random.shuffle(tmp2)
        self.query_sample = self.cali_dataset[self.query_index][tmp2[:self.n_sample]]
        ####
        
        if (self.k1 is None) or (self.k2 is None):
            self.k2 = int(min(len(self.sample), len(self.query_sample))/100)
            self.k1 = max(int(self.k2/2), 1)
        
        print('Calculating Anchor Pairs...')
        anchor_pairs = acquire_pairs(self.sample, self.sample, self.k2, self.metric)
        print('Calculating Query Pairs...')
        query_pairs = acquire_pairs(self.query_sample, self.query_sample, self.k2, self.metric)
        print('Calculating KNN Pairs...')
        pairs = acquire_pairs(self.sample, self.query_sample, self.k1, self.metric)
        print('Calculating Random Walk Pairs...')
        anchor_pairs_dict = create_pairs_dict(anchor_pairs)
        query_pairs_dict = create_pairs_dict(query_pairs)
        pair_plus = []
        for x, y in pairs:
            start = (x, y)
            for i in range(50):
                pair_plus.append(start)
                start = (random.choice(anchor_pairs_dict[start[0]]), random.choice(query_pairs_dict[start[1]]))

        self.datasetA = self.dataset[self.query_index][tmp2[:self.n_sample]][[y for x,y in pair_plus], :]
        self.datasetB = self.dataset[self.anchor_index][tmp[:self.n_sample]][[x for x,y in pair_plus], :]
        print('Done.')

    def __len__(self):
        return 10*1024


    def __getitem__(self, index):
        return random.choice(self.datasetA), random.choice(self.datasetB)

########## PLOT FUNCTIONS ##########
from imap.imap import cat_data


########## NEURAL NETWORK UTILITY ##########
from imap.imap import Mish
from imap.imap import weights_init_normal

def calculate_gradient_penalty(real_data, fake_data, D):
    eta = torch.FloatTensor(real_data.size(0),1).uniform_(0,1)
    eta = eta.expand(real_data.size(0), real_data.size(1))
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_data + ((1 - eta) * fake_data)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def train(scd, n_dataset, n_epochs):
    scd.change_dataset(n_dataset)
    scd.calculate_mnn_pairs()

    n_epochs = n_epochs
    n_classes = 2
    data_size = scd.dataset[0].shape[1]
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    latent_dim = 256
    n_critic = 100


    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    dataloader = DataLoader(
        dataset = scd,
        batch_size=1024,
    )


    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.relu = nn.ReLU(inplace=True)
            self.encoder = nn.Sequential(
                nn.Linear(data_size, 1024),
                nn.BatchNorm1d(1024),
                Mish(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                Mish(),
                nn.Linear(512, latent_dim),
                nn.BatchNorm1d(latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.BatchNorm1d(512),
                Mish(),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                Mish(),
                nn.Linear(1024, data_size),
            )

        def forward(self, x):
            latent_data = self.encoder(x)
            gen_data = self.decoder(latent_data)
            return self.relu(gen_data + x)


    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(data_size, 512),
                Mish(),
                nn.Linear(512, 512),
                Mish(),
            )

            # Output layers
            self.adv_layer = nn.Sequential(nn.Linear(512, 1))

        def forward(self, data):
            out = self.model(data)
            validity = self.adv_layer(out)
            return validity

    # Initialize generator and discriminator
    G_AB = Generator()
    D_B = Discriminator()

    if cuda:
        G_AB.cuda()
        D_B.cuda()

    # Initialize weights
    G_AB.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    optimizer_G_AB = torch.optim.Adam(G_AB.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        G_AB.train()
        for i, (data_A, data_B) in enumerate(dataloader):
            batch_size = data_A.shape[0]

            # Configure input
            real_data = Variable(data_B.type(FloatTensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D_B.zero_grad()
            z = Variable(data_A.type(FloatTensor))
            gen_data = G_AB(z)


            # Loss for real images
            real_validity  = D_B(real_data)
            fake_validity  = D_B(gen_data)


            # Compute W-div gradient penalty
            div_gp = calculate_gradient_penalty(real_data, gen_data, D_B)

            # Adversarial loss
            db_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10*div_gp
            db_loss.backward()
            optimizer_D_B.step()


            # -----------------
            #  Train Generator
            # -----------------

            if i % n_critic == 0:
                optimizer_G_AB.zero_grad()
                z = Variable(data_A.type(FloatTensor), requires_grad=True)
                gen_data = G_AB(z)
                fake_validity = D_B(gen_data)
                gab_loss = -torch.mean(fake_validity)
                gab_loss.backward()

                optimizer_G_AB.step()


        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, n_epochs,
               db_loss.item(),
               gab_loss.item(),
              )
        )


    G_AB.eval()
    with torch.no_grad():
        z = Variable(FloatTensor(scd.dataset[scd.query_index]))
        static_sample = G_AB(z)
        fake_data = static_sample.cpu().detach().numpy()
    return fake_data


########## WORKFLOW ##########
def integrate_data(data: sc.AnnData, cali_data, key='batch', n_top_genes=None, pp=False, n_epochs=150, inc=False, orders=None, metric='angular', k1=None, k2=None, n_batch=2, n_sample=3000, seed=8):
    if seed is not None:
        setup_seed(seed)
    adata = data_preprocess(data, key, n_top_genes=n_top_genes, n_batch=n_batch) if pp else data
    print('Step 2: Blending with GAN...')
    print('Adata Info: ')
    print(adata)
    batches = sorted(list(set(adata.obs[key])))
    cali_batches = sorted(list(set(cali_data.obs[key])))
    scd = ScDataset(n_sample)
    scd.metric = metric
    scd.k1 = k1
    scd.k2 = k2
    scd.variable = np.array(adata.var_names)
    adata_values, orders = extract_data(adata, key, batches, orders=orders)

    print('Orders:','<-'.join(batches[i] for i in orders))
    scd.dataset = [adata_values[i] for i in orders]
    
    cali_adata_values, cali_orders = extract_data(cali_data, key, cali_batches, orders=[batches[i] for i in orders])
    scd.cali_dataset = [cali_adata_values[i] for i in cali_orders]

    scd.transform = np.copy(scd.dataset[scd.anchor_index])
    for i in range(1, len(scd.dataset)):
        print(f'Merging dataset {batches[orders[i]]} to {batches[orders[0]]}')
        fake_data = train(scd, i, n_epochs=n_epochs)
        if inc:
            scd.dataset[scd.anchor_index] = np.r_[scd.dataset[scd.anchor_index], fake_data]
            scd.cali_dataset[scd.anchor_index] = np.r_[scd.cali_dataset[scd.anchor_index], scd.cali_dataset[i]]
        scd.transform = np.r_[scd.transform, fake_data]

    start = 0
    output_results = np.zeros(scd.transform.shape)
    for i in orders:
        length = np.sum(adata.obs[key] == batches[i])
        output_results[adata.obs[key] == batches[i]] = scd.transform[start:start+length]
        start += length

    return output_results
