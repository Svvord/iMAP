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

import shap

########## DATA PREPROCESS ##########
def sub_data_preprocess(adata: sc.AnnData, n_top_genes: int=None) -> sc.AnnData:
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes)
    return adata


def data_preprocess(adata: sc.AnnData, key: str='batch', n_top_genes: int=None, n_batch: int=2) -> sc.AnnData:
    print('Preprocessing Data in Different Batches...')
    if (n_batch > len(set(adata.obs[key]))) or n_batch==0:
        print('Establishing Adata for Next Step...')
        hv_adata = sub_data_preprocess(adata, n_top_genes=n_top_genes)
        hv_adata = hv_adata[:, hv_adata.var['highly_variable']]
        print('PreProcess Done.')
        return hv_adata

    batch_dataset = []
    for batch in sorted(list(set(adata.obs[key]))):
        batch_dataset.append(sub_data_preprocess(adata[adata.obs[key] == batch].copy()))

    print('Acquiring Common Variables...')
    highly_vars = []
    common_var = set()
    for dataset in batch_dataset:
        if common_var:
            common_var &= set(dataset.var_names)
        else:
            common_var = set(dataset.var_names)
        highly_vars.append(list(dataset.var_names[dataset.var['highly_variable']]))

    print('Acquiring Common Highly Variables...')
    _ = {}
    for highly_var in highly_vars:
        for gene in set(highly_var):
            if gene not in _:
                _[gene] = 1
            else:
                _[gene] += 1

    processed_hv = []
    for i, highly_var in enumerate(highly_vars):
        tmp = []
        for gene in highly_var:
            if _[gene] >= n_batch:
                tmp.append(gene)
        processed_hv.append(tmp)
    highly_vars = processed_hv

    common_high_var = set()
    for highly_var in highly_vars:
        if common_high_var:
            common_high_var |= set(highly_var)
        else:
            common_high_var = set(highly_var)
    common_high_var &= common_var
    common_high_var = sorted(list(common_high_var))

    print('Establishing Adata for Next Step...')
    if len(common_high_var) < 1500:
        hv_adata = sub_data_preprocess(adata, n_top_genes=n_top_genes)
        hv_adata = hv_adata[:, hv_adata.var['highly_variable']]
        print('PreProcess Done.')
        return hv_adata

    hv_adata = None
    obs_names = None
    for dataset in batch_dataset:
        var_index = list(dataset.var_names)
        foo_ = np.array(dataset.X)[:, [var_index.index(item) for item in common_high_var]]
        if hv_adata is None:
            hv_adata = foo_
            obs_names = np.array(dataset.obs_names)
        else:
            hv_adata = np.r_[hv_adata, foo_]
            obs_names = np.r_[obs_names, np.array(dataset.obs_names)]
    hv_adata = sc.AnnData(pd.DataFrame(hv_adata, columns=common_high_var))
    hv_adata.obs_names = obs_names

    techs = []
    for i, tech in enumerate(sorted(list(set(adata.obs[key])))):
        techs.extend([tech] * len(batch_dataset[i]))

    techs = np.array(techs)
    hv_adata.obs[key] =pd.DataFrame(techs, index=np.array(hv_adata.obs_names))
    print('PreProcess Done.')
    return hv_adata


########## UTILITY FUNCTIONS ##########
def extract_data(data: sc.AnnData, key: str, batches, orders=None):
    adata_values = [np.array(data.X[data.obs[key] == batch]) for batch in batches]
    if orders is None:
        std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
        orders = np.argsort(std_)[::-1]
    else:
        orders = np.array([batches.index(item) for item in orders])
    return adata_values, orders


########## CLASS SINGLE CELL DATASET ##########
class ScDataset(Dataset):
    def __init__(self):
        self.dataset = []
        self.variable = None
        self.labels = None
        self.transform = None
        self.sample = None
        self.trees = []

    def __len__(self):
        return 10 * 1024

    def __getitem__(self, index):
        dataset_samples = []
        for j, dataset in enumerate(self.dataset):
            rindex1 = np.random.randint(len(dataset))
            rindex2 = np.random.randint(len(dataset))
            alpha = np.random.uniform(0, 1)
            sample = alpha*dataset[rindex1] + (1-alpha)*dataset[rindex2]
            dataset_samples.append(sample)
        return dataset_samples


########## PLOT FUNCTIONS ##########
def cat_data(data_A: np.float32, data_B: np.float32, labels: List[List[int]]=None):
    data = np.r_[data_A, data_B]
    if labels is None:
        label = np.zeros(len(data_A)+len(data_B))
        label[-len(data_B):] = 1
        label = np.array([label]).T
    else:
        label = np.r_[labels[0], labels[1]]
    return data, label


########## NEURAL NETWORK UTILITY ##########
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_size, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Mish(),
            nn.Linear(512, 1024),
            Mish(),
            nn.Linear(1024, data_size),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(n_classes, 512),
            Mish(),
            nn.Linear(512, 1024),
            Mish(),
            nn.Linear(1024, data_size),
        )

    def forward(self, ec, es):
        return self.relu(self.decoder(torch.cat((ec, es), dim=-1))+self.decoder2(es))

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
    prob_interpolated, _ = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

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
        self.adv_layer = nn.Sequential(
            nn.Linear(512, 1, bias=False),
        )
        # Classify layers
        self.cls_layer = nn.Sequential(
            nn.Linear(512, n_classes, bias=False),
        )

    def forward(self, data):
        out = self.model(data)
        validity = self.adv_layer(out)
        classify = self.cls_layer(out)
        return validity, classify

def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def iMAP(
    adata,
    key = 'batch',
    n_epochs = 600,
    num_workers=0,
    lr = 0.0002,
    b1 = 0.5,
    b2 = 0.999,
    latent_dim = 256,
    n_critic = 5,
    lambda_co = 3,
    lambda_rc = 1,
    seed = 8,
    ):
    setup_seed(seed)
    batches = sorted(list(set(adata.obs[key])))
    scd = ScDataset()
    scd.variable = np.array(adata.var_names)
    adata_values, orders = extract_data(adata, key, batches, orders=None)

    ####20200408
    obs_names = [np.array(adata.obs_names[adata.obs[key] == batch]) for batch in batches]

    ec_obs_names = None
    for item in orders:
        if ec_obs_names is None:
            ec_obs_names = obs_names[item]
        else:
            ec_obs_names = np.r_[ec_obs_names, obs_names[item]]

    ####

    print('Step 1: Calibrating Celltype...')
    scd.dataset = [adata_values[i] for i in orders]

    dataloader = DataLoader(
        dataset = scd,
        batch_size=512,
        num_workers=num_workers,
    )

    global data_size
    global n_classes

    data_size = scd.dataset[0].shape[1]
    n_classes = len(scd.dataset)

    # Initialize generator and discriminator
    EC = Encoder(latent_dim)
    Dec = Decoder(latent_dim + n_classes)
    D = Discriminator()
    mse_loss = torch.nn.MSELoss()

    if cuda:
        D.cuda()
        EC.cuda()
        Dec.cuda()
        mse_loss.cuda()

    # Initialize weights
    D.apply(weights_init_normal)
    EC.apply(weights_init_normal)
    Dec.apply(weights_init_normal)

    optimizer_Dec = torch.optim.Adam(Dec.parameters(), lr=lr, betas=(b1, b2))
    optimizer_EC = torch.optim.Adam(EC.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        Dec.train()
        EC.train()
        D.train()

        for i, data in enumerate(dataloader):
            datum = [Variable(item.type(FloatTensor)) for item in data]
            batch_size = datum[0].shape[0]

            ES_data1 = -np.zeros((n_classes * batch_size, n_classes))
            for j in range(n_classes):
                ES_data1[j*batch_size:(j+1)*batch_size, j] = 1
            ES_data1 = Variable(torch.tensor(ES_data1).type(FloatTensor))
            ES_data2 = -np.zeros((n_classes * batch_size, n_classes))
            ES_data2[np.arange(n_classes*batch_size),np.random.randint(n_classes, size=n_classes*batch_size)] = 1
            ES_data2 = Variable(torch.tensor(ES_data2).type(FloatTensor))

            real_data = torch.cat(datum, dim=0)
            optimizer_D.zero_grad()
            gen_data = Dec(EC(real_data), ES_data1)
            # Loss for real images
            real_validity, pred_label  = D(real_data)
            fake_validity, _  = D(gen_data)
            # Compute W-div gradient penalty
            div_gp = calculate_gradient_penalty(real_data, gen_data, D)
            # Adversarial loss
            loss_D_cls = criterion_cls(pred_label, ES_data1)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10*div_gp + loss_D_cls
            d_loss.backward()
            optimizer_D.step()

            optimizer_Dec.zero_grad()
            optimizer_EC.zero_grad()
            if i % n_critic == 0:
                loss1_data1 = torch.cat(datum, dim=0)
                loss4 = mse_loss(EC(loss1_data1), EC(Dec(EC(loss1_data1), ES_data2)))
                ae_loss = mse_loss(Dec(EC(loss1_data1), ES_data1), loss1_data1)

                gen_data = Dec(EC(real_data), ES_data1)
                fake_validity, _ = D(gen_data)
                g_loss = - torch.mean(fake_validity)

                gen_data = Dec(EC(real_data), ES_data2)
                _, pred_label = D(gen_data)
                loss_D_cls = criterion_cls(pred_label, ES_data2)

                all_loss = (lambda_co * loss4) + (lambda_rc * ae_loss) + 0.01*g_loss + 0.01*loss_D_cls
                all_loss.backward()

                optimizer_Dec.step()
                optimizer_EC.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [Reconstruction loss: %f] [Cotent loss: %f] [G loss: %f]"
            % (epoch+1, n_epochs,
               d_loss.item(),
               ae_loss.item(),
               loss4.item(),
               g_loss.item(),
              )
        )

    D.eval()
    Dec.eval()
    EC.eval()
    with torch.no_grad():
        data = Variable(FloatTensor(scd.dataset[0]), volatile=True)
        label = np.full((len(scd.dataset[0]),1), batches[orders[0]])
        static_sample = EC(data)
        transform_data = static_sample.cpu().detach().numpy()
        for j in range(1, len(scd.dataset)):
            data = Variable(FloatTensor(scd.dataset[j]), volatile=True)
            static_sample = EC(data)
            fake_data = static_sample.cpu().detach().numpy()
            fake_label = np.full((len(scd.dataset[j]),1), batches[orders[j]])
            transform_data, label = cat_data(transform_data, fake_data, [label, fake_label])
        ec_data = sc.AnnData(transform_data)
        ec_data.obs_names = ec_obs_names
        ec_data.obs[key] = label

    return EC, ec_data


def iMAP_fast(
    adata,
    key = 'batch',
    n_epochs = 150,
    num_workers=0,
    lr = 0.0005,
    b1 = 0.5,
    b2 = 0.999,
    latent_dim = 256,
    n_critic = 5,
    lambda_co = 3,
    lambda_rc = 1,
    seed = 8,
    ):
    setup_seed(seed)
    batches = sorted(list(set(adata.obs[key])))
    scd = ScDataset()
    scd.variable = np.array(adata.var_names)
    adata_values, orders = extract_data(adata, key, batches, orders=None)

    obs_names = [np.array(adata.obs_names[adata.obs[key] == batch]) for batch in batches]
    ec_obs_names = None
    for item in orders:
        if ec_obs_names is None:
            ec_obs_names = obs_names[item]
        else:
            ec_obs_names = np.r_[ec_obs_names, obs_names[item]]


    print('Step 1: Calibrating Celltype...')
    scd.dataset = [adata_values[i] for i in orders]

    dataloader = DataLoader(
        dataset = scd,
        batch_size=512,
        num_workers=num_workers,
    )

    global data_size
    global n_classes

    data_size = scd.dataset[0].shape[1]
    n_classes = len(scd.dataset)

    # Initialize generator and discriminator
    EC = Encoder(latent_dim)
    Dec = Decoder(latent_dim + n_classes)
    mse_loss = torch.nn.MSELoss()

    if cuda:
        EC.cuda()
        Dec.cuda()
        mse_loss.cuda()

    # Initialize weights
    EC.apply(weights_init_normal)
    Dec.apply(weights_init_normal)

    optimizer_Dec = torch.optim.Adam(Dec.parameters(), lr=lr, betas=(b1, b2))
    optimizer_EC = torch.optim.Adam(EC.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        Dec.train()
        EC.train()

        for i, data in enumerate(dataloader):
            datum = [Variable(item.type(FloatTensor)) for item in data]
            batch_size = datum[0].shape[0]

            ES_data1 = -np.zeros((n_classes * batch_size, n_classes))
            for j in range(n_classes):
                ES_data1[j*batch_size:(j+1)*batch_size, j] = 1
            ES_data1 = Variable(torch.tensor(ES_data1).type(FloatTensor))
            ES_data2 = -np.zeros((n_classes * batch_size, n_classes))
            ES_data2[np.arange(n_classes*batch_size),np.random.randint(n_classes, size=n_classes*batch_size)] = 1
            ES_data2 = Variable(torch.tensor(ES_data2).type(FloatTensor))

            optimizer_Dec.zero_grad()
            optimizer_EC.zero_grad()

            loss1_data1 = torch.cat(datum, dim=0)
            loss4 = mse_loss(EC(loss1_data1), EC(Dec(EC(loss1_data1), ES_data2)))
            ae_loss = mse_loss(Dec(EC(loss1_data1), ES_data1), loss1_data1)

            all_loss = (lambda_co * loss4) + (lambda_rc * ae_loss)
            all_loss.backward()

            optimizer_Dec.step()
            optimizer_EC.step()

        print(
            "[Epoch %d/%d] [Reconstruction loss: %f] [Cotent loss: %f]"
            % (epoch+1, n_epochs,
               ae_loss.item(),
               loss4.item(),
              )
        )

    Dec.eval()
    EC.eval()
    with torch.no_grad():
        data = Variable(FloatTensor(scd.dataset[0]), volatile=True)
        label = np.full((len(scd.dataset[0]),1), batches[orders[0]])
        static_sample = EC(data)
        transform_data = static_sample.cpu().detach().numpy()
        for j in range(1, len(scd.dataset)):
            data = Variable(FloatTensor(scd.dataset[j]), volatile=True)
            static_sample = EC(data)
            fake_data = static_sample.cpu().detach().numpy()
            fake_label = np.full((len(scd.dataset[j]),1), batches[orders[j]])
            transform_data, label = cat_data(transform_data, fake_data, [label, fake_label])
        ec_data = sc.AnnData(transform_data)
        ec_data.obs_names = ec_obs_names
        ec_data.obs[key] = label

    return EC, ec_data

def explain_importance(EC, adata, celltypes, latent_dim = 256, batch_size=2048, lr = 0.0002, b1 = 0.5, b2 = 0.999, n_epochs = 200, print_loss = False, n_samples=200, to_explain = None):
    data = np.array(adata.X)
    if to_explain is None:
        to_explain = Variable(FloatTensor(data))
    else:
        to_explain = Variable(FloatTensor(to_explain))
    n_celltype = len(set(celltypes))
    cse_loss = torch.nn.CrossEntropyLoss()

    class EcClassifier(nn.Module):
        def __init__(self, EC, latent_dim, n_celltype):
            super(EcClassifier, self).__init__()
            self.Ec = EC
            self.encoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.BatchNorm1d(128),
                Mish(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                Mish(),
                nn.Linear(64, n_celltype),
            )

        def forward(self, x):
            return self.encoder(self.Ec(x))

    class EcLD:
        def __init__(self, data, celltypes):
            self.dataset = data
            self.labels = celltypes
            self.maps = sorted(list(set(self.labels)))

        def __getitem__(self, index):
            return self.dataset[index], self.maps.index(self.labels[index])

        def __len__(self):
            return (len(self.dataset))


    ecld = EcLD(data, celltypes)

    dataloader = DataLoader(
        dataset = ecld,
        batch_size=batch_size,
        shuffle=True,
    )
    for para in EC.parameters():
        para.requires_grad = False

    ec_explainer = EcClassifier(EC, latent_dim, n_celltype)

    if cuda:
        ec_explainer.encoder.cuda()
        cse_loss.cuda()
    # Initialize weights
    ec_explainer.encoder.apply(weights_init_normal)
    optimizer_Ec_explainer = torch.optim.Adam(ec_explainer.encoder.parameters(), lr=lr, betas=(b1, b2))
    ec_explainer.Ec.eval()
    ec_explainer.encoder.train()
    for epoch in range(n_epochs):
        for i, (data, labels) in enumerate(dataloader):
            data = Variable(data.type(FloatTensor))
            labels = Variable(torch.tensor(labels).type(LongTensor))
            optimizer_Ec_explainer.zero_grad()
            loss = cse_loss(ec_explainer(data), labels)
            loss.backward()
            optimizer_Ec_explainer.step()
        if print_loss:
            print(loss.item())
    ec_explainer.eval()

    print('Start Calculating Gene Importance...')
    e = shap.GradientExplainer((ec_explainer, ec_explainer.Ec.encoder[0]), Variable(FloatTensor(data)))
    ec_importance = []

    for i,item in enumerate(to_explain):
        if i%100 == 99:
            print(i+1, 'done...')
        shap_values,indexes = e.shap_values(to_explain[i:i+1], ranked_outputs=1, nsamples=n_samples)
        ec_importance.append(shap_values[0][0])

    print('Saving Files...')
    outputs = pd.DataFrame(np.array(ec_importance), index=adata.obs_names)
    outputs.to_csv('./ec_importance.csv', header=adata.var_names)

def contrast_explain_importance(adata, celltypes, batch_size=2048, lr = 0.0002, b1 = 0.5, b2 = 0.999, n_epochs = 200, print_loss = False, n_samples=200, to_explain = None):
    data = np.array(adata.X)
    latent_dim = len(data[0])
    if to_explain is None:
        to_explain = Variable(FloatTensor(data))
    else:
        to_explain = Variable(FloatTensor(to_explain))
    n_celltype = len(set(celltypes))
    cse_loss = torch.nn.CrossEntropyLoss()

    class EcClassifier(nn.Module):
        def __init__(self, latent_dim, n_celltype):
            super(EcClassifier, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.BatchNorm1d(128),
                Mish(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                Mish(),
                nn.Linear(64, n_celltype),
            )

        def forward(self, x):
            return self.encoder(x)

    class EcLD:
        def __init__(self, data, celltypes):
            self.dataset = data
            self.labels = celltypes
            self.maps = sorted(list(set(self.labels)))

        def __getitem__(self, index):
            return self.dataset[index], self.maps.index(self.labels[index])

        def __len__(self):
            return (len(self.dataset))


    ecld = EcLD(data, celltypes)

    dataloader = DataLoader(
        dataset = ecld,
        batch_size=batch_size,
        shuffle=True,
    )

    ec_explainer = EcClassifier(latent_dim, n_celltype)

    if cuda:
        ec_explainer.encoder.cuda()
        cse_loss.cuda()
    # Initialize weights
    ec_explainer.encoder.apply(weights_init_normal)
    optimizer_Ec_explainer = torch.optim.Adam(ec_explainer.encoder.parameters(), lr=lr, betas=(b1, b2))

    ec_explainer.encoder.train()
    for epoch in range(n_epochs):
        for i, (data, labels) in enumerate(dataloader):
            data = Variable(data.type(FloatTensor))
            labels = Variable(torch.tensor(labels).type(LongTensor))
            optimizer_Ec_explainer.zero_grad()
            loss = cse_loss(ec_explainer(data), labels)
            loss.backward()
            optimizer_Ec_explainer.step()
        if print_loss:
            print(loss.item())
    ec_explainer.eval()

    print('Start Calculating Gene Importance...')
    e = shap.GradientExplainer((ec_explainer, ec_explainer.encoder[0]), Variable(FloatTensor(data)))
    ec_importance = []

    for i,item in enumerate(to_explain):
        if i%100 == 99:
            print(i+1, 'done...')
        shap_values,indexes = e.shap_values(to_explain[i:i+1], ranked_outputs=1, nsamples=n_samples)
        ec_importance.append(shap_values[0][0])

    print('Saving Files...')
    outputs = pd.DataFrame(np.array(ec_importance), index=adata.obs_names)
    outputs.to_csv('./batch_importance.csv', header=adata.var_names)
