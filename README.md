
# iMAP - Integration of multiple single-cell datasets by adversarial paired transfer networks

### Installation

#### 1. Prerequisites

<ul>
    <li>Install Python >= 3.6. Typically, you should use the Linux system and install a newest version of <a href='https://www.anaconda.com/'>Anaconda</a> or <a href = 'https://docs.conda.io/en/latest/miniconda.html'> Miniconda </a>.</li>
    <li>Install pytorch >= 1.1.0. To obtain the optimal performance of deep learning-based models, you should have a Nivia GPU and install the appropriate version of CUDA. (We tested with CUDA >= 9.0)</li>
    <li> Install scanpy >= 1.5.1 for pre-processing </li>
    <li>(Optional) Install <a href='https://github.com/slundberg/shap'>SHAP</a> for interpretation.</li>
</ul>

#### 2. Installation

The functions required for the stage I and II of iMAP could be imported from “imap.imap” and “imap.utils”, respectively.

### iMAP workflow

In this tutorial, we will show the entire iMAP pipeline using the 'cell_lines' dataset (See <font color='red'>main text</font>). The total workfow includes: 
<ol>
    <li>Loading and preprocessing data;</li>
    <li>Running the main iMAP two-stage batch effect removal procedure;</li>
    <li>(Optional) Visulizations; </li>
    <li>(Optional) Interpreting the trained model. </li>    
</ol>


```python
#### IMPORT LIBRARY ####
import scanpy as sc
import imap.imap as imap
import imap.utils as utils
```

#### 1. Loading and preprocessing data

The 'cell_lines' dataset is already stored at <a href=''><font color='blue'>'./data/cell_lines.loom'</font></a>. We use the scanpy API to read the file and store as a 'adata' object. Two kinds of metadata were necessary: **'batch'** and **'celltype'**, while the latter was only used for visualizations and interpretations. 


```python
adata = sc.read_loom('../data/cell_lines.loom',sparse=False)  #Load cell line dataset.
celltype = adata.obs['celltype']  #Save celltype for visualize the results conveniently. (Not necessary for iMAP).
```

Preprocessing: We provide a simple pre-process procedure as a function 'imap.data_preprocess'. One essential request for the successful running of iMAP is that the input data should be in the form of **log-TPM-like (non-negative)**.

Be default, Scanpy API ‘scanpy.pp.highly_variable_genes’ is used in iMAP to select highly variable genes from each batch to assist in discovering biological variations. We also recommend users preprocess the data in their own way, as long as the data meets iMAP's requirements. The numbers of genes are relatively appropriate at the level of about 2000. However, it is acceptable to test the performance of 1000-4000 under different data. 


```python
adata = imap.data_preprocess(adata, 'batch', n_batch=3)  #Preprocess the data.
adata  #Output the basic information of the preprocessed data.
```

#### 2. Batch effect removal by iMAP

**Stage I - Key tunable hyperparameters:**

`adata`: Annotated data matrix(`scanpy.AnnData` class) to input the model. An AnnData object adata stores a data matrix adata.X(log-TPM-like single cell transcriptomics data), annotation of observations adata.obs(batch information must be contined) and variables adata.var as pd.DataFrame. Names of observations(cell names) and variables(gene names) can be accessed via adata.obs_names and adata.var_names, respectively.

`key`: Name of annotation of batch information in adata. Set to 'batch' here.

`n_epochs`: Number of epochs (in Stage I) to train the model. **It has a great effect on the results of iMAP.** The number of epochs should be set according to the number of cells in your dataset. For example, 150 epochs is generally fine for around or greater than 10,000 cells. 100 epochs or fewer for fewer than 5,000 cells(i.e. n_epochs was set to 50 for **541 cells** in DC_rm). The number of training epochs could be slightly increased (i.e. n_epochs was set to 200 for Cell lines dataset), to make the model better exploit datasets with subtle heterogeneities.

`lr`: learning rate. Set to 0.0005 here. 

`lambda_co`, `lambda_rc`: Coefficients to balance the content loss and the reconstruction loss. Set to 3, 1 separately.

**Stage I - Other tunable hyperparameters:**

`latent_dim`: The dimension of content representations vector. Set to 256 here.

`b1`, `b2`: Parameters for Adam optimizer. Set to 0.5, 0.999 separately.

`num_workers`: Number of CPU for pytorch DataLoader to train the model. Set to 0 to load data automately.

`seed`: Random seed used to initialize the pseudo-random number generator. If seed is 'None', then random state will try to read data from /dev/urandom (or the Windows analogue) and so on according to the mechanism of the different random functions used in iMAP. Set to 8 here.

**Stage II - Key tunable hyperparameters:**

`adata`: Same as Stage I.

`cali_data`: Annotated data matrix to create rwMNN pairs.(It's actually the `ec_data` which returned by Stage I.)

`key`: Same as Stage I.

`n_epochs`: Number of epochs (in Stage II) to train the model. **It is also the most important hyparameters for training.** The principle of parameter adjustment is the same as the previous stage(i.e. n_epoch is set to 150 as default, 40 for DC_rm, and 300 for cell lines).

`inc`: Set to `True` for multi-batch datasets for a better mixture.

`orders`: Sequence to align sub-dataset. 'None' for automatically sequencing. Set to 'None' here. For example, `orders=['Mix', 'Jurkat, '293t']` for manually sequecing.

`lr`: Learning rate. Set to 0.0002 here. 

`k1`, `k2`: k1 for k nearest neighbors between two sub-datasets in calculation MNN pairs. k2 for k nearest neighbors within the identical sub-datasets for random walk extending. **It is important because the quality of rwMNN pairs directly affect the final blending effect.** “None” for calculating automatically. All set to 'None' here. For datasets with a particularly small number of cells, it is recommended to adjust them according to the actual situation(i.e. k1, k2 is appropriately increased to 1, 5(which is 1,2 for automatically calculating) in 549 cells of DC_rm datasets for a better mixture).

**Stage II - Other tunable hyperparameters:**

`n_sample`: Number of samples to train the model. Set to 3000 here. 

`seed`: Same as Stage I




**Stage I: `imap.iMAP_fast` will return `EC`(encoder for extracting celltype contents) and `ec_data`(celltype contents of input adata).**

**Stage II: `utils.integrate_data` will return `output_results`(log-TPM-like single cell data aligned by iMAP).**



```python
### Stage I
EC, ec_data = imap.iMAP_fast(adata, key="batch", n_epochs=200) 

### Stage II
output_results = utils.integrate_data(adata, ec_data, inc = False, n_epochs=300)
```

#### 3. Visualizations


```python
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

#### UMAP ####
def data2umap(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    embedding_ = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        n_components = 2,
        learning_rate = 1.0,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1,
        repulsion_strength = 1,
        negative_sample_rate = 5,
        angular_rp_forest = False,
        verbose = False
    ).fit_transform(embedding)
    return embedding_
def umap_plot(data, hue, title, save_path):
    import seaborn as sns
    fig = sns.lmplot(
        x = 'UMAP_1',
        y = 'UMAP_2',
        data = data,
        fit_reg = False,
        legend = True,
        size = 9,
        hue = hue,
        scatter_kws = {'s':4, "alpha":0.6}
    )
    plt.title(title, weight='bold').set_fontsize('20')
    fig.savefig(save_path)
    plt.close()
def gplot(embedding_, batch_info, celltype_info, filename):
    test = pd.DataFrame(embedding_, columns=['UMAP_1', 'UMAP_2'])
    test['Label1'] = batch_info
    test['Label2'] = celltype_info
    title = f' '
    for i in range(1,3):
        hue = f'Label{i}'
        save_path = './pic/'+filename + f'{i}.png'
        umap_plot(test, hue, title, save_path)
```

**Visualizations for the representations from stage I:** The results are saved in './pic/' directory. (256d dimensional reduction results colorder by 'celltype')


```python
embedding_ = data2umap(np.array(ec_data.X), n_pca=30)
gplot(embedding_, np.array(ec_data.obs['batch']), np.array([celltype[item] for item in ec_data.obs_names]), 'cellline_ec_')
```

**Visualizations for the final output results**:The results are saved in './pic/' directory. 


```python
embedding_ = data2umap(output_results, n_pca=30)
gplot(embedding_, np.array(adata.obs['batch']), np.array([celltype[item] for item in adata.obs_names]), 'cellline_G_')
```

#### 4. Interpretations

Here we use <a href='https://github.com/slundberg/shap'>SHAP</a> to explain the decoupling model. The `imap.explaix_importance` function takes `EC`, `adata`, `celltypes` as inputs to explain which genetic features the model learned are important for cell types. The `imap.contrast_explain_importance` function takes adata and platform annotation as inputs and calulate the different genetic feature between the platforms.

The results are saved in './' directory.


```python
celltypes = np.array(adata.obs['celltype'])
platform = np.array(adata.obs['batch'])
```


```python
imap.explain_importance(EC, adata, celltypes, print_loss=True, n_epochs=400)
imap.contrast_explain_importance(adata, platform, print_loss=True, n_epochs=400)
```
