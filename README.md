# transferlearning_offtargets

## Repository Structure

1. _data_: all data required for the experiments
2. _notebooks_: all scientific experiments
3. _src_: python source code used in the scientific experiments (required to execute the scientific experiments)


## For similarity analysis

**Measurement of the data similarity using boostrapping**

1. _transferlearning_crispr_data_similarity_bootstrapping_cos.ipynb_: for experiments with the **cosine** distance
2. _transferlearning_crispr_data_similarity_bootstrapping_euc.ipynb_: for experiments with the **euclidean** distance
3. _transferlearning_crispr_data_similarity_bootstrapping_man.ipynb_: for experiments with the **manhattan** distance


## For data encoding 

The encoder class is inherited from [1] "CRISPR-Net: A Recurrent Convolutional Network Quantiﬁes CRISPR Off-Target Activities with Mismatches and Indels", J. Lin et al, https://onlinelibrary.wiley.com/doi/epdf/10.1002/advs.201903562

