# transferlearning_offtargets

Performing off-target predictions using transfer learning.


## Repository Structure

1. _data_: all data required for the experiments
2. _notebooks_: all scientific experiments
3. _src_: python source code used in the scientific experiments (required to execute the scientific experiments)


## Data

For information on the data, refer to the data readme available [here](https://github.com/dagrate/transferlearning_offtargets/blob/main/data/readme.md)



## For similarity analysis

**Measurement of the data similarity using boostrapping**

1. _transferlearning_crispr_data_similarity_bootstrapping_cos.ipynb_: for experiments with the **cosine** distance
2. _transferlearning_crispr_data_similarity_bootstrapping_euc.ipynb_: for experiments with the **euclidean** distance
3. _transferlearning_crispr_data_similarity_bootstrapping_man.ipynb_: for experiments with the **manhattan** distance


## For data encoding 

The encoder class is inherited from [1] "CRISPR-Net: A Recurrent Convolutional Network Quantiﬁes CRISPR Off-Target Activities with Mismatches and Indels", J. Lin et al, https://onlinelibrary.wiley.com/doi/epdf/10.1002/advs.201903562

1. _transferlearning_data_encoding.ipynb_: for data encoding
2. _transferlearning_crispr_datapipeline.ipynb_: data pipeline to preprocess the encoded data for models, notebook format
3. _transferlearning_data_encoding_pipeline.ipynb_: data pipeline in src code to preprocess and save the encoded data in a pkl format


## For hyperparameters search 

This folder contains the notebooks used to perform hyperparameters search. 
Training all the models while doing hypertuning requires extensive computation.
We first performed the hyperparameters search (in priority for the deep learning models), and then we trained and saved the models (see section below).

1. _transferlearning_RF_MLP_models_hyperparameters_search.ipynb_: simple search for MLP and random forest algorithms
2. _transferlearning_DLmodels_hypertuning.ipynb_: hyperparameters search for the deep learning models


## For models training

1. _transferlearning_clfs_train_and_save_models_CD33.ipynb_: train and save models with hypertuned parameters on CD33 data set
2. _transferlearning_clfs_train_and_save_models_circleSeq10gRNA.ipynb_: train and save models with hypertuned parameters on Circle-Seq data set
3. _transferlearning_clfs_train_and_save_models_siteSeq.ipynb_: train and save models with hypertuned parameters on Site_Seq data set


## For models predictions

1. _transferlearning_predict_roc_recall_curves_cd33.ipynb_: transfer learning models predictions with hypertuned parameters on CD33 data set
2. _transferlearning_predict_roc_recall_curves_circle_seq_10grna.ipynb_: transfer learning models predictions with hypertuned parameters on Circle-Seq 10gRNA data set
3. _transferlearning_predict_roc_recall_curves_siteSeq.ipynb_: transfer learning models predictions with hypertuned parameters Site-Seq CD33 data set 

## For pipelines

In the folder src, different modules and functions are available to conduct the experiments in the notebooks.

1. _simanalysis.py_: python file containing functions for the similarity analysis of the data sets
2. _transferlearning_datapipeline.py_: python file containing functions for the data encoding and pre-processsing
3. _transferlearning_modelpipeline.py_: python file containing functions for the models
4. _transferlearning_tensorflow_models.py_: python file containing functions for the tensorflow models
5. _transferlearning_utils.py_: python file containing various functions used across all notebooks at the different stages of the experiments

