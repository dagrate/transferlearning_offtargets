# -*- coding: utf-8 -*-
__date__ = '13-Feb-22'
__revised__ = '14-Jan-22'
__author__ = 'Jeremy Charlier'
#
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
import transferlearning_utils as tlrn
import datautils
#
#
# ENCODE DATA
def encodeDataPipeline(datatrain, datatrainfcts):
  encodeddata = []
  for idata in range(len(datatrain)):
    datanm = datatrain[idata]
    data = datatrainfcts[idata](datanm)
    encodeddata.append(data)
  # ENDFOR
  return encodeddata
# ENDFUNCTION encodeDataPipeline
#
def datapipeline():
  flpath = '/data/'
  nms = [
    flpath+'listgarten_elevation_cd33.csv',
    flpath+'CIRCLE_seq_10gRNA_wholeDataset.csv',
    flpath+'SITE-Seq_offTarget_wholeDataset.csv',
    flpath+'listgarten_elevation_guideseq.csv',
    flpath+'Listgarten_22gRNA_wholeDataset.csv',
    flpath+'Kleinstiver_5gRNA_wholeDataset.csv',
    flpath+'listgarten_elevation_hmg.csv',
    flpath+'guideseq.csv'
  ]
  nmfcts = [
    datautils.load_elevation_CD33_dataset,
    datautils.load_CIRCLE_data,
    datautils.load_siteseq_data,
    datautils.load_elevation_guideseq_data,
    datautils.load_22sgRNA_data,
    datautils.load_Kleinstiver_data,
    datautils.load_elevation_hmg_dataset,
    datautils.load_guideseq_crispr_data
  ]
  print("!!! DATA ENCODING !!!")
  encodeddata = encodeDataPipeline(nms, nmfcts)
  print("--- end of data encoding ---")
  return encodeddata
# ENDFUNCTION datapipeline
#
#
# ==============================
# ===     getEncodedData     ===
# ==============================
def getEncodedData(
    is_read_pkl_encoded_data = True, 
    path_to_module = '', 
    dataPath = '/data/encoded_data.pkl',
    dataSetPosition = 0):
  """Get the data for training and transfer learning experiments.
  __date__ = 07-Jan-23
  __lastUpdate__ = 09-Jan-23
  """
  if is_read_pkl_encoded_data:
    f = open(path_to_module + '/data/encoded_data.pkl', 'rb')
    encdata = pkl.load(f)
    f.close()
  else:
    encdata = datapipeline()
  #
  data = encdata[dataSetPosition]
  xtrain, xtest, ytrain, ytest = tlrn.dataSplitRF(data)
  xtrainres, ytrainres = xtrain, ytrain
  return Bunch(
    encdata = encdata, data = data,
    xtrain = xtrain, xtest = xtest, xtrainres = xtrainres,
    ytrain = ytrain, ytest = ytest, ytrainres = ytrainres)
#
#
# ====================
# ===     main     ===
# ====================
if __name__ == "__main__":
  if False: databunch = datapipeline()
