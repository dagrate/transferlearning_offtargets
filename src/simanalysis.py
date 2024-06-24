__author__ = 'jeremy charlier'
__date__ = '18-Mar-22'
__revised__ = '26-Nov-22'
#
# append to python sys path
import sys
path_to_module = '' # put the right path if needed
path_to_module += '' # put the right path if needed
sys.path.append(path_to_module)
#
import random; random.seed(42)
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from imblearn.under_sampling import RandomUnderSampler as rus
from transferlearning_datapipeline import datapipeline
p = print
#
# PIPELINE FOR DATA SIMILARITY ANALYSIS
def urnd(x, digits=4): return np.round(x, digits) ;
#
def distpreproc(data): return data.reshape(-1, data.shape[1]*data.shape[2]) ;
#
def getData(is_read_pkl_encoded_data):
  if is_read_pkl_encoded_data:
    f = open(path_to_module+'/data/encoded_data.pkl', 'rb')
    data = pkl.load(f)
    f.close()
  else:
    data = datapipeline()
  return data
# ENDFUNCTION
#
def usofar(udist, cursofar, bestsofar):
  if 'cosine' in str(udist):
    cdtn=cursofar>bestsofar
  else:
    cdtn=cursofar<bestsofar
  # ENDIF
  if cdtn: bestsofar=cursofar ;
  return bestsofar
# ENDFUNCTION
#
def printMinDistance(
    udist,
    bootdata,
    alldata, 
    snms,
    nelts = 500,
    nmc = 100,
    israndomsearch = True,
    is_avg_randsearch = True,
    verbose = 1):
  """Print the minimal distance between data sets for similarity analysis.
  """
  if 'cos' in str(udist): exit_nmc = 1.0
  uint = np.random.randint
  if verbose == 1: ilist=[];
  elif verbose == 2: p('\n!!!'+str(udist)+'!!!\n');
  #for idata in range(len(bootdata)): # idata = NBR OF DATASETS
  mat1 = distpreproc(bootdata.data)
  nmcarr = np.zeros((len(alldata), len(mat1), nmc))
  for jdata in range(len(alldata)): # jdata = NBR OF DATASETS
      cosvl=[]
      mat2 = distpreproc(alldata[jdata].data)
      # random search or exhaustive search
      nloop = nmc if israndomsearch==True else len(mat2);
      if nelts == 0: nelts = len(mat1); # loop through all seq.
      for ind in range(nelts): # loop to individual elts of dataset I
        bestsofar = 0.0 if 'cosine' in str(udist) else 100000.0;
        arr1 = mat1[ind].reshape(1,-1)
        for imc in range(nloop):
          indxmat2 = uint(0, len(mat2), 1)[0] if israndomsearch==True else imc;
          arr2 = mat2[indxmat2].reshape(1,-1)
          ivalue = udist(arr1, arr2)
          nmcarr[jdata, ind, imc] = ivalue
          bestsofar = usofar( # search for minimal distance
            udist, # distance function, for inst. cos_sim
            ivalue, # current distance value
            bestsofar) # best distance so far
        # ENDFOR imc
          if not is_avg_randsearch: # exit for loop when cos_dist == 1.0
            if bestsofar == exit_nmc:
              break
        cosvl.append(bestsofar) # store smallest distance
      # ENDFOR ind
      if is_avg_randsearch:
        if verbose==1: ilist.append(urnd(np.mean(cosvl)));
        elif verbose==2:
          p(bootdata.name, snms[jdata])
          p("{:.3f};{:.4f}".format(
            urnd(np.mean(cosvl)), urnd(np.std(cosvl)))
          )
      else:
        if verbose==1: ilist.append(urnd(np.max(cosvl)));
      # ENDIF
    # ENDIF idata!=jdata
  # ENDFOR jdata
  p(ilist)
  # ENDFOR idata
  # return Bunch(minvl=cosvl, epochsvl=nmcarr) # to be re-worked
  return ilist
# END OF FUNCTION printMinDistance
#
#
#
# ===================================
# ===     getBootStrappedData     ===
# ===================================
def getBootStrappedData(
    snms, 
    encdata,
    bootdata_len = 1000,
    isPrint = False):
  """
  Create smaller data sets for transfer learning experiments.

  Revision history
  ----------------
  26-Nov-22: increase to 1k samples + change class imbalance
  """
  bootdata = []
  for cnt in range(len(snms)):
    data_sample_size = len(encdata[cnt].data)
    xtrain, xtest, ytrain, ytest = train_test_split(
      encdata[cnt].data,
      encdata[cnt].target,
      test_size = bootdata_len/data_sample_size,
      stratify = encdata[cnt].target)
    nwdata, nwtarget = xtest[:bootdata_len], ytest[:bootdata_len]
    #
    # enforce the class imbalance except for cd33
    if cnt != 0:
      if cnt > 2: # no risk of data leakage
        indx = (ytrain == 1)
        xboot = xtrain
        yboot = ytrain
      else: # to avoid data leakage
        indx = (ytest == 1)
        xboot = xtest
        yboot = ytest
      ipos = indx.sum()
      posamp = xboot[indx]
      postarget = yboot[indx]
      nwdata[-ipos:] = posamp
      nwtarget[-ipos:] = postarget
      #
      # shuffle data
      perm = np.random.permutation(len(nwdata))
      nwdata = nwdata[perm]
      nwtarget = nwtarget[perm]
    # ENDIF
    if isPrint:
      p('\nbootstrapped data:', snms[cnt])
      p('new boostrapped data shape:', nwdata.shape)
      p('positive samples:', len(nwtarget)-nwtarget.sum())
      p('negative samples:', nwtarget.sum())
    curnm = 'bootstrapped_'+snms[cnt]
    bootdata.append(Bunch(
      name = curnm, data = nwdata, target = nwtarget))
  return bootdata
#
#
def similarityPipeline(
    dist_fun,
    bootdata_len = 250,
    is_read_pkl_encoded_data = True,
    snms = [
      'listgarten_elevation_cd33.csv',
      'CIRCLE_seq_10gRNA_wholeDataset.csv',
      'SITE-Seq_offTarget_wholeDataset.csv',
      'listgarten_elevation_guideseq.csv',
      'Listgarten_22gRNA_wholeDataset.csv',
      'Kleinstiver_5gRNA_wholeDataset.csv',
      'listgarten_elevation_hmg.csv',
      'guideseq.csv'],
    is_randomsearch = True,
    nelts = 0,
    verbose = 1,
    is_save_csv = True,
    nmcs = [10, 25, 50, 100, 500, 1000, 2000]):
  encdata = getData(is_read_pkl_encoded_data)
  bootdata = getBootStrappedData(
    snms, 
    encdata,
    bootdata_len
  )
  dist_val = []
  p('\n--- DISTANCES COMPUTATION ----\n')
  for nmc in nmcs:
      p('\n--- NMC ' + str(nmc) + '---\n')
      for cnt in range(len(snms)):
        
        dist_val.append(
          printMinDistance(
            dist_fun,
            bootdata[cnt],
            encdata,
            snms, 
            nelts=nelts, 
            nmc=nmc,
            israndomsearch = is_randomsearch, 
            verbose = verbose
          )
        )
        if is_save_csv:
          np.savetxt(
            'dist_'+str(nmc)+'.csv', 
            dist_val,delimiter =", ", 
            fmt ='% s'
          )
        # ENDIF
      # ENDIF
    # ENDFOR cnt
  # ENDFOR nmc
# ENDFUNCTION
#
#
# =============================
# ===     buildBootData     ===
# =============================
def buildBootData(encdata, bootdata_len = 1000, nmc = 15):
  """Build the bootstrapped data sets for transfer learning experiments.
  __date__ = 11-Jan-22
  __lastUpdate__ = 11-Jan-22
  """
  cd33_boots = []
  circle_boots = []
  site_boots = []
  elev_gseq_boots = []
  grna22_boots = []
  grna5_boots = []
  elev_hmg_boots = []
  #
  for imnc in range(nmc):
    data_bootstrapped = getBootStrappedData(
      [
        'listgarten_elevation_cd33.csv',
        'CIRCLE_seq_10gRNA_wholeDataset.csv',
        'SITE-Seq_offTarget_wholeDataset.csv',
        'listgarten_elevation_guideseq.csv',
        'Listgarten_22gRNA_wholeDataset.csv',
        'Kleinstiver_5gRNA_wholeDataset.csv',
        'listgarten_elevation_hmg.csv'
      ],
      encdata, bootdata_len = bootdata_len
    )
    cd33_boots.append( data_bootstrapped[0] )
    circle_boots.append( data_bootstrapped[1] )
    site_boots.append( data_bootstrapped[2] )
    elev_gseq_boots.append( data_bootstrapped[3] )
    grna22_boots.append( data_bootstrapped[4] )
    grna5_boots.append( data_bootstrapped[5] )
    elev_hmg_boots.append( data_bootstrapped[6] )
  bootdata = Bunch(
    cd33_boots = cd33_boots,
    circle_boots = circle_boots,
    site_boots = site_boots,
    elev_gseq_boots = elev_gseq_boots,
    grna22_boots = grna22_boots,
    grna5_boots = grna5_boots,
    elev_hmg_boots = elev_hmg_boots
  )
  return bootdata
#