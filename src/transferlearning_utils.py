import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from transferlearning_modelpipeline import modelpipeline
#
from sklearn.metrics import average_precision_score as avgPrecScore
from sklearn.metrics import precision_recall_curve as PRC
P = print
#
def urnd(val): return np.round(val, 3);
#
def reshapeArr(arr, est='rf'):
  shp = np.prod(arr.shape[1:])
  if est == 'rf':
    arr = arr.astype('float32')
    arr = arr.reshape(-1,shp)
  # ENDIF
  return arr
# ENDFUNCTION
#
def printClassImbalance(y):
  nmin = min(
    (y==1).sum(),
    (y==0).sum()
  )
  nmaj = max(
    (y==1).sum(),
    (y==0).sum()
  )
  print(">> Nbr of samples in minority class: %s" % nmin)
  print(">> Nbr of samples in majority class: %s" % nmaj)
  print(">> Class imbalance: %s " % np.round(nmin/len(y), 3))
# ENDFUNCTION
#
def encodeDataPipeline(datatrain, datatrainfcts):
  encodeddata = []
  for idata in range(len(datatrain)):
    datanm = datatrain[idata]
    data = datatrainfcts[idata](datanm)
    encodeddata.append(data)
  # end for
  return encodeddata
# ENDFUNCTION
#
def dataSplitRF(data, testsize=.3, verbose=False):
  if verbose:
    printClassImbalance(data.target)
  xtrain,xtest,ytrain,ytest = train_test_split(
    data.data,
    pd.Series(data.target),
    test_size = testsize,
    shuffle = True,
    random_state = 0
  )
  xtrain = reshapeArr(xtrain)
  xtest = reshapeArr(xtest)
  return xtrain, xtest, ytrain, ytest
#
#
#
# ==============================
# ===      plotRocCurve      ===
# ==============================
def plotRocCurve(
		estimators, ucolor,
    icol = 1, disp = False):
  """Plot ROC Curve for the test data set after estimator training.
  __date__ = 01-Sep-22
  __lastUpdate__ = 07-Jan-23
  """
  plt.figure(figsize=(8, 6))
  plt.plot([0, 1], [0, 1], 'k--')
  for iclf in range(len(ucolor)):
    clf = estimators[iclf]
    cdtIF = ('MLP' in ucolor[iclf][0])
    cdtIF += (ucolor[iclf][0] == 'RF')
    cdtIF += (ucolor[iclf][0] == 'LR')
    #
    if cdtIF:
      fprs, tprs = getFprTpr( estimators[iclf], disp = disp )
      tmpAUC = urnd(auc(fprs, tprs))
      plt.plot(
        fprs, tprs, ucolor[iclf][1],
        label = ucolor[iclf][0] + ' (AUC: %s \u00B1 0.001)' % (tmpAUC))
    else:
      plt.plot(
        clf.fpr, clf.tpr, ucolor[iclf][1],
        label = ucolor[iclf][0] + ' (AUC: %s \u00B1 0.001)' % (clf.roc_auc))
  # >>> plot legend <<<
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.legend(loc = 'best')
  plt.savefig('plotROCCurveTestSet.pdf')
#
#
# ===========================
# ===     plotPRCurve     ===
# ===========================
def plotPRCurve(estimators, ucolor):
  """Plot Precision Recall Curve for the test data set after estimator training.
  __date__ = 20-Dec-22
  __lastUpdate__ = 21-Jan-23
  """
  plt.figure(figsize=(8, 6))
  for iclf in range(len(ucolor)):
    # >>> parameters for precision-recall curve <<<
    clf = estimators[iclf]
    strEst = str(clf.estimator)
    cdtIF = ('MLP' in strEst)
    cdtIF += ('RandomForest' in strEst)
    cdtIF += ('LogisticRegression' in strEst)
    #
    if cdtIF:
      ytest = clf.y_test
      yscore = clf.yscore[:, 1]
    else:
      ytest = clf.data_dict['y_test']
      yscore = clf.ypred
    # >>> plot precision-recall curve <<<
    precision, recall, thresholds = PRC(ytest, yscore)
    tmpAP = urnd(clf.avgPrecScore)
    plt.plot(
      recall, precision, ucolor[iclf][1],
      label = ucolor[iclf][0] + ' (AP: %s \u00B1 0.001)' % (tmpAP))
  # >>> plot legend <<<
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc = 'best')
  plt.savefig('plotPRCurveTestSet.pdf')
#
#
# ============================================
# ===       plotRocCurve_tl_boostrap       ===
# ============================================
def plotRocCurve_tl_boostrap(
		fprs_all, tprs_all,
    roc_auc_means, roc_auc_stds,
    clf_labels, lgndText, lgndLines,
    is_savefig = False, savefig_nm = 'boot_plot.pdf'):
  """Plot ROC Curve for the TL experiments."""
  plt.figure( figsize = (8, 6) )
  plt.plot([0, 1], [0, 1], 'k--')
  for iclf in clf_labels:
    plt.plot(
      fprs_all[iclf], tprs_all[iclf], lgndLines[iclf],
      label = lgndText[iclf]+' (AUC: %s \u00B1 %s)' % (
        urnd(roc_auc_means[iclf]),
        urnd(roc_auc_stds[iclf])
      )
    )
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.legend(loc='best')
  if is_savefig:
    plt.savefig(savefig_nm)
  else:
    plt.show()
#
#
#
# =========================================
# ===       plotPrecisionRecallTL       ===
# =========================================
def plotPrecisionRecallTL(
  precisions, recalls,
  meansSorted, stdsSorted, clflbls,
  lgndText, lgndLines,
  is_savefig = True, savefig_nm = 'boot_plot_ap.pdf'):
  """Plot AP curve for the TL experiments."""
  plt.figure( figsize = (8, 6) )
  for iclf in clflbls:
    plt.plot(
      recalls[iclf], precisions[iclf], lgndLines[iclf],
      label = lgndText[iclf]+' (AP: %s \u00B1 %s)' % (
        urnd(meansSorted[iclf]),
        urnd(stdsSorted[iclf])
      )
    )
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='best')
  if is_savefig:
    plt.savefig(savefig_nm)
  else:
    plt.show()
#
#
# =============================
# ===       getFprTpr       ===
# =============================
def getFprTpr(estimator, disp = False):
  """Get false positive rates and true positive rates."""
  preds = estimator.yscore
  if len(preds.shape) == 2:
    ypreds = preds[:,1]
  else:
    ypreds = preds
  fprs, tprs, _ = roc_curve(estimator.y_test, ypreds)
  if disp:
    print('false positive rates:\n', fprs)
    print('true positive rates:\n', tprs)
  return fprs, tprs
#
#
# ============================================
# ===     computeAveragePrecisionScore     ===
# ============================================
def computeAveragePrecisionScore(estimators):
  """Compute the average precision scores.
  ___date__ = 15-Dec-22
  __lastUpdate__ = 07-Jan-23
  """
  for iclf in range(len(estimators)):
    clf = estimators[iclf]
    cdtIF = ('MLP' in str(clf.estimator))
    cdtIF += ('RandomForest' in str(clf.estimator))
    cdtIF += ('LogisticRegression' in str(clf.estimator))
    #
    if cdtIF:
      clf.avgPrecScore = avgPrecScore(clf.y_test, clf.yscore[:, 1])
    else:
      clf.avgPrecScore = avgPrecScore(clf.data_dict['y_test'], clf.ypred)
  return estimators
#
#
# =============================
# ===    assignTLdataset    ===
# =============================
def assignTLdataset(
    cd33_boots, circle_boots, site_boots,
    elev_gseq_boots, grna22_boots, grna5_boots,
    elev_hmg_boots, whichTLdataset):
  msg_print = '=== TRANSFER LEARNING ON '
  if whichTLdataset == 0:
    tl_boots = cd33_boots
    msg_print += 'CD33'
  elif whichTLdataset == 1:
    tl_boots = circle_boots
    msg_print += 'CIRCLE SEQ'
  elif whichTLdataset == 2:
    tl_boots = site_boots
    msg_print += 'SITE SEQ'
  elif whichTLdataset == 3:
    tl_boots = elev_gseq_boots
    msg_print += 'ELEVATION GUIDE SEQ'
  elif whichTLdataset == 4:
    tl_boots = grna22_boots
    msg_print += '22GRNA'
  elif whichTLdataset == 5:
    tl_boots = grna5_boots
    msg_print += '5GRNA'
  elif whichTLdataset == 6:
    tl_boots = elev_hmg_boots
    msg_print += 'ELEVATION HMG'
  msg_print += ' BOOTSTRAPPED DATA ==='
  print(msg_print)
  return tl_boots
#
#
# =====================================
# ===    computeMetricsForTLplot    ===
# =====================================
def computeMetricsForTLplot(
    cd33_boots, circle_boots, site_boots,
    elev_gseq_boots, grna22_boots, grna5_boots,
    elev_hmg_boots, whichTLdataset,
    mlp_trnd, mlp_trnd_2layers,
    rf_trnd, lr_trnd,
    ffn3_trnd, ffn5_trnd, ffn10_trnd,
    cnn3_trnd, cnn5_trnd, cnn10_trnd,
    lstm3_trnd, gru3_trnd,
    is_verbose_training = False):
  """
  Compute the predictive metrics on the boostrapped data sets.

  Revision History
  ----------------
  26-Nov-22: initial version
  """
  tl_boots = assignTLdataset(
    cd33_boots, circle_boots, site_boots,
    elev_gseq_boots, grna22_boots, grna5_boots,
    elev_hmg_boots, whichTLdataset
  )
  # >>> for SKLEARN models <<<
  fprs_mlp_nmc = []
  tprs_mlp_nmc = []
  fprs_mlp_2layers_nmc = []
  tprs_mlp_2layers_nmc = []
  fprs_rf_nmc = []
  tprs_rf_nmc = []
  fprs_lr_nmc = []
  tprs_lr_nmc = []
  roc_auc_mlp_nmc = []
  roc_auc_mlp_2layers_nmc = []
  roc_auc_rf_nmc = []
  roc_auc_lr_nmc = []
  #
  # >>> for FNNS models <<<
  fprs_ffn3_nmc = []
  tprs_ffn3_nmc = []
  roc_auc_ffn3_nmc = []
  roc_auc_ffn5_nmc = []
  tprs_ffn5_nmc = []
  fprs_ffn5_nmc = []
  roc_auc_ffn10_nmc = []
  tprs_ffn10_nmc = []
  fprs_ffn10_nmc = []
  #
  # >>> for CNNS models <<<
  fprs_cnn3_nmc = []
  tprs_cnn3_nmc = []
  roc_auc_cnn3_nmc = []
  roc_auc_cnn5_nmc = []
  tprs_cnn5_nmc = []
  fprs_cnn5_nmc = []
  roc_auc_cnn10_nmc = []
  tprs_cnn10_nmc = []
  fprs_cnn10_nmc = []
  #
  # >>> for RNNS models <<<
  fprs_lstm3_nmc = []
  tprs_lstm3_nmc = []
  roc_auc_lstm3_nmc = []
  fprs_gru3_nmc = []
  tprs_gru3_nmc = []
  roc_auc_gru3_nmc = []
  #
  for inmc in range(len(cd33_boots)):
    # === DATA PRE-PROCESSING ===
    # >>> SKLEARN MODELS <<<
    xtldata = reshapeArr( tl_boots[inmc].data )
    ytldata = pd.Series( tl_boots[inmc].target )
    #
    # >>> FFNS MODELS <<<
    xtldata_ffn = reshapeArr( tl_boots[inmc].data )
    ytldata_ffn = pd.Series( tl_boots[inmc].target )
    # >>> CNNS MODELS <<<
    xtldata_cnn = tl_boots[inmc].data.copy().reshape(-1, 24, 7, 1)
    ytldata_cnn = pd.Series( tl_boots[inmc].target.copy() )
    # >>> RNNS MODELS <<<
    xtldata_rnn = tl_boots[inmc].data.copy().reshape(-1, 24, 7, 1)
    ytldata_rnn = pd.Series( tl_boots[inmc].target.copy() )
    #
    # === MODEL PIPELINE ===
    # === MLP 1 LAYER ===
    mlp_tl = modelpipeline(
      mlp_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    cur_fpr, cur_tpr = getFprTpr(mlp_tl)
    roc_auc_mlp_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_mlp_nmc[0]):
        fprs_mlp_nmc.append(cur_fpr)
        tprs_mlp_nmc.append(cur_tpr)
    else:
      fprs_mlp_nmc.append(cur_fpr)
      tprs_mlp_nmc.append(cur_tpr)
    #
    # === MLP 2 LAYER ===
    mlp_tl_2layers = modelpipeline(
      mlp_trnd_2layers.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    cur_fpr, cur_tpr = getFprTpr( mlp_tl_2layers )
    roc_auc_mlp_2layers_nmc.append( auc( cur_fpr, cur_tpr ) )
    if inmc > 0:
      if len(cur_fpr) == len(fprs_mlp_nmc[0]):
        fprs_mlp_2layers_nmc.append(cur_fpr)
        tprs_mlp_2layers_nmc.append(cur_tpr)
    else:
      fprs_mlp_2layers_nmc.append(cur_fpr)
      tprs_mlp_2layers_nmc.append(cur_tpr)
    #
    # === RANDOM FOREST ===
    rf_tl = modelpipeline(
      rf_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    cur_fpr, cur_tpr = getFprTpr(rf_tl)
    roc_auc_rf_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_rf_nmc[0]):
        fprs_rf_nmc.append(cur_fpr)
        tprs_rf_nmc.append(cur_tpr)
    else:
      fprs_rf_nmc.append(cur_fpr)
      tprs_rf_nmc.append(cur_tpr)
    #
    # === LOGISTIC REGRESSION ===
    lr_tl = modelpipeline(
      lr_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    cur_fpr, cur_tpr = getFprTpr(lr_tl)
    roc_auc_lr_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_lr_nmc[0]):
        fprs_lr_nmc.append(cur_fpr)
        tprs_lr_nmc.append(cur_tpr)
    else:
      fprs_lr_nmc.append(cur_fpr)
      tprs_lr_nmc.append(cur_tpr)
    #
    # === FFN 3 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_ffn,
      ffn3_trnd.trained_model.predict(xtldata_ffn)
    )
    roc_auc_ffn3_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_ffn3_nmc[0]):
        fprs_ffn3_nmc.append(cur_fpr)
        tprs_ffn3_nmc.append(cur_tpr)
    else:
      fprs_ffn3_nmc.append(cur_fpr)
      tprs_ffn3_nmc.append(cur_tpr)
    #
    # === FFN 5 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_ffn,
      ffn5_trnd.trained_model.predict(xtldata_ffn)
    )
    roc_auc_ffn5_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_ffn5_nmc[0]):
        fprs_ffn5_nmc.append(cur_fpr)
        tprs_ffn5_nmc.append(cur_tpr)
    else:
      fprs_ffn5_nmc.append(cur_fpr)
      tprs_ffn5_nmc.append(cur_tpr)
    #
    # === FFN 10 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_ffn,
      ffn10_trnd.trained_model.predict(xtldata_ffn)
    )
    roc_auc_ffn10_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_ffn10_nmc[0]):
        fprs_ffn10_nmc.append(cur_fpr)
        tprs_ffn10_nmc.append(cur_tpr)
    else:
      fprs_ffn10_nmc.append(cur_fpr)
      tprs_ffn10_nmc.append(cur_tpr)
    #
    # === CNN 3 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_cnn,
      cnn3_trnd.trained_model.predict(xtldata_cnn)
    )
    roc_auc_cnn3_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_cnn3_nmc[0]):
        fprs_cnn3_nmc.append(cur_fpr)
        tprs_cnn3_nmc.append(cur_tpr)
    else:
      fprs_cnn3_nmc.append(cur_fpr)
      tprs_cnn3_nmc.append(cur_tpr)
    #
    # === CNN 5 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_cnn,
      cnn5_trnd.trained_model.predict(xtldata_cnn)
    )
    roc_auc_cnn5_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_cnn5_nmc[0]):
        fprs_cnn5_nmc.append(cur_fpr)
        tprs_cnn5_nmc.append(cur_tpr)
    else:
      fprs_cnn5_nmc.append(cur_fpr)
      tprs_cnn5_nmc.append(cur_tpr)
    #
    # === CNN 10 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_cnn,
      cnn10_trnd.trained_model.predict(xtldata_cnn)
    )
    roc_auc_cnn10_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_cnn10_nmc[0]):
        fprs_cnn10_nmc.append(cur_fpr)
        tprs_cnn10_nmc.append(cur_tpr)
    else:
      fprs_cnn10_nmc.append(cur_fpr)
      tprs_cnn10_nmc.append(cur_tpr)
    #
    # === LSTM 3 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_rnn, lstm3_trnd.trained_model.predict(xtldata_rnn)
    )
    roc_auc_lstm3_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_lstm3_nmc[0]):
        fprs_lstm3_nmc.append(cur_fpr)
        tprs_lstm3_nmc.append(cur_tpr)
    else:
      fprs_lstm3_nmc.append(cur_fpr)
      tprs_lstm3_nmc.append(cur_tpr)
    #
    # === GRU 3 LAYERS ===
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ytldata_rnn, gru3_trnd.trained_model.predict(xtldata_rnn)
    )
    roc_auc_gru3_nmc.append(auc(cur_fpr, cur_tpr))
    if inmc > 0:
      if len(cur_fpr) == len(fprs_gru3_nmc[0]):
        fprs_gru3_nmc.append(cur_fpr)
        tprs_gru3_nmc.append(cur_tpr)
    else:
      fprs_gru3_nmc.append(cur_fpr)
      tprs_gru3_nmc.append(cur_tpr)
  #
  # === AGGREGATE NMC RESULTS ===
  roc_auc_means = {}
  roc_auc_stds = {}
  tprs_vstack = {}
  fprs_vstack = {}
  #
  clf_labels = [
    'MLP', 'MLP_2layers',
    'RF', 'LR',
    'FFN3', 'FFN5', 'FFN10',
    'CNN3', 'CNN5', 'CNN10',
    'LSTM3', 'GRU3'
  ]
  clf_rocaucs = [
    roc_auc_mlp_nmc, roc_auc_mlp_2layers_nmc,
    roc_auc_rf_nmc, roc_auc_lr_nmc,
    roc_auc_ffn3_nmc, roc_auc_ffn5_nmc, roc_auc_ffn10_nmc,
    roc_auc_cnn3_nmc, roc_auc_cnn5_nmc, roc_auc_cnn10_nmc,
    roc_auc_lstm3_nmc, roc_auc_gru3_nmc
  ]
  clf_tprs = [
    tprs_mlp_nmc, tprs_mlp_2layers_nmc,
    tprs_rf_nmc, tprs_lr_nmc,
    tprs_ffn3_nmc, tprs_ffn5_nmc, tprs_ffn10_nmc,
    tprs_cnn3_nmc, tprs_cnn5_nmc, tprs_cnn10_nmc,
    tprs_lstm3_nmc, tprs_gru3_nmc
  ]
  clf_fprs = [
    fprs_mlp_nmc, fprs_mlp_2layers_nmc,
    fprs_rf_nmc, fprs_lr_nmc,
    fprs_ffn3_nmc, fprs_ffn5_nmc, fprs_ffn10_nmc,
    fprs_cnn3_nmc, fprs_cnn5_nmc, fprs_cnn10_nmc,
    fprs_lstm3_nmc, fprs_gru3_nmc
  ]
  for iclf in range(len(clf_labels)):
    # >>> POST-PROCESSING FOR VSTACK <<<
    clf_rs = [clf_tprs[iclf], clf_fprs[iclf]]
    for jclf in clf_rs:
      max_index = np.inf
      for irow in range(len(jclf)):
        if max_index > len(jclf[irow]):
          max_index = len(jclf[irow])
      for irow in range(len(jclf)):
        jclf[irow] = jclf[irow][:max_index]  
    # >>> METRICS FOR ROC CURVE <<<
    i_roc_auc_nmc = np.asarray(clf_rocaucs[iclf])
    roc_auc_means[clf_labels[iclf]] = i_roc_auc_nmc.mean()
    roc_auc_stds[clf_labels[iclf]] = i_roc_auc_nmc.std()
    tprs_vstack[clf_labels[iclf]] = np.vstack(clf_tprs[iclf]).mean(axis = 0)
    fprs_vstack[clf_labels[iclf]] = np.vstack(clf_fprs[iclf]).mean(axis = 0)
  #
  return fprs_vstack, tprs_vstack, roc_auc_means, roc_auc_stds
#
#
# =========================================
# ===     computeMetricsForTLplotV2     ===
# =========================================
def computeMetricsForTLplotV2(
    cd33_boots, circle_boots, site_boots,
    elev_gseq_boots, grna22_boots, grna5_boots,
    elev_hmg_boots, whichTLdataset,
    mlp_trnd, mlp_trnd_2layers,
    rf_trnd, lr_trnd,
    ffn3_trnd, ffn5_trnd, ffn10_trnd,
    cnn3_trnd, cnn5_trnd, cnn10_trnd,
    lstm3_trnd, gru3_trnd,
    is_verbose_training = False):
  """
  Compute the predictive metrics on the boostrapped data sets.

  Revision History
  ----------------
  26-Nov-22: initial version
  14-Dec-22: update for average precision recall
  """
  def _storeMetricsTL(
      clf, inmc,
      clfMetrics):
    """Compute transfer learning metrics for sklearn classifiers."""
    # roc auc score
    cur_fpr, cur_tpr = getFprTpr(clf)
    clfMetrics['rocauc'].append(auc(cur_fpr, cur_tpr))
    #
    # average precision recall score
    curPrecision, curRecall, curThresholds = PRC(clf.y_test, clf.yscore[:, 1])
    clfMetrics['avgprecscores'].append(avgPrecScore(clf.y_test, clf.yscore[:, 1]))
    #
    if inmc > 0:
      if len(cur_fpr) == len(clfMetrics['fprs'][0]):
        clfMetrics['fprs'].append(cur_fpr)
        clfMetrics['tprs'].append(cur_tpr)
        clfMetrics['precisions'].append(curPrecision)
        clfMetrics['recalls'].append(curRecall)
    else:
      clfMetrics['fprs'].append(cur_fpr)
      clfMetrics['tprs'].append(cur_tpr)
      clfMetrics['precisions'].append(curPrecision)
      clfMetrics['recalls'].append(curRecall)
    return clfMetrics
  #
  def _storeMetricsTLTF(
      xdata = None, ydata = None,
      clf = None, clfMetrics = None, inmc = None):
    """Compute transfer learning metrics for tensorflow classifiers."""
    # roc auc score
    cur_fpr, cur_tpr, cur_thresholds = roc_curve(
      ydata, clf.trained_model.predict(xdata)
    )
    clfMetrics['rocauc'].append(auc(cur_fpr, cur_tpr))
    #
    # average precision recall score
    curPrecision, curRecall, curThresholds = PRC(
      ydata, clf.trained_model.predict(xdata))
    clfMetrics['avgprecscores'].append(avgPrecScore(
      ydata, clf.trained_model.predict(xdata))
    )
    #
    if inmc > 0:
      if len(cur_fpr) == len(clfMetrics['fprs'][0]):
        clfMetrics['fprs'].append(cur_fpr)
        clfMetrics['tprs'].append(cur_tpr)
        clfMetrics['precisions'].append(curPrecision)
        clfMetrics['recalls'].append(curRecall)
    else:
      clfMetrics['fprs'].append(cur_fpr)
      clfMetrics['tprs'].append(cur_tpr)
      clfMetrics['precisions'].append(curPrecision)
      clfMetrics['recalls'].append(curRecall)
    return clfMetrics
  #
  tl_boots = assignTLdataset(
    cd33_boots, circle_boots, site_boots,
    elev_gseq_boots, grna22_boots, grna5_boots,
    elev_hmg_boots, whichTLdataset
  )
  #
  # === update as of dec 16: avoid the use of multiple lists ===
  mlpMetrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  mlp2layersMetrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  rfMetrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  lrMetrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  fnn3Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  fnn5Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  fnn10Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  cnn3Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  cnn5Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  cnn10Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  lstm3Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  gru3Metrics = {
    'fprs': [], 'tprs': [], 'rocauc': [],
    'precisions': [], 'recalls': [], 'avgprecscores': []
  }
  for inmc in range(len(cd33_boots)):
    # === DATA PRE-PROCESSING ===
    # >>> SKLEARN MODELS <<<
    xtldata = reshapeArr( tl_boots[inmc].data )
    ytldata = pd.Series( tl_boots[inmc].target )
    #
    # >>> FFNS MODELS <<<
    xtldata_ffn = reshapeArr( tl_boots[inmc].data )
    ytldata_ffn = pd.Series( tl_boots[inmc].target )
    # >>> CNNS MODELS <<<
    xtldata_cnn = tl_boots[inmc].data.copy().reshape(-1, 24, 7, 1)
    ytldata_cnn = pd.Series( tl_boots[inmc].target.copy() )
    # >>> RNNS MODELS <<<
    xtldata_rnn = tl_boots[inmc].data.copy().reshape(-1, 24, 7, 1)
    ytldata_rnn = pd.Series( tl_boots[inmc].target.copy() )
    #
    # === MODEL PIPELINE ===
    # === MLP 1 LAYER ===
    clfTL = modelpipeline(
      mlp_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    mlpMetrics = _storeMetricsTL(clfTL, inmc, mlpMetrics)
    #
    # === MLP 2 LAYER ===
    clfTL = modelpipeline(
      mlp_trnd_2layers.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    mlp2layersMetrics = _storeMetricsTL(clfTL, inmc, mlp2layersMetrics)
    #
    # === RANDOM FOREST ===
    clfTL = modelpipeline(
      rf_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    rfMetrics = _storeMetricsTL(clfTL, inmc, rfMetrics)
    #
    # === LOGISTIC REGRESSION ===
    clfTL = modelpipeline(
      lr_trnd.estimator,
      xtldata, ytldata,
      xtldata, ytldata,
      is_verbose_training
    ).modelPredict()
    lrMetrics = _storeMetricsTL(clfTL, inmc, lrMetrics)
    #
    # === FFN 3 LAYERS ===
    fnn3Metrics = _storeMetricsTLTF(
      xdata = xtldata_ffn,
      ydata = ytldata_ffn,
      clf = ffn3_trnd,
      clfMetrics = fnn3Metrics,
      inmc = inmc
    )
    #
    # === FFN 5 LAYERS ===
    fnn5Metrics = _storeMetricsTLTF(
      xdata = xtldata_ffn,
      ydata = ytldata_ffn,
      clf = ffn5_trnd,
      clfMetrics = fnn5Metrics,
      inmc = inmc
    )
    #
    # === FFN 10 LAYERS ===
    fnn10Metrics = _storeMetricsTLTF(
      xdata = xtldata_ffn,
      ydata = ytldata_ffn,
      clf = ffn10_trnd,
      clfMetrics = fnn10Metrics,
      inmc = inmc
    )
    #
    # === CNN 3 LAYERS ===
    cnn3Metrics = _storeMetricsTLTF(
      xdata = xtldata_cnn,
      ydata = ytldata_cnn,
      clf = cnn3_trnd,
      clfMetrics = cnn3Metrics,
      inmc = inmc
    )
    #
    # === CNN 5 LAYERS ===
    cnn5Metrics = _storeMetricsTLTF(
      xdata = xtldata_cnn,
      ydata = ytldata_cnn,
      clf = cnn5_trnd,
      clfMetrics = cnn5Metrics,
      inmc = inmc
    )
    #
    # === CNN 10 LAYERS ===
    cnn10Metrics = _storeMetricsTLTF(
      xdata = xtldata_cnn,
      ydata = ytldata_cnn,
      clf = cnn10_trnd,
      clfMetrics = cnn10Metrics,
      inmc = inmc
    )
    #
    # === LSTM 3 LAYERS ===
    lstm3Metrics = _storeMetricsTLTF(
      xdata = xtldata_rnn,
      ydata = ytldata_rnn,
      clf = lstm3_trnd,
      clfMetrics = lstm3Metrics,
      inmc = inmc
    )
    #
    # === GRU 3 LAYERS ===
    gru3Metrics = _storeMetricsTLTF(
      xdata = xtldata_rnn,
      ydata = ytldata_rnn,
      clf = gru3_trnd,
      clfMetrics = gru3Metrics,
      inmc = inmc
    )
  #
  # === AGGREGATE NMC RESULTS ===
  roc_auc_means = {}
  roc_auc_stds = {}
  tprs_vstack = {}
  fprs_vstack = {}
  avgprecscores_means = {}
  avgprecscores_stds = {}
  precisions_vstack = {}
  recalls_vstack = {}
  #
  clf_labels = [
    'MLP', 'MLP_2layers', 'RF', 'LR',
    'FFN3', 'FFN5', 'FFN10',
    'CNN3', 'CNN5', 'CNN10',
    'LSTM3', 'GRU3'
  ]
  clf_rocaucs = [
    mlpMetrics['rocauc'], mlp2layersMetrics['rocauc'],
    rfMetrics['rocauc'], lrMetrics['rocauc'],
    fnn3Metrics['rocauc'], fnn5Metrics['rocauc'], fnn10Metrics['rocauc'],
    cnn3Metrics['rocauc'], cnn5Metrics['rocauc'], cnn10Metrics['rocauc'],
    lstm3Metrics['rocauc'], gru3Metrics['rocauc']
  ]
  clf_tprs = [
    mlpMetrics['tprs'], mlp2layersMetrics['tprs'],
    rfMetrics['tprs'], lrMetrics['tprs'],
    fnn3Metrics['tprs'], fnn5Metrics['tprs'], fnn10Metrics['tprs'],
    cnn3Metrics['tprs'], cnn5Metrics['tprs'], cnn10Metrics['tprs'],
    lstm3Metrics['tprs'], gru3Metrics['tprs']
  ]
  clf_fprs = [
    mlpMetrics['fprs'], mlp2layersMetrics['fprs'],
    rfMetrics['fprs'], lrMetrics['fprs'],
    fnn3Metrics['fprs'], fnn5Metrics['fprs'], fnn10Metrics['fprs'],
    cnn3Metrics['fprs'], cnn5Metrics['fprs'], cnn10Metrics['fprs'],
    lstm3Metrics['fprs'], gru3Metrics['fprs']
  ]
  clf_precisions = [
    mlpMetrics['precisions'], mlp2layersMetrics['precisions'],
    rfMetrics['precisions'], lrMetrics['precisions'],
    fnn3Metrics['precisions'], fnn5Metrics['precisions'], fnn10Metrics['precisions'],
    cnn3Metrics['precisions'], cnn5Metrics['precisions'], cnn10Metrics['precisions'],
    lstm3Metrics['precisions'], gru3Metrics['precisions']
  ]
  clf_recalls = [
    mlpMetrics['recalls'], mlp2layersMetrics['recalls'],
    rfMetrics['recalls'], lrMetrics['recalls'],
    fnn3Metrics['recalls'], fnn5Metrics['recalls'], fnn10Metrics['recalls'],
    cnn3Metrics['recalls'], cnn5Metrics['recalls'], cnn10Metrics['recalls'],
    lstm3Metrics['recalls'], gru3Metrics['recalls']
  ]
  clf_avgprecscores = [
    mlpMetrics['avgprecscores'], mlp2layersMetrics['avgprecscores'],
    rfMetrics['avgprecscores'], lrMetrics['avgprecscores'],
    fnn3Metrics['avgprecscores'], fnn5Metrics['avgprecscores'], fnn10Metrics['avgprecscores'],
    cnn3Metrics['avgprecscores'], cnn5Metrics['avgprecscores'], cnn10Metrics['avgprecscores'],
    lstm3Metrics['avgprecscores'], gru3Metrics['avgprecscores']
  ]
  for iclf in range(len(clf_labels)):
    # >>> POST-PROCESSING FOR VSTACK <<<
    clf_rs = [clf_tprs[iclf], clf_fprs[iclf], clf_precisions, clf_recalls]
    for jclf in clf_rs:
      max_index = np.inf
      for irow in range(len(jclf)):
        if max_index > len(jclf[irow]):
          max_index = len(jclf[irow])
      for irow in range(len(jclf)):
        jclf[irow] = jclf[irow][:max_index]  
    # >>> METRICS FOR ROC CURVE <<<
    i_roc_auc_nmc = np.asarray(clf_rocaucs[iclf])
    roc_auc_means[clf_labels[iclf]] = i_roc_auc_nmc.mean()
    roc_auc_stds[clf_labels[iclf]] = i_roc_auc_nmc.std()
    tprs_vstack[clf_labels[iclf]] = np.vstack(clf_tprs[iclf]).mean(axis = 0)
    fprs_vstack[clf_labels[iclf]] = np.vstack(clf_fprs[iclf]).mean(axis = 0)
    #
    iAvgprecscores = np.asarray(clf_avgprecscores[iclf])
    avgprecscores_means[clf_labels[iclf]] = iAvgprecscores.mean()
    avgprecscores_stds[clf_labels[iclf]] = iAvgprecscores.std()
    precisions_vstack[clf_labels[iclf]] = np.vstack(clf_precisions[iclf]).mean(axis = 0)
    recalls_vstack[clf_labels[iclf]] = np.vstack(clf_recalls[iclf]).mean(axis = 0)
  #
  dictMetrics = {
    "fprs_vstack": fprs_vstack, "tprs_vstack": tprs_vstack,
    "roc_auc_means": roc_auc_means, "roc_auc_stds": roc_auc_stds,
    "precisions_vstack": precisions_vstack, "recalls_vstack": recalls_vstack,
    "avgprecscores_means": avgprecscores_means, "avgprecscores_stds": avgprecscores_stds
  }
  return dictMetrics
#
#
# ===============================
# ===     pipelinePlotsTL     ===
# ===============================
def pipelinePlotsTL(whichTLdataset, dictMetrics, scoringMetric = None):
  """Pipeline for transfer learning plots 
  __date__ = 14-Jan-22
  __lastUpdated__ = 14-Jan-22
  """
  # === INITIALIZE VARIABLES ===
  lgndLines = {
    'MLP': 'b:',
    'MLP_2layers': 'g:',
    'RF': 'r:',
    'LR': 'c:',
    'FFN3': 'm-',
    'FFN5': 'y-',
    'FFN10': 'k-', 
    'CNN3': 'b-',
    'CNN5': 'g-',
    'CNN10': 'r-',
    'LSTM3': 'c-.',
    'GRU3': 'm-.'
  }
  lgndText = {
    'MLP': 'MLP 1 layer',
    'MLP_2layers': 'MLP 2 layers',
    'RF': 'RF',
    'LR': 'LR',
    'FFN3': 'FFN3',
    'FFN5': 'FFN5',
    'FFN10': 'FFN10', 
    'CNN3': 'CNN3',
    'CNN5': 'CNN5',
    'CNN10': 'CNN10',
    'LSTM3': 'LSTM',
    'GRU3': 'GRU'
  }
  # === SORT DECREASING ORDER ===
  refMetric = dictMetrics[scoringMetric+'_means']
  refMetricStd = dictMetrics[scoringMetric+'_stds']
  #
  indxsort = np.array(list(refMetric.values())).argsort()[::-1]
  clflbls = np.asarray(list(refMetric.keys()))[indxsort]
  indxvalues = np.asarray(list(refMetric.values()))[indxsort]
  #
  meansSorted = dict(zip(clflbls, indxvalues))
  stdsSorted = dict(zip(
    np.asarray(list(refMetricStd.keys()))[indxsort],
    np.asarray(list(refMetricStd.values()))[indxsort]
  ))
  # === VARIABLES FOR PLOTS ===
  if scoringMetric == 'avgprecscores':
    metric1 = 'precisions_vstack'
    metric2 = 'recalls_vstack'
    savnm = 'avgprec'
    plotFnct = plotPrecisionRecallTL
  else:
    metric1 = 'fprs_vstack'
    metric2 = 'tprs_vstack'
    savnm = 'aucroc'
    plotFnct = plotRocCurve_tl_boostrap
  # === PLOT ===
  plotFnct(
    dictMetrics[metric1], dictMetrics[metric2],
    meansSorted, stdsSorted, clflbls,
    lgndText, lgndLines,
    is_savefig = True,
    savefig_nm = 'boot_plot_'+savnm+str(whichTLdataset)+'.pdf'
  )
#
#
#
# ==============================================================================
# ==============================================================================
#                   FUNCTIONS FOR SIMILARITY POST-PROC
# ==============================================================================
# ==============================================================================
def ucorr(ar1, ar2): return np.corrcoef(ar1, ar2)[0,1];
#
def corrpipeline(lbls, ar1, ar2):
  if ar1.shape != ar2.shape: P('shape arrays mismatch.'); return None;
  allcorr = 0
  nmc = len(ar1)
  for irow in range(nmc):
    tmpcor = urnd(ucorr(ar1[irow], ar2[irow]))
    allcorr += tmpcor
    P(
      "%-35s: %s" % (
        lbls[irow],
        str("{:.3f}".format(tmpcor))))
  # ENDFOR
  allcorr /= nmc
  P('\n%-35s: %s' % ('all correlation', str(urnd(allcorr))))
# END FUNCTION
#
def postprocmetrics(bnch):
  aggauc, aggf1, aggap = [], [], []
  for idata in range(len(bnch.modelaucs)):
    l1 = [bnch.modelaucs.mean(axis=1)[idata]]
    l2 = list(bnch.transfrlrngaucs.mean(axis=2)[idata])
    aggauc.append([*l1, *l2])
    #
    l1 = [bnch.modelf1s.mean(axis=1)[idata]]
    l2 = list(bnch.transfrlrngf1s.mean(axis=2)[idata])
    aggf1.append([*l1, *l2])
    #
    l1 = [bnch.modelavgprecs.mean(axis=1)[idata]]
    l2 = list(bnch.transfrlrngavgprecs.mean(axis=2)[idata])
    aggap.append([*l1, *l2])
  # ENDFOR
  return aggauc, aggf1, aggap
# END FUNCTION
#