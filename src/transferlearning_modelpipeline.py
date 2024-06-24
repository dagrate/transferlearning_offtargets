import numpy as np
import pandas as pd
from sklearn.metrics import (
  classification_report, roc_auc_score,
  confusion_matrix, f1_score,
  roc_curve, precision_score, recall_score,
  auc, average_precision_score,
  precision_recall_curve, accuracy_score
)
#
class modelpipeline():
  def __init__(self, estimator, xtrain, ytrain, xtest, ytest, verbose):
    self.estimator = estimator
    self.x_train = xtrain
    self.y_train = ytrain
    self.x_test = xtest
    self.y_test = ytest
    self.verbose = verbose
    self.ypred = np.zeros((len(self.y_test)))
    self.yscore = np.zeros((len(self.y_test),2))
  # end of __init__
  #
  def brierScore(y_test, yscore):
    """Compute the Brier score (0 = best, 1 = worst). 
    Parameters
    ----------
    y_test : array-like
      true target series
    yscore : array-like
      predicted scores
    Returns
    -------
    bscore : float 
      Brier score
    """
    bscore = 1/len(y_test)
    bscore *= np.sum( np.power( yscore[:,1]-y_test, 2 ) )
    return bscore
  # end of brierScore
  #
  def dispConfMatrixAsArray(y_test, ypred, disp=True):
    """Display and return the confusion matrix as array.
    Parameters
    ----------
    y_test : array-like
      true target series
    ypred : array-like
      predicted target series
    disp : boolean
      diplay the confusion matrix
    Returns
    -------
    confmatrix : array-like
      pandas dataframe of the confusion matrix
    """
    confmatrix = confusion_matrix(y_test,ypred)
    tn,fp,fn,tp = confmatrix.ravel()
    if disp == True:
      print('\nConfusion Matrix')
      print("%-3s" % 'TN:', "%-5s" % tn,
        "|  %-3s" % 'FP:', "%-5s" % fp)
      print("%-3s" % 'FN:', "%-5s" % fn,
        "|  %-3s" % 'TP:', "%-5s" % tp)
    return confmatrix
  # end of dispConfMatrixAsArray
  #
  def getClassificationMetrics(self):
    """Compute metrics for classification models.
    Parameters
    ----------
    self : class-like
      class object
    """
    posLabel = np.unique(self.y_test)
    print("\nModel Metrics:")
    print("%-40s" % ("Mean Accuracy:"),
      "{:.3f}".format(self.estimator.score(self.x_test, self.y_test))
    )
    print("%-40s" % ("ROC AUC Score:"),
      "{:.3f}".format(roc_auc_score(self.y_test, self.yscore[:,1]))
    )
    print("%-40s" % ("Brier Score:"), "{:.3f}".format(
      modelpipeline.brierScore(self.y_test, self.yscore))
    )
    for n in posLabel:
      print("%-40s" % ("F1 Score Class " + str(n) + " :"), 
        "{:.3f}".format(
          f1_score(self.y_test,self.ypred,pos_label=n))
      )
      print("%-40s" % ("Recall Score Class "+str(n)+" :"), 
        "{:.3f}".format(
          recall_score(self.y_test,self.ypred,pos_label=n))
      )
      print("%-40s" % ("Avrg Precision Score Class "+str(n)+" :"), 
        "{:.3f}".format(
          average_precision_score(self.y_test,self.yscore[:,1],pos_label=n))
      )
    # end for
    _ = modelpipeline.dispConfMatrixAsArray(self.y_test,self.ypred,disp=True)
  # end of getClassificationMetrics
  #
  def modelTrain(self):
    """Training pipeline.
    """
    self.estimator = self.estimator.fit(self.x_train,self.y_train)
    return self
  # end of modelTrain
  #
  def modelPredict(self):
    """Predict pipeline.
    """
    self.ypred = self.estimator.predict(self.x_test)
    self.yscore = self.estimator.predict_proba(self.x_test)
    if self.verbose:
      modelpipeline.getClassificationMetrics(self)
    return self
  # end of modeltrain
# end of modelpipeline
