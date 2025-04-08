import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
#
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import (models, layers)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.layers import (
  Conv2D, MaxPooling2D, MaxPool2D,
  concatenate, BatchNormalization, 
  Dense, Dropout, Flatten, Input
)
import transferlearning_utils as tlrn
#
# ========================================================
# ========================================================
#        GLOBAL FUNCTIONS FOR TENSORFLOW MODELS
# ========================================================
# ========================================================
def define_data_test_size(idata = 0):
  if idata == 0:
    data_test_size = .3
  elif idata == 1:
    data_test_size = .01
  elif idata == 2:
    data_test_size = .025
  return data_test_size
#
def data_split_for_fnns(
    data,
    testsize = .3,
    verbose = False):
  if verbose:
    tlrn.printClassImbalance( data.target )
  xtrain, xtest, ytrain, ytest = train_test_split(
    data.data,
    pd.Series( data.target ),
    test_size = testsize,
    shuffle = True,
    random_state = 0
  )
  xtrain = tlrn.reshapeArr( xtrain )
  xtest = tlrn.reshapeArr( xtest )
  return xtrain, xtest, ytrain, ytest
#
def data_split_for_cnns(
    data,
    testsize = .3,
    verbose = False):
  if verbose:
    tlrn.printClassImbalance( data.target )
  xtrain, xtest, ytrain, ytest = train_test_split(
    data.data,
    pd.Series( data.target ),
    test_size = testsize,
    shuffle = True,
    random_state = 0
  )
  xtrain = xtrain.reshape(-1, 24, 7, 1)
  xtest = xtest.reshape(-1, 24, 7, 1)
  return xtrain, xtest, ytrain, ytest
#
def data_split_for_rnns(
    data,
    testsize = .3,
    verbose = False):
  if verbose:
    tlrn.printClassImbalance( data.target )
  xtrain, xtest, ytrain, ytest = train_test_split(
    data.data,
    pd.Series( data.target ),
    test_size = testsize,
    shuffle = True,
    random_state = 0
  )
  return xtrain, xtest, ytrain, ytest
#
def bootstrapForHypertuning(
    data_to_bootstrap,
    n_samples = 2000,
    is_reshape = True):
  # === split pos. class and neg. class ===
  indx_false = data_to_bootstrap.target == 0
  indx_true = data_to_bootstrap.target == 1
  data_hypertune_false = data_to_bootstrap.data[indx_false][:n_samples]
  target_hypertune_false = data_to_bootstrap.target[indx_false][:n_samples]
  data_hypertune_true = data_to_bootstrap.data[indx_true][:n_samples]
  target_hypertune_true = data_to_bootstrap.target[indx_true][:n_samples]
  # === concatenate data ===
  data_hypertune = np.concatenate(
    (
      data_hypertune_false,
      data_hypertune_true
    ),
    axis = 0
  )
  target_hypertune = np.concatenate(
    (
      target_hypertune_false,
      target_hypertune_true
    ),
    axis = 0
  )
  # === shuffle data ===
  rng = np.random.default_rng()
  numbers = rng.choice(
    len( target_hypertune ),
    size = len( target_hypertune ),
    replace = False
  )
  data_hypertune = data_hypertune[numbers]
  target_hypertune = target_hypertune[numbers]
  if is_reshape:
    data_hypertune = tlrn.reshapeArr( data_hypertune )
  return data_hypertune, target_hypertune
#
def fix_callbacks(
    monitor = 'val_loss', factor = 0.2, 
    verbose = 0, mode = 'auto',
    cooldown = 0, min_lr = 0,
    min_delta_reduce_learning = 0.02,
    patience_reduce_learning = 8,
    patience_early_stopping = 8,
    min_delta_early_stopping = 0.02
    ):
  """Callbacks for deep learning models."""
  reduce_learning = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = monitor,
    factor = factor, 
    verbose = verbose, 
    mode = mode,    
    min_lr = min_lr,
    cooldown = 0,
    patience = patience_reduce_learning,
    min_delta = min_delta_reduce_learning,
  )
  eary_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = monitor,
    verbose = verbose,
    mode = mode,
    patience = patience_early_stopping,
    min_delta = min_delta_early_stopping
  )
  return [reduce_learning, eary_stopping]
#
# ========================================================
# ========================================================
#                      FNNS MODELS
# ========================================================
# ========================================================
def model_ffn3(
    unit_layer_1 = 64,
    unit_layer_2 = 32,
    unit_layer_3 = 16,
    unit_dropout_1 = .25,
    is_batch_normalization_1 = True,
    is_summary = False):
  """Define FNN3 model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(168,)))
  model.add(tf.keras.layers.Dense(unit_layer_1, activation='relu'))
  if is_batch_normalization_1:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_2, activation='relu'))
  if unit_dropout_1 > 0:
    model.add( Dropout( unit_dropout_1 ) )
  model.add(tf.keras.layers.Dense(unit_layer_3, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='relu'))
  if is_summary:
    model.build([168])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def model_ffn5(
    unit_layer_1 = 128, unit_layer_2 = 64,
    unit_layer_3 = 32, unit_layer_4 = 16,
    unit_layer_5 = 8,
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_summary = False):
  """Define FNN5 model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(168,)))
  model.add(tf.keras.layers.Dense(unit_layer_1, activation='relu'))
  if is_batch_normalization_1:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_2, activation='relu'))
  if is_batch_normalization_2:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_3, activation='relu'))
  if unit_dropout_1 > 0:
    model.add( Dropout( unit_dropout_1 ) )
  model.add(tf.keras.layers.Dense(unit_layer_4, activation='relu'))
  if unit_dropout_2 > 0:
    model.add( Dropout( unit_dropout_2 ) )
  model.add(tf.keras.layers.Dense(unit_layer_5, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='relu'))
  if is_summary:
    model.build([168])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def model_ffn10(
    unit_layer_1 = 256, unit_layer_2 = 128,
    unit_layer_3 = 128, unit_layer_4 = 64,
    unit_layer_5 = 64, unit_layer_6 = 32,
    unit_layer_7 = 32, unit_layer_8 = 16,
    unit_layer_9 = 16, unit_layer_10 = 8,
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    unit_dropout_3 = .25, unit_dropout_4 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True,
    is_batch_normalization_4 = True,
    is_batch_normalization_5 = True,
    is_batch_normalization_6 = True,
    is_summary = False):
  """Define FNN10 model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(168,)))
  model.add(tf.keras.layers.Dense(unit_layer_1, activation='relu'))
  if is_batch_normalization_1:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_2, activation='relu'))
  if is_batch_normalization_2:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_3, activation='relu'))
  if unit_dropout_1 > 0:
    model.add( Dropout( unit_dropout_1 ) )
  model.add(tf.keras.layers.Dense(unit_layer_4, activation='relu'))
  if is_batch_normalization_3:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_5, activation='relu'))
  if is_batch_normalization_4:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_6, activation='relu'))
  if unit_dropout_2 > 0:
    model.add( Dropout( unit_dropout_2 ) )
  model.add(tf.keras.layers.Dense(unit_layer_7, activation='relu'))
  if is_batch_normalization_5:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(unit_layer_8, activation='relu'))
  if unit_dropout_3 > 0:
    model.add( Dropout( unit_dropout_3 ) )
  model.add(tf.keras.layers.Dense(unit_layer_9, activation='relu'))
  if unit_dropout_4 > 0:
    model.add( Dropout( unit_dropout_4 ) )
  model.add(tf.keras.layers.Dense(unit_layer_10, activation='relu'))
  if is_batch_normalization_6:
    model.add( BatchNormalization() )
  model.add(tf.keras.layers.Dense(1, activation='relu'))
  if is_summary:
    model.build([168])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy'])
  return model
#
def find_params_for_fnn_3_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(
        0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(
        0, len(params_grid['layer_2_grid']))],
    'rand_layer_3': params_grid['layer_3_grid'][
      np.random.randint(
        0, len(params_grid['layer_3_grid']))],
    'rand_dropout': params_grid['dropout_grid'][
      np.random.randint(
        0, len(params_grid['dropout_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_fnn_3 = model_ffn3(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    unit_layer_3 = random_params_grid['rand_layer_3'],
    unit_dropout_1 = random_params_grid['rand_dropout'],
    is_batch_normalization_1 = True
  )
  model_fnn_3.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 200,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_fnn_3, random_params_grid
#
def fnn_3_layers_training_pipeline(
    data_dict,
    params_grid):
  dl_model = model_ffn3(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    unit_layer_3 = params_grid['unit_layer_3'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 200,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def fnn_5_layers_training_pipeline(
    data_dict,
    params_grid):
  dl_model = model_ffn5(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    unit_layer_3 = params_grid['unit_layer_3'],
    unit_layer_4 = params_grid['unit_layer_4'],
    unit_layer_5 = params_grid['unit_layer_5'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_1'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 500,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def fnn_10_layers_training_pipeline(
    data_dict,
    params_grid):
  dl_model = model_ffn10(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    unit_layer_3 = params_grid['unit_layer_3'],
    unit_layer_4 = params_grid['unit_layer_4'],
    unit_layer_5 = params_grid['unit_layer_5'],
    unit_layer_6 = params_grid['unit_layer_6'],
    unit_layer_7 = params_grid['unit_layer_7'],
    unit_layer_8 = params_grid['unit_layer_8'],
    unit_layer_9 = params_grid['unit_layer_9'],
    unit_layer_10 = params_grid['unit_layer_10'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    unit_dropout_3 = params_grid['unit_dropout_3'],
    unit_dropout_4 = params_grid['unit_dropout_4'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2'],
    is_batch_normalization_3 = params_grid['is_batch_normalization_3'],
    is_batch_normalization_4 = params_grid['is_batch_normalization_4'],
    is_batch_normalization_5 = params_grid['is_batch_normalization_5'],
    is_batch_normalization_6 = params_grid['is_batch_normalization_6']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def find_params_for_fnn_5_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(
        0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(
        0, len(params_grid['layer_2_grid']))],
    'rand_layer_3': params_grid['layer_3_grid'][
      np.random.randint(
        0, len(params_grid['layer_3_grid']))],
    'rand_layer_4': params_grid['layer_4_grid'][
      np.random.randint(
        0, len(params_grid['layer_4_grid']))],
    'rand_layer_5': params_grid['layer_5_grid'][
      np.random.randint(
        0, len(params_grid['layer_5_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(
        0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(
        0, len(params_grid['dropout_2_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_fnn_5 = model_ffn5(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    unit_layer_3 = random_params_grid['rand_layer_3'],
    unit_layer_4 = random_params_grid['rand_layer_4'],
    unit_layer_5 = random_params_grid['rand_layer_5'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True
  )
  model_fnn_5.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 500,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_fnn_5, random_params_grid
#
def find_params_for_fnn_10_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(
        0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(
        0, len(params_grid['layer_2_grid']))],
    'rand_layer_3': params_grid['layer_3_grid'][
      np.random.randint(
        0, len(params_grid['layer_3_grid']))],
    'rand_layer_4': params_grid['layer_4_grid'][
      np.random.randint(
        0, len(params_grid['layer_4_grid']))],
    'rand_layer_5': params_grid['layer_5_grid'][
      np.random.randint(
        0, len(params_grid['layer_5_grid']))],
    'rand_layer_6': params_grid['layer_6_grid'][
      np.random.randint(
        0, len(params_grid['layer_6_grid']))],
    'rand_layer_7': params_grid['layer_7_grid'][
      np.random.randint(
        0, len(params_grid['layer_7_grid']))],
    'rand_layer_8': params_grid['layer_8_grid'][
      np.random.randint(
        0, len(params_grid['layer_8_grid']))],
    'rand_layer_9': params_grid['layer_9_grid'][
      np.random.randint(
        0, len(params_grid['layer_9_grid']))],
    'rand_layer_10': params_grid['layer_10_grid'][
      np.random.randint(
        0, len(params_grid['layer_10_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(
        0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(
        0, len(params_grid['dropout_2_grid']))],
    'rand_dropout_3': params_grid['dropout_3_grid'][
      np.random.randint(
        0, len(params_grid['dropout_3_grid']))],
    'rand_dropout_4': params_grid['dropout_4_grid'][
      np.random.randint(
        0, len(params_grid['dropout_4_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_fnn_10 = model_ffn10(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    unit_layer_3 = random_params_grid['rand_layer_3'],
    unit_layer_4 = random_params_grid['rand_layer_4'],
    unit_layer_5 = random_params_grid['rand_layer_5'],
    unit_layer_6 = random_params_grid['rand_layer_6'],
    unit_layer_7 = random_params_grid['rand_layer_7'],
    unit_layer_8 = random_params_grid['rand_layer_8'],
    unit_layer_9 = random_params_grid['rand_layer_9'],
    unit_layer_10 = random_params_grid['rand_layer_10'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    unit_dropout_3 = random_params_grid['rand_dropout_3'],
    unit_dropout_4 = random_params_grid['rand_dropout_4'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True,
    is_batch_normalization_4 = True,
    is_batch_normalization_5 = True,
    is_batch_normalization_6 = True
  )
  model_fnn_10.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_fnn_10, random_params_grid
#
def save_params_for_fnn_3_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'unit_layer_3': random_params_grid['rand_layer_3'],
    'unit_dropout_1': random_params_grid['rand_dropout'],
    'is_batch_normalization_1': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
def save_params_for_fnn_5_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'unit_layer_3': random_params_grid['rand_layer_3'],
    'unit_layer_4': random_params_grid['rand_layer_4'],
    'unit_layer_5': random_params_grid['rand_layer_5'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
def save_params_for_fnn_10_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'unit_layer_3': random_params_grid['rand_layer_3'],
    'unit_layer_4': random_params_grid['rand_layer_4'],
    'unit_layer_5': random_params_grid['rand_layer_5'],
    'unit_layer_6': random_params_grid['rand_layer_6'],
    'unit_layer_7': random_params_grid['rand_layer_7'],
    'unit_layer_8': random_params_grid['rand_layer_8'],
    'unit_layer_9': random_params_grid['rand_layer_9'],
    'unit_layer_10': random_params_grid['rand_layer_10'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'unit_dropout_3': random_params_grid['rand_dropout_3'],
    'unit_dropout_4': random_params_grid['rand_dropout_4'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'is_batch_normalization_3': True,
    'is_batch_normalization_4': True,
    'is_batch_normalization_5': True,
    'is_batch_normalization_6': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
# ========================================================
# ========================================================
#                      CNNS MODELS
# ========================================================
# ========================================================
def model_cnn3(
    unit_layer_1 = 32, unit_layer_2 = 32,
    activation_layer_1 = 'relu', activation_layer_2 = 'relu',
    activation_layer_3 = 'relu',
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    input_shape = (24,7,1), is_summary = False):
  """Define CNN3 model."""
  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.Conv2D(
      unit_layer_1,
      kernel_size = (3, 3),
      activation = activation_layer_1,
      input_shape = input_shape
    )
  )
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if is_batch_normalization_1:
    model.add(BatchNormalization())
  if unit_dropout_1 > 0:
    model.add(Dropout(unit_dropout_1))
  model.add(Flatten())
  model.add(
    tf.keras.layers.Dense(unit_layer_2, activation = activation_layer_2)
  )
  if is_batch_normalization_2:
    model.add(BatchNormalization())
  if unit_dropout_2 > 0:
    model.add(Dropout(unit_dropout_2))
  model.add(
    tf.keras.layers.Dense(1, activation = activation_layer_3)
  )
  if is_summary:
    model.build([24,7,1])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def model_cnn5(
    unit_layer_1 = 32, unit_layer_2 = 64,
    unit_layer_3 = 128, unit_layer_4 = 64,
    activation_layer_1 = 'relu', activation_layer_2 = 'relu',
    activation_layer_3 = 'relu', activation_layer_4 = 'relu',
    activation_layer_5 = 'relu',
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True,
    input_shape = (24,7,1), is_summary = False):
  """Define CNN5 model."""
  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.Conv2D(
      unit_layer_1,
      kernel_size = (3, 3),
      activation = activation_layer_1,
      input_shape = input_shape
    )
  )
  model.add(
    tf.keras.layers.Conv2D(
      unit_layer_2,
      kernel_size = (1, 1),
      activation = activation_layer_2
    )
  )
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if is_batch_normalization_1:
    model.add(BatchNormalization())
  if unit_dropout_1 > 0:
    model.add(Dropout(unit_dropout_1))
  model.add(Flatten())
  model.add(
    tf.keras.layers.Dense(unit_layer_3, activation = activation_layer_3)
  )
  if is_batch_normalization_2:
    model.add(BatchNormalization())
  if unit_dropout_2 > 0:
    model.add(Dropout(unit_dropout_2))
  model.add(
    tf.keras.layers.Dense(unit_layer_4, activation = activation_layer_4)
  )
  if is_batch_normalization_3:
    model.add(BatchNormalization())
  model.add(
    tf.keras.layers.Dense(1, activation = activation_layer_5)
  )
  if is_summary:
    model.build([24,7,1])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def model_cnn10(
    unit_layer_1 = 32, unit_layer_2 = 64,
    unit_layer_3 = 128, unit_layer_4 = 64,
    unit_layer_5 = 64, unit_layer_6 = 64,
    unit_layer_7 = 32, unit_layer_8 = 32,
    unit_layer_9 = 8,
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    unit_dropout_3 = .25, unit_dropout_4 = .25,
    unit_dropout_5 = .25, unit_dropout_6 = .25,
    activation_layer_1 = 'relu', activation_layer_2 = 'relu',
    activation_layer_3 = 'relu', activation_layer_4 = 'relu',
    activation_layer_5 = 'relu', activation_layer_6 = 'relu',
    activation_layer_7 = 'relu', activation_layer_8 = 'relu',
    activation_layer_9 = 'relu', activation_layer_10 = 'relu',
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True,
    is_batch_normalization_4 = True,
    is_batch_normalization_5 = True,
    is_batch_normalization_6 = True,
    is_batch_normalization_7 = True,
    input_shape = (24,7,1), is_summary = False):
  """Define CNN10 model."""
  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.Conv2D(
      unit_layer_1,
      kernel_size = (3, 3),
      activation = activation_layer_1,
      input_shape = input_shape
    )
  )
  model.add(
    tf.keras.layers.Conv2D(
      unit_layer_2,
      kernel_size = (1, 1),
      activation = activation_layer_2,
    )
  )
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if is_batch_normalization_1:
    model.add(BatchNormalization())
  if unit_dropout_1 > 0:
    model.add(Dropout(unit_dropout_1))
  model.add(Flatten())
  model.add(
    tf.keras.layers.Dense(unit_layer_3, activation = activation_layer_3)
  )
  if is_batch_normalization_2:
    model.add( BatchNormalization() )
  if unit_dropout_2 > 0:
    model.add( Dropout( unit_dropout_2 ) )
  model.add(
    tf.keras.layers.Dense(unit_layer_4, activation = activation_layer_4)
  )
  if is_batch_normalization_3:
    model.add( BatchNormalization() )
  if unit_dropout_3 > 0:
    model.add( Dropout( unit_dropout_3 ) )
  model.add(
    tf.keras.layers.Dense(unit_layer_5, activation = activation_layer_5)
  )
  if is_batch_normalization_4:
    model.add( BatchNormalization() )
  model.add(
    tf.keras.layers.Dense(unit_layer_6, activation = activation_layer_6)
  )
  if is_batch_normalization_5:
    model.add( BatchNormalization() )
  if unit_dropout_4 > 0:
    model.add( Dropout( unit_dropout_4 ) )
  model.add(
    tf.keras.layers.Dense(unit_layer_7, activation = activation_layer_7)
  )
  if is_batch_normalization_6:
    model.add( BatchNormalization() )
  model.add(
    tf.keras.layers.Dense(unit_layer_8, activation = activation_layer_8)
  )
  if is_batch_normalization_6:
    model.add( BatchNormalization() )
  if unit_dropout_5 > 0:
    model.add( Dropout( unit_dropout_5 ) )
  model.add(
    tf.keras.layers.Dense(unit_layer_9, activation = activation_layer_9)
  )
  if is_batch_normalization_7:
    model.add( BatchNormalization() )
  if unit_dropout_6 > 0:
    model.add( Dropout( unit_dropout_6 ) )
  model.add(
    tf.keras.layers.Dense(1, activation = activation_layer_10)
  )
  if is_summary:
    model.build([24,7,1])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def cnn_3_layers_training_pipeline(data_dict, params_grid):
  """CNN3 layers training pipeline."""
  dl_model = model_cnn3(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    activation_layer_1 = params_grid['activation_layer_1'],
    activation_layer_2 = params_grid['activation_layer_2'],
    activation_layer_3 = params_grid['activation_layer_3'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def cnn_5_layers_training_pipeline(data_dict, params_grid):
  """CNN5 layers training pipeline."""
  dl_model = model_cnn5(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    unit_layer_3 = params_grid['unit_layer_3'],
    unit_layer_4 = params_grid['unit_layer_4'],
    activation_layer_1 = params_grid['activation_layer_1'],
    activation_layer_2 = params_grid['activation_layer_2'],
    activation_layer_3 = params_grid['activation_layer_3'],
    activation_layer_4 = params_grid['activation_layer_4'],
    activation_layer_5 = params_grid['activation_layer_5'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2'],
    is_batch_normalization_3 = params_grid['is_batch_normalization_3']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 2000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def cnn_10_layers_training_pipeline(data_dict, params_grid):
  """CNN10 layers training pipeline."""
  dl_model = model_cnn10(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    unit_layer_3 = params_grid['unit_layer_3'],
    unit_layer_4 = params_grid['unit_layer_4'],
    unit_layer_5 = params_grid['unit_layer_5'],
    unit_layer_6 = params_grid['unit_layer_6'],
    unit_layer_7 = params_grid['unit_layer_7'],
    unit_layer_8 = params_grid['unit_layer_8'],
    unit_layer_9 = params_grid['unit_layer_9'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    unit_dropout_3 = params_grid['unit_dropout_3'],
    unit_dropout_4 = params_grid['unit_dropout_4'],
    unit_dropout_5 = params_grid['unit_dropout_5'],
    unit_dropout_6 = params_grid['unit_dropout_6'],
    activation_layer_1 = params_grid['activation_layer_1'],
    activation_layer_2 = params_grid['activation_layer_2'],
    activation_layer_3 = params_grid['activation_layer_3'],
    activation_layer_4 = params_grid['activation_layer_4'],
    activation_layer_5 = params_grid['activation_layer_5'],
    activation_layer_6 = params_grid['activation_layer_6'],
    activation_layer_7 = params_grid['activation_layer_7'],
    activation_layer_8 = params_grid['activation_layer_8'],
    activation_layer_9 = params_grid['activation_layer_9'],
    activation_layer_10 = params_grid['activation_layer_10'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2'],
    is_batch_normalization_3 = params_grid['is_batch_normalization_3'],
    is_batch_normalization_4 = params_grid['is_batch_normalization_4'],
    is_batch_normalization_5 = params_grid['is_batch_normalization_5'],
    is_batch_normalization_6 = params_grid['is_batch_normalization_6'],
    is_batch_normalization_7 = params_grid['is_batch_normalization_7']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 3000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def find_params_for_cnn_3_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(0, len(params_grid['layer_2_grid']))],
    'rand_activation_layer_1': params_grid['activation_layer_1_grid'][
      np.random.randint(0, len(params_grid['activation_layer_1_grid']))],
    'rand_activation_layer_2': params_grid['activation_layer_2_grid'][
      np.random.randint(0, len(params_grid['activation_layer_2_grid']))],
    'rand_activation_layer_3': params_grid['activation_layer_3_grid'][
      np.random.randint(0, len(params_grid['activation_layer_3_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(0, len(params_grid['dropout_2_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_cnn_3 = model_cnn3(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    activation_layer_1 = random_params_grid['rand_activation_layer_1'],
    activation_layer_2 = random_params_grid['rand_activation_layer_2'],
    activation_layer_3 = random_params_grid['rand_activation_layer_3'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True
  )
  model_cnn_3.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_cnn_3, random_params_grid
#
def find_params_for_cnn_5_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(0, len(params_grid['layer_2_grid']))],
    'rand_layer_3': params_grid['layer_3_grid'][
      np.random.randint(0, len(params_grid['layer_3_grid']))],
    'rand_layer_4': params_grid['layer_4_grid'][
      np.random.randint(0, len(params_grid['layer_4_grid']))],
    'rand_activation_layer_1': params_grid['activation_layer_1_grid'][
      np.random.randint(0, len(params_grid['activation_layer_1_grid']))],
    'rand_activation_layer_2': params_grid['activation_layer_2_grid'][
      np.random.randint(0, len(params_grid['activation_layer_2_grid']))],
    'rand_activation_layer_3': params_grid['activation_layer_3_grid'][
      np.random.randint(0, len(params_grid['activation_layer_3_grid']))],
    'rand_activation_layer_4': params_grid['activation_layer_4_grid'][
      np.random.randint(0, len(params_grid['activation_layer_4_grid']))],
    'rand_activation_layer_5': params_grid['activation_layer_5_grid'][
      np.random.randint(0, len(params_grid['activation_layer_5_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(0, len(params_grid['dropout_2_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_cnn = model_cnn5(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    unit_layer_3 = random_params_grid['rand_layer_3'],
    unit_layer_4 = random_params_grid['rand_layer_4'],
    activation_layer_1 = random_params_grid['rand_activation_layer_1'],
    activation_layer_2 = random_params_grid['rand_activation_layer_2'],
    activation_layer_3 = random_params_grid['rand_activation_layer_3'],
    activation_layer_4 = random_params_grid['rand_activation_layer_4'],
    activation_layer_5 = random_params_grid['rand_activation_layer_5'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True
  )
  model_cnn.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 2000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_cnn, random_params_grid
#
def find_params_for_cnn_10_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(0, len(params_grid['layer_2_grid']))],
    'rand_layer_3': params_grid['layer_3_grid'][
      np.random.randint(0, len(params_grid['layer_3_grid']))],
    'rand_layer_4': params_grid['layer_4_grid'][
      np.random.randint(0, len(params_grid['layer_4_grid']))],
    'rand_layer_5': params_grid['layer_5_grid'][
      np.random.randint(0, len(params_grid['layer_5_grid']))],
    'rand_layer_6': params_grid['layer_6_grid'][
      np.random.randint(0, len(params_grid['layer_6_grid']))],
    'rand_layer_7': params_grid['layer_7_grid'][
      np.random.randint(0, len(params_grid['layer_7_grid']))],
    'rand_layer_8': params_grid['layer_8_grid'][
      np.random.randint(0, len(params_grid['layer_8_grid']))],
    'rand_layer_9': params_grid['layer_9_grid'][
      np.random.randint(0, len(params_grid['layer_9_grid']))],
    'rand_activation_layer_1': params_grid['activation_layer_1_grid'][
      np.random.randint(0, len(params_grid['activation_layer_1_grid']))],
    'rand_activation_layer_2': params_grid['activation_layer_2_grid'][
      np.random.randint(0, len(params_grid['activation_layer_2_grid']))],
    'rand_activation_layer_3': params_grid['activation_layer_3_grid'][
      np.random.randint(0, len(params_grid['activation_layer_3_grid']))],
    'rand_activation_layer_4': params_grid['activation_layer_4_grid'][
      np.random.randint(0, len(params_grid['activation_layer_4_grid']))],
    'rand_activation_layer_5': params_grid['activation_layer_5_grid'][
      np.random.randint(0, len(params_grid['activation_layer_5_grid']))],
    'rand_activation_layer_6': params_grid['activation_layer_6_grid'][
      np.random.randint(0, len(params_grid['activation_layer_6_grid']))],
    'rand_activation_layer_7': params_grid['activation_layer_7_grid'][
      np.random.randint(0, len(params_grid['activation_layer_7_grid']))],
    'rand_activation_layer_8': params_grid['activation_layer_8_grid'][
      np.random.randint(0, len(params_grid['activation_layer_8_grid']))],
    'rand_activation_layer_9': params_grid['activation_layer_9_grid'][
      np.random.randint(0, len(params_grid['activation_layer_9_grid']))],
    'rand_activation_layer_10': params_grid['activation_layer_10_grid'][
      np.random.randint(0, len(params_grid['activation_layer_10_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(0, len(params_grid['dropout_2_grid']))],
    'rand_dropout_3': params_grid['dropout_3_grid'][
      np.random.randint(0, len(params_grid['dropout_3_grid']))],
    'rand_dropout_4': params_grid['dropout_4_grid'][
      np.random.randint(0, len(params_grid['dropout_4_grid']))],
    'rand_dropout_5': params_grid['dropout_5_grid'][
      np.random.randint(0, len(params_grid['dropout_5_grid']))],
    'rand_dropout_6': params_grid['dropout_6_grid'][
      np.random.randint(0, len(params_grid['dropout_6_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_cnn = model_cnn10(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    unit_layer_3 = random_params_grid['rand_layer_3'],
    unit_layer_4 = random_params_grid['rand_layer_4'],
    unit_layer_5 = random_params_grid['rand_layer_5'],
    unit_layer_6 = random_params_grid['rand_layer_6'],
    unit_layer_7 = random_params_grid['rand_layer_7'],
    unit_layer_8 = random_params_grid['rand_layer_8'],
    unit_layer_9 = random_params_grid['rand_layer_9'],
    activation_layer_1 = random_params_grid['rand_activation_layer_1'],
    activation_layer_2 = random_params_grid['rand_activation_layer_2'],
    activation_layer_3 = random_params_grid['rand_activation_layer_3'],
    activation_layer_4 = random_params_grid['rand_activation_layer_4'],
    activation_layer_5 = random_params_grid['rand_activation_layer_5'],
    activation_layer_6 = random_params_grid['rand_activation_layer_6'],
    activation_layer_7 = random_params_grid['rand_activation_layer_7'],
    activation_layer_8 = random_params_grid['rand_activation_layer_8'],
    activation_layer_9 = random_params_grid['rand_activation_layer_9'],
    activation_layer_10 = random_params_grid['rand_activation_layer_10'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    unit_dropout_3 = random_params_grid['rand_dropout_3'],
    unit_dropout_4 = random_params_grid['rand_dropout_4'],
    unit_dropout_5 = random_params_grid['rand_dropout_5'],
    unit_dropout_6 = random_params_grid['rand_dropout_6'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    is_batch_normalization_3 = True,
    is_batch_normalization_4 = True,
    is_batch_normalization_5 = True,
    is_batch_normalization_6 = True,
    is_batch_normalization_7 = True
  )
  model_cnn.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 3000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_cnn, random_params_grid
#
def save_params_for_cnn_3_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'activation_layer_1': random_params_grid['rand_activation_layer_1'],
    'activation_layer_2': random_params_grid['rand_activation_layer_2'],
    'activation_layer_3': random_params_grid['rand_activation_layer_3'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
def save_params_for_cnn_5_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'unit_layer_3': random_params_grid['rand_layer_3'],
    'unit_layer_4': random_params_grid['rand_layer_4'],
    'activation_layer_1': random_params_grid['rand_activation_layer_1'],
    'activation_layer_2': random_params_grid['rand_activation_layer_2'],
    'activation_layer_3': random_params_grid['rand_activation_layer_3'],
    'activation_layer_4': random_params_grid['rand_activation_layer_4'],
    'activation_layer_5': random_params_grid['rand_activation_layer_5'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'is_batch_normalization_3': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
def save_params_for_cnn_10_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'unit_layer_3': random_params_grid['rand_layer_3'],
    'unit_layer_4': random_params_grid['rand_layer_4'],
    'unit_layer_5': random_params_grid['rand_layer_5'],
    'unit_layer_6': random_params_grid['rand_layer_6'],
    'unit_layer_7': random_params_grid['rand_layer_7'],
    'unit_layer_8': random_params_grid['rand_layer_8'],
    'unit_layer_9': random_params_grid['rand_layer_9'],
    'activation_layer_1': random_params_grid['rand_activation_layer_1'],
    'activation_layer_2': random_params_grid['rand_activation_layer_2'],
    'activation_layer_3': random_params_grid['rand_activation_layer_3'],
    'activation_layer_4': random_params_grid['rand_activation_layer_4'],
    'activation_layer_5': random_params_grid['rand_activation_layer_5'],
    'activation_layer_6': random_params_grid['rand_activation_layer_6'],
    'activation_layer_7': random_params_grid['rand_activation_layer_7'],
    'activation_layer_8': random_params_grid['rand_activation_layer_8'],
    'activation_layer_9': random_params_grid['rand_activation_layer_9'],
    'activation_layer_10': random_params_grid['rand_activation_layer_10'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'unit_dropout_3': random_params_grid['rand_dropout_3'],
    'unit_dropout_4': random_params_grid['rand_dropout_4'],
    'unit_dropout_5': random_params_grid['rand_dropout_5'],
    'unit_dropout_6': random_params_grid['rand_dropout_6'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'is_batch_normalization_3': True,
    'is_batch_normalization_4': True,
    'is_batch_normalization_5': True,
    'is_batch_normalization_6': True,
    'is_batch_normalization_7': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
# ========================================================
# ========================================================
#                      RNNS MODELS
# ========================================================
# ========================================================
def model_lstm3(
    unit_layer_1 = 32, unit_layer_2 = 32,
    activation_layer_1 = 'relu', activation_layer_2 = 'relu',
    activation_layer_3 = 'relu', 
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    input_shape = (24,7), is_summary = False):
  """Define CNN3 model."""
  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.LSTM(
      unit_layer_1,
      activation = activation_layer_1,
      input_shape = input_shape
    )
  )
  if is_batch_normalization_1:
    model.add(BatchNormalization())
  if unit_dropout_1 > 0:
    model.add(Dropout(unit_dropout_1))
  model.add(
    tf.keras.layers.Dense(unit_layer_2, activation = activation_layer_2)
  )
  if is_batch_normalization_2:
    model.add(BatchNormalization())
  if unit_dropout_2 > 0:
    model.add(Dropout(unit_dropout_2))
  model.add(
    tf.keras.layers.Dense(1, activation = activation_layer_3)
  )
  if is_summary:
    model.build([24,7,1])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def model_gru3(
    unit_layer_1 = 32, unit_layer_2 = 32,
    activation_layer_1 = 'relu', activation_layer_2 = 'relu',
    activation_layer_3 = 'relu',
    unit_dropout_1 = .25, unit_dropout_2 = .25,
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True,
    input_shape = (24,7), is_summary = False):
  """Define CNN3 model."""
  model = tf.keras.models.Sequential()
  model.add(
    tf.keras.layers.GRU(
      unit_layer_1,
      activation = activation_layer_1,
      input_shape = input_shape
    )
  )
  if is_batch_normalization_1:
    model.add(BatchNormalization())
  if unit_dropout_1 > 0:
    model.add(Dropout(unit_dropout_1))
  model.add(
    tf.keras.layers.Dense(unit_layer_2, activation = activation_layer_2)
  )
  if is_batch_normalization_2:
    model.add(BatchNormalization())
  if unit_dropout_2 > 0:
    model.add(Dropout(unit_dropout_2))
  model.add(
    tf.keras.layers.Dense(1, activation = activation_layer_3)
  )
  if is_summary:
    model.build([24,7,1])
    model.summary()
  model.compile(
    loss = binary_crossentropy,
    optimizer = 'rmsprop',
    metrics = ['accuracy']
  )
  return model
#
def lstm_3_layers_training_pipeline(data_dict, params_grid):
  """LSTM 3 layers training pipeline."""
  dl_model = model_lstm3(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    activation_layer_1 = params_grid['activation_layer_1'],
    activation_layer_2 = params_grid['activation_layer_2'],
    activation_layer_3 = params_grid['activation_layer_3'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def gru_3_layers_training_pipeline(data_dict, params_grid):
  """GRU 3 layers training pipeline."""
  dl_model = model_gru3(
    unit_layer_1 = params_grid['unit_layer_1'],
    unit_layer_2 = params_grid['unit_layer_2'],
    activation_layer_1 = params_grid['activation_layer_1'],
    activation_layer_2 = params_grid['activation_layer_2'],
    activation_layer_3 = params_grid['activation_layer_3'],
    unit_dropout_1 = params_grid['unit_dropout_1'],
    unit_dropout_2 = params_grid['unit_dropout_2'],
    is_batch_normalization_1 = params_grid['is_batch_normalization_1'],
    is_batch_normalization_2 = params_grid['is_batch_normalization_2']
  )
  dl_model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = params_grid['unit_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return dl_model
#
def find_params_for_lstm_3_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(0, len(params_grid['layer_2_grid']))],
    'rand_activation_layer_1': params_grid['activation_layer_1_grid'][
      np.random.randint(0, len(params_grid['activation_layer_1_grid']))],
    'rand_activation_layer_2': params_grid['activation_layer_2_grid'][
      np.random.randint(0, len(params_grid['activation_layer_2_grid']))],
    'rand_activation_layer_3': params_grid['activation_layer_3_grid'][
      np.random.randint(0, len(params_grid['activation_layer_3_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(0, len(params_grid['dropout_2_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model_lstm_3 = model_lstm3(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    activation_layer_1 = random_params_grid['rand_activation_layer_1'],
    activation_layer_2 = random_params_grid['rand_activation_layer_2'],
    activation_layer_3 = random_params_grid['rand_activation_layer_3'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True
  )
  model_lstm_3.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model_lstm_3, random_params_grid
#
def find_params_for_gru_3_layers(
    data_dict,
    params_grid):
  random_params_grid = {
    'rand_layer_1': params_grid['layer_1_grid'][
      np.random.randint(0, len(params_grid['layer_1_grid']))],
    'rand_layer_2': params_grid['layer_2_grid'][
      np.random.randint(0, len(params_grid['layer_2_grid']))],
    'rand_activation_layer_1': params_grid['activation_layer_1_grid'][
      np.random.randint(0, len(params_grid['activation_layer_1_grid']))],
    'rand_activation_layer_2': params_grid['activation_layer_2_grid'][
      np.random.randint(0, len(params_grid['activation_layer_2_grid']))],
    'rand_activation_layer_3': params_grid['activation_layer_3_grid'][
      np.random.randint(0, len(params_grid['activation_layer_3_grid']))],
    'rand_dropout_1': params_grid['dropout_1_grid'][
      np.random.randint(0, len(params_grid['dropout_1_grid']))],
    'rand_dropout_2': params_grid['dropout_2_grid'][
      np.random.randint(0, len(params_grid['dropout_2_grid']))],
    'rand_batch': params_grid['batch_grid'][
      np.random.randint(0, len(params_grid['batch_grid']))]
  }
  model = model_gru3(
    unit_layer_1 = random_params_grid['rand_layer_1'],
    unit_layer_2 = random_params_grid['rand_layer_2'],
    activation_layer_1 = random_params_grid['rand_activation_layer_1'],
    activation_layer_2 = random_params_grid['rand_activation_layer_2'],
    activation_layer_3 = random_params_grid['rand_activation_layer_3'],
    unit_dropout_1 = random_params_grid['rand_dropout_1'],
    unit_dropout_2 = random_params_grid['rand_dropout_2'],
    is_batch_normalization_1 = True,
    is_batch_normalization_2 = True
  )
  model.fit(
    data_dict['x_boot'], 
    data_dict['y_boot'],
    batch_size = random_params_grid['rand_batch'],
    epochs = 1000,
    verbose = 0,
    validation_data = (data_dict['x_test'], data_dict['y_test']),
    callbacks = fix_callbacks()
  )
  return model, random_params_grid
#
def save_params_for_lstm_3_gru_3_layers(random_params_grid):
  best_params = {
    'unit_layer_1': random_params_grid['rand_layer_1'],
    'unit_layer_2': random_params_grid['rand_layer_2'],
    'activation_layer_1': random_params_grid['rand_activation_layer_1'],
    'activation_layer_2': random_params_grid['rand_activation_layer_2'],
    'activation_layer_3': random_params_grid['rand_activation_layer_3'],
    'unit_dropout_1': random_params_grid['rand_dropout_1'],
    'unit_dropout_2': random_params_grid['rand_dropout_2'],
    'is_batch_normalization_1': True,
    'is_batch_normalization_2': True,
    'unit_batch': random_params_grid['rand_batch']
  }
  return best_params
#
# ========================================================
# ========================================================
#                      RANDOM SEARCH
# ========================================================
# ========================================================
def print_model_hypertuning(model):
  """Print the model currently hyper-tuned."""
  if 'model_ffn3' in str(model):
    print('>> FNN3 hypertuning.')
  if 'model_ffn5' in str(model):
    print('>> FNN5 hypertuning.')
  if 'model_ffn10' in str(model):
    print('>> FNN10 hypertuning.')
  if 'model_cnn3' in str(model):
    print('>> CNN3 hypertuning.')
  if 'model_cnn5' in str(model):
    print('>> CNN5 hypertuning.')
  if 'model_cnn10' in str(model):
    print('>> CNN10 hypertuning.')
  if 'model_lstm3' in str(model):
    print('>> LSTM3 hypertuning.')
  if 'model_gru3' in str(model):
    print('>> GRU3 hypertuning.')
#
def random_search_for_tensorflow_models(
    enc_data,
    model,
    params_grid,
    n_iterations = 30,
    is_verbose_iterations = False):
  """Random search for tensorflow models."""
  # last_revised = 15-May-22
  for idata in range(3):
    print("current data set:", idata)
    data_test_size = define_data_test_size(idata)
    # added as of 26-apr-22
    if 'ffn' in str(model):
      x_boot, y_boot = bootstrapForHypertuning( enc_data[idata] )
      x_train, x_test, y_train, y_test = data_split_for_fnns(
        enc_data[idata],
        testsize = data_test_size
      )
    if 'cnn' in str(model):
      x_boot, y_boot = bootstrapForHypertuning(
        enc_data[idata], is_reshape = False)
      x_boot = x_boot.reshape(-1, 24, 7, 1)
      x_train, x_test, y_train, y_test = data_split_for_cnns(
        enc_data[idata],
        testsize = data_test_size
      )
    if 'lstm' in str(model) or 'gru' in str(model):
      x_boot, y_boot = bootstrapForHypertuning(
        enc_data[idata], is_reshape = False)
      # x_boot = x_boot.reshape(-1, 24, 7, 1)
      x_train, x_test, y_train, y_test = data_split_for_cnns(
        enc_data[idata],
        testsize = data_test_size
      )
    # end of 26-apr-22
    print('x_boot.shape:', x_boot.shape)
    print('x_test.shape:', x_test.shape)
    data_dict = {
      'x_boot': x_boot, 'y_boot': y_boot,
      'x_test': x_test, 'y_test': y_test
    }
    print_model_hypertuning(model)
    best_so_far = 0
    for i_iteration in range( n_iterations ):
      if is_verbose_iterations:
        print('iteration', i_iteration, 'out of', n_iterations)
      if 'model_ffn3' in str(model):
        dl_mdl, random_params_grid = find_params_for_fnn_3_layers(
          data_dict,
          params_grid
        )
      if 'model_ffn5' in str(model):
        dl_mdl, random_params_grid = find_params_for_fnn_5_layers(
          data_dict,
          params_grid
        )
      if 'model_ffn10' in str(model):
        dl_mdl, random_params_grid = find_params_for_fnn_10_layers(
          data_dict,
          params_grid
        )
      if 'model_cnn3' in str(model):
        dl_mdl, random_params_grid = find_params_for_cnn_3_layers(
          data_dict,
          params_grid
        )
      if 'model_cnn5' in str(model):
        dl_mdl, random_params_grid = find_params_for_cnn_5_layers(
          data_dict,
          params_grid
        )
      if 'model_cnn10' in str(model):
        dl_mdl, random_params_grid = find_params_for_cnn_10_layers(
          data_dict,
          params_grid
        )
      if 'model_lstm3' in str(model):
        dl_mdl, random_params_grid = find_params_for_lstm_3_layers(
          data_dict,
          params_grid
        )
      if 'model_gru3' in str(model):
        dl_mdl, random_params_grid = find_params_for_gru_3_layers(
          data_dict,
          params_grid
        )
      #
      fpr, tpr, thresholds = metrics.roc_curve(
        y_test,
        dl_mdl.predict(x_test)
      )
      roc_auc_iter = np.round(metrics.auc(fpr, tpr), 3)
      if roc_auc_iter > best_so_far:
        if 'model_ffn3' in str(model):
          best_params = save_params_for_fnn_3_layers(random_params_grid)
        if 'model_ffn5' in str(model):
          best_params = save_params_for_fnn_5_layers(random_params_grid)
        if 'model_ffn10' in str(model):
          best_params = save_params_for_fnn_10_layers(random_params_grid)
        if 'model_cnn3' in str(model):
          best_params = save_params_for_cnn_3_layers(random_params_grid)
        if 'model_cnn5' in str(model):
          best_params = save_params_for_cnn_5_layers(random_params_grid)
        if 'model_cnn10' in str(model):
          best_params = save_params_for_cnn_10_layers(random_params_grid)
        if 'model_lstm3' in str(model) or 'model_gru3' in str(model):
          best_params = save_params_for_lstm_3_gru_3_layers(random_params_grid)
        best_score = roc_auc_iter
        best_so_far = best_score
    print('\n\nbest score:', best_score)
    print('best_params =', best_params)
    print()
#
class dlModelsPipeline():
  def __init__(self, estimator, data_dict, params_grid):
    self.estimator = estimator
    self.data_dict = data_dict
    self.params_grid = params_grid
    self.ypred = np.zeros((len(self.data_dict['y_test'])))
    # to be determined if needed for tf models:
    # self.yscore = np.zeros((len(self.y_test), 2))
    self.fpr = None
    self.tpr = None
    self.thresholds = None
    self.roc_auc = 0.0
  # end of __init__
  #
  def modelTrain(self):
    """Training Function."""
    if 'model_ffn3' in str(self.estimator):
      print('> FFN3 training')
      self.trained_model = fnn_3_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_ffn5' in str(self.estimator):
      print('> FFN5 training')
      self.trained_model = fnn_5_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_ffn10' in str(self.estimator):
      print('> FFN10 training')
      self.trained_model = fnn_10_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_cnn3' in str(self.estimator):
      print('> CNN3 training')
      self.trained_model = cnn_3_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_cnn5' in str(self.estimator):
      print('> CNN5 training')
      self.trained_model = cnn_5_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_cnn10' in str(self.estimator):
      print('> CNN10 training')
      self.trained_model = cnn_10_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_lstm3' in str(self.estimator):
      print('> LSTM3 training')
      self.trained_model = lstm_3_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
    if 'model_gru3' in str(self.estimator):
      print('> GRU3 training')
      self.trained_model = gru_3_layers_training_pipeline(
        self.data_dict,
        self.params_grid
      )
  # end of modelTrain
  #
  def modelPredict(self):
    self.ypred = self.trained_model.predict(
      self.data_dict['x_test']
    )
    self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
      self.data_dict['y_test'],
      self.ypred
    )
    self.roc_auc = np.round(
      metrics.auc(
        self.fpr,
        self.tpr
      ),
      3
    )
  # end of modelPredict
# end of class