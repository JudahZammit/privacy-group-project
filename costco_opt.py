import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import optuna
from costco import CoSTCo
import sklearn as sk
from sklearn.metrics import r2_score
import tensorflow_addons as tfa

np.random.seed(8040)

splits = [{'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]}]


#TODO: subtract one from user id, check if timestamp is week or day
#      remove v2 tag, expand dim for y
  
#for s in range(6): 
#  splits[s]['X'] = np.load('./data/X_{}.npy'.format(s))
#  splits[s]['y'] = np.load('./data/y_{}.npy'.format(s)) 
for s in range(6): 
  splits[s]['X'] = np.load('./movie_lens/splits/X_{}_v2.npy'.format(s))
  splits[s]['y'] = np.expand_dims(np.load('./movie_lens/splits/y_{}_v2.npy'.format(s)),axis=1) 
  

def objective(trial):

  rank = trial.suggest_int('rank',0,7)
  filters = trial.suggest_int('filters',0,8)
  reg = trial.suggest_float('reg',0,1)
  epochs = 5000

  split_size = splits[0]['y'].shape[0]

  y_true = []  
  for s in range(5):
    y_true.append(splits[s]['y'])
  y_true = np.concatenate(y_true,axis=0)
  y_pred = np.zeros_like(y_true)


  for i in range(5):
    tfk.backend.clear_session()  
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for s in range(5):
      if s == i:
        X_val = splits[s]['X']
        y_val = splits[s]['y']
      else:
        X_train.append(splits[s]['X'])
        y_train.append(splits[s]['y'])
    X_train = np.concatenate(X_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)

    # We deffinintly have an extra in dim 1 and maybe 7* extra in dim 3
    tensor_shape = (611,9724,8214)

    # Make the model
    model = CoSTCo(rank =2**rank,filters = 2**filters,shape = tensor_shape,reg = reg)
    
    # Compile the CoSTCo
    magic = 64
    opt = tf.keras.optimizers.Adam(learning_rate=3e-3)
    early_stopping = tfk.callbacks.EarlyStopping(monitor='tf_r2',mode = 'max', 
                                                 patience=5,
                                                 min_delta = 0.001,
                                                 restore_best_weights = True)
    def tf_r2(y_true, y_pred):
      SS_res =  tf.reduce_sum(tf.math.square( y_true-y_pred )) 
      SS_tot = tf.reduce_sum(tf.math.square( y_true - tf.reduce_mean(y_true) ) ) 
      return ( 1 - SS_res/(SS_tot + tfk.backend.epsilon()) )


    model.compile(loss = 'mse',optimizer = opt,metrics = [tf_r2])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype('int32'),y_train))
    train_dataset = train_dataset.batch(32*magic)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.astype('int32'),y_val))
    val_dataset = val_dataset.batch(32*magic)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    # Train CoSTCo
    model.fit(train_dataset,validation_data = val_dataset,
              epochs = epochs,
              verbose = 1,
              callbacks = [early_stopping])
 
    y_pred[split_size*(i):split_size*(i+1)] = model.predict(X_val)
      
  r2 = r2_score(y_true,y_pred)

  return r2

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2000)
  


r2 = []
for r in reversed(range(0,9,1)):

  print()
  print()
  print()
  print()
  print(r)
  print()
  print()
  print()
  print()
  rank = 10
  filters = r
  reg = 0
  
  epochs = 5000

  split_size = splits[0]['y'].shape[0]

  y_true = []  
  for s in range(5):
    y_true.append(splits[s]['y'])
  y_true = np.concatenate(y_true,axis=0)
  y_pred = np.zeros_like(y_true)


  for i in range(1):
    tfk.backend.clear_session()  
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for s in range(5):
      if s == i:
        X_val = splits[s]['X']
        y_val = splits[s]['y']
      else:
        X_train.append(splits[s]['X'])
        y_train.append(splits[s]['y'])
    X_train = np.concatenate(X_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)

    tensor_shape = (611,9724,8214)

    # Make the model
    model = CoSTCo(rank = rank,filters = 2**filters,shape = tensor_shape,reg = reg)
    
    # Compile the CoSTCo
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    early_stopping = tfk.callbacks.EarlyStopping(monitor='tf_r2',mode = 'max',
                                                 patience=5,
                                                 min_delta = 0.0001,
                                                 restore_best_weights = True)

    
    def tf_r2(y_true, y_pred):
      SS_res =  tf.reduce_sum(tf.math.square( y_true-y_pred )) 
      SS_tot = tf.reduce_sum(tf.math.square( y_true - tf.reduce_mean(y_true) ) ) 
      return ( 1 - SS_res/(SS_tot + tfk.backend.epsilon()) )

    model.compile(loss = 'mse',optimizer = opt,metrics = [tf_r2])

    # Train CoSTCo
    model.fit(X_train,y_train,validation_data = (X_val,y_val),epochs = epochs,batch_size = 32,verbose = 1,callbacks = [early_stopping])
 
    y_pred[split_size*(i):split_size*(i+1)] = model(X_val)
      
  r2.append(r2_score(y_true,y_pred))
