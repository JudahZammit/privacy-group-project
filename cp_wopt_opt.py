import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import optuna
from costco import CP_WOPT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


np.random.seed(8040)

splits = [{'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]}]

  
for s in range(6): 
  splits[s]['X'] = np.load('X_{}.npy'.format(s))
  splits[s]['y'] = np.load('y_{}.npy'.format(s)) 


def objective(trial):

  rank = trial.suggest_int('rank',1,100)
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

    tensor_shape = (80,206,2)

    # Make the model
    model = CP_WOPT(rank = rank,shape = tensor_shape)

    pre_c1 = model.C_1(np.array(range(80))).numpy()  
    pre_c2 = model.C_2(np.array(range(206))).numpy()
    pre_c3 = model.C_3(np.array(range(2))).numpy()
 
    # Compile the CoSTCo
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    early_stopping = tfk.callbacks.EarlyStopping(monitor='loss', 
                                                 patience=20,
                                                 min_delta = 0.001,
                                                 restore_best_weights = True)


    model.compile(loss = 'mse',optimizer = opt,metrics = ['mae'])

    # Train CoSTCo
    model.fit(X_train,y_train,validation_data = (X_val,y_val),
              epochs = epochs,
              batch_size = y_train.size,verbose = 1,callbacks = [early_stopping])
    
    post_c1 = model.C_1(np.array(range(80))).numpy()
    post_c2 = model.C_2(np.array(range(206))).numpy()
    post_c3 = model.C_3(np.array(range(2))).numpy()
 
    assert not np.all(post_c1 == pre_c1)
    assert not np.all(post_c2 == pre_c2)
    assert not np.all(post_c3 == pre_c3)

    y_pred[split_size*(i):split_size*(i+1)] = model(X_val)
      
  mae = np.mean(np.abs(y_pred-y_true))


  return mae

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)


