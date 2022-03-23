import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import sklearn as sk
from costco import CoSTCo
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error

np.random.seed(8040)

splits = [{'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]},
          {'X':[],'y':[]}]

  

#for s in range(6): 
#  splits[s]['X'] = np.load('./data/X_{}.npy'.format(s))
#  splits[s]['y'] = np.load('./data/y_{}.npy'.format(s)) 
    

for s in range(6): 
  splits[s]['X'] = np.load('./movie_lens/splits/X_{}_v2.npy'.format(s))
  splits[s]['y'] = np.expand_dims(np.load('./movie_lens/splits/y_{}_v2.npy'.format(s)),axis=1) 

X_train = []
y_train = []
for s in range(5):
  X_train.append(splits[s]['X'])
  y_train.append(splits[s]['y'])
X_train = np.concatenate(X_train,axis=0)
y_train = np.concatenate(y_train,axis=0)

X_test = splits[5]['X']
y_test = splits[5]['y']
    
tensor_shape = (611,9724,8214)
rank = 0
filters = 3
reg = 0.0834

mae = []
r2 = []
mape = []
mdae = []
mse = []

y_test_pred_out = []
y_test_out= []
y_train_pred_out = []
y_train_out = []

for _ in range(10):

        # Make the model
        model = CoSTCo(rank = 2**rank,filters = 2**filters,shape = tensor_shape,reg =reg)

        magic = 64

        # Compile the CoSTCo
        opt = tf.keras.optimizers.Adam(learning_rate=3e-3)

        def tf_r2(y_true, y_pred):
          SS_res =  tf.reduce_sum(tf.math.square( y_true-y_pred )) 
          SS_tot = tf.reduce_sum(tf.math.square( y_true - tf.reduce_mean(y_true) ) ) 
          return ( 1 - SS_res/(SS_tot + tfk.backend.epsilon()) )

        model.compile(loss = 'mse',optimizer = opt,metrics = [tf_r2])
        early_stopping = tfk.callbacks.EarlyStopping(monitor='tf_r2',mode = 'max', 
                                                 patience=5,
                                                 min_delta = 0.001,
                                                 restore_best_weights = True)
    
        

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype('int32'),y_train))
        train_dataset = train_dataset.batch(32*magic)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


        # Train CoSTCo
        model.fit(train_dataset,
              epochs = 5000,
              verbose = 1,
              callbacks = [early_stopping])
         
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(train_dataset)


        # calculate metrics

        mae.append(mean_absolute_error(y_test,y_pred))
        r2.append(r2_score(y_test,y_pred))
        mape.append(mean_absolute_percentage_error(y_test,y_pred))
        mdae.append(median_absolute_error(y_test,y_pred))
        mse.append(mean_squared_error(y_test,y_pred))

        y_test_pred_out.append(y_pred)
        y_test_out.append(y_test)
        y_train_pred_out.append(y_train_pred)
        y_train_out.append(y_train)

y_test_pred_out = np.stack(y_test_pred_out,axis = 0)
y_test_out = np.stack(y_test_out,axis = 0)
y_train_pred_out = np.stack(y_train_pred_out,axis = 0)
y_train_out = np.stack(y_train_out,axis = 0)

print("R2: {}+-{}".format(np.mean(r2),np.std(r2))) 
print("MAE: {}+-{}".format(np.mean(mae),np.std(mae))) 
print("MAPE: {}+-{}".format(np.mean(mape),np.std(mape))) 
print("MSE: {}+-{}".format(np.mean(mse),np.std(mse))) 
print("MDAE: {}+-{}".format(np.mean(mdae),np.std(mdae))) 

np.save('CoSTCo_results/y_test_pred.npy',y_test_pred_out[...,0])
np.save('CoSTCo_results/y_test.npy',y_test_out[...,0])
np.save('CoSTCo_results/y_train_pred.npy',y_train_pred_out[...,0])
np.save('CoSTCo_results/y_train.npy',y_train_out[...,0])
model.save_weights('./CoSTCo_results/weights')

