import numpy as np
import sklearn as sk
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
#plt.style.use("seaborn-deep")
#plt.style.use("seaborn-pastel")
#plt.style.use("seaborn-white")
plt.style.use("seaborn")
#plt.style.use("dark_background")
#plt.style.use("bmh")
plt.style.use("ggplot")
#plt.style.use("fivethirtyeight")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
np.random.seed(8040)

# A list of each method acrynom and a dictionary giving their long name
methods = ['flc','cos']
long_name = {'flc':'FL-CoSTCo','cos':'CoSTCo'}
metrics = {'MAE':mean_absolute_error,
           'R2':r2_score,
           'MAPE':mean_absolute_percentage_error,
           'MDAE':median_absolute_error,
           'MSE':mean_squared_error}

# A dictionary where we will store the predictions for each method
data = {}

for m in methods:

    # IMPORTANT: CHANGE long_name['cos'] to long_name[m] when fl-costco results are in
    
    # y_pred is a list of prediction made by the method
    # We ran each method 10 times, so we have ten sets of such predictions
    y_test_pred = np.load('./{}_results/y_test_pred.npy'.format(long_name['cos']))*5
     

    # y_test is the ground truth of y_pred
    y_test = np.load('./{}_results/y_test.npy'.format(long_name['cos']))*5
    
    # y_pred is a list of prediction made by the method
    # We ran each method 10 times, so we have ten sets of such predictions
    y_train_pred = np.load('./{}_results/y_train_pred.npy'.format(long_name['cos']))*5


    # y_test is the ground truth of y_pred
    y_train = np.load('./{}_results/y_train.npy'.format(long_name['cos']))*5


    # We store this data in the data dictionary
    data[m] = {'y_test_pred':y_test_pred,'y_test':y_test,'y_train':y_train,'y_train_pred':y_train_pred}


cos_mean = []
cos_top_std = []
cos_bot_std = []

flc_mean = []
flc_top_std = []
flc_bot_std = []



# We calculate the mean and std for each methods fit and predict time
for m in metrics:
    
    cos_met = metrics[m](data['cos']['y_test'].transpose(),data['cos']['y_test_pred'].transpose(),multioutput='raw_values')
    flc_met = metrics[m](data['flc']['y_test'].transpose(),data['flc']['y_test_pred'].transpose(),multioutput='raw_values')

    cos_mean.append(np.mean(cos_met))
    flc_mean.append(np.mean(flc_met))
    
    if m == 'R2':
      cos_mean[-1] = 1 - cos_mean[-1]    
      flc_mean[-1] = 1 - flc_mean[-1]    
    
    cos_top_std.append(np.std(cos_met))
    flc_top_std.append(np.std(flc_met))
    
    # If the mean - the standard deviation is negative
    # this would not look good when plotted
    # If this is the case, we replace it with zero
    if cos_mean[-1] - cos_top_std[-1] < 0:
        cos_bot_std.append(cos_mean[-1])
    else:
        cos_bot_std.append(cos_top_std[-1])
    
    if flc_mean[-1] - flc_top_std[-1] < 0:
        flc_bot_std.append(flc_mean[-1])
    else:
        flc_bot_std.append(flc_top_std[-1])

# Plot this data as a bar plot with the standard deviation shown as a black line
plt.bar([1,4,7,10,13], cos_mean, 1, yerr=(cos_bot_std,cos_top_std), color = ['lightcoral'])
plt.bar([2,5,8,11,14],flc_mean, 1, yerr=(flc_bot_std,flc_top_std), color = ['darkolivegreen'])
xticks = list(metrics.keys())
xticks[1] = '1 - R2'
plt.xticks([1.5,4.5,7.5,10.5,13.5],xticks)
colors = {'CoSTCo':'lightcoral', 'FL-CoSTCo':'darkolivegreen',}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.title('Train Performance')

# show plot
plt.show()



cos_mean = []
cos_top_std = []
cos_bot_std = []

flc_mean = []
flc_top_std = []
flc_bot_std = []



# We calculate the mean and std for each methods fit and predict time
for m in metrics:
    
    cos_met = metrics[m](data['cos']['y_train'].transpose(),data['cos']['y_train_pred'].transpose(),multioutput='raw_values')
    flc_met = metrics[m](data['flc']['y_train'].transpose(),data['flc']['y_train_pred'].transpose(),multioutput='raw_values')

    cos_mean.append(np.mean(cos_met))
    flc_mean.append(np.mean(flc_met))

    if m == 'R2':
      cos_mean[-1] = 1 - cos_mean[-1]    
      flc_mean[-1] = 1 - flc_mean[-1]    

    cos_top_std.append(np.std(cos_met))
    flc_top_std.append(np.std(flc_met))
    
    # If the mean - the standard deviation is negative
    # this would not look good when plotted
    # If this is the case, we replace it with zero
    if cos_mean[-1] - cos_top_std[-1] < 0:
        cos_bot_std.append(cos_mean[-1])
    else:
        cos_bot_std.append(cos_top_std[-1])
    
    if flc_mean[-1] - flc_top_std[-1] < 0:
        flc_bot_std.append(flc_mean[-1])
    else:
        flc_bot_std.append(flc_top_std[-1])

# Plot this data as a bar plot with the standard deviation shown as a black line
plt.bar([1,4,7,10,13], cos_mean, 1, yerr=(cos_bot_std,cos_top_std), color = ['lightcoral'])
plt.bar([2,5,8,11,14],flc_mean, 1, yerr=(flc_bot_std,flc_top_std), color = ['darkolivegreen'])
xticks = list(metrics.keys())
xticks[1] = '1 - R2'
plt.xticks([1.5,4.5,7.5,10.5,13.5],xticks)
colors = {'CoSTCo':'lightcoral', 'FL-CoSTCo':'darkolivegreen',}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.title('Test Performance')

# show plot
plt.show()
