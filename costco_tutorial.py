import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl


def generate_data(num_devices,num_websites,num_items):
  '''
  This function generates a dataset
  
  It takes a number of devices, number of websites and number of items.
  for each device it generates a matrix representing that user's reviews left on a set of items accross different websites
  these reviews are an integer between 0 and 5
  
  For each devices' matrix a 50% of the entries are missing at random.
  This is represented by a binary matrix

  It returns the review matrices as a list 
  It also returns the binary matrices as a list 

  NO NEED TO UNDERSTAND/READ THE HOW THE FUNCTION WORKS AS LONG AS YOU UNDERSTAND THE INPUT AND OUTPUT
  '''

  # A list containing the device data
  device_data = []

  # A list containing binary matrices indicating
  # which entryies in the device data is missing
  device_missing = []


  # Don't worry about this for now, it is just used to generate the data
  web_param = np.random.normal(size = (num_websites,16))
  item_param = np.random.normal(size = (num_items,16))

  for _ in range(num_devices):
    
    # Generate data for a device
    # The result is a matrix representing the reviews left by this user on
    # various websites for various items
    # don't worry too much about how the data is generated
    data = tf.einsum('ir,jr,r->ij',web_param,item_param,np.random.normal(size = (16)))
    
    # scale the data between 0,5 to represent review stars
    data -= np.min(data)
    data /= np.max(data)
    data *= 5
    data = np.round(data)

    # Generate a binary matrix
    # if the entry is 1, the value is missing from data
    # if the entry is 0, the value is observed
    missing = np.random.randint(0,2,size = (num_websites,num_items))

    # remove the missing values from the data
    data = data*(1 - missing)

    # append to the list of device data
    device_data.append(data)

    # append the missing value matrix to the list
    device_missing.append(missing)

  return device_data , device_missing

# The number of devices that have data 
NUM_DEVICES = 100

# The number of websites
NUM_WEBSITES = 20

# The number of items accross all websites
NUM_ITEMS = 2000


# Generate the data
device_data,device_missing = generate_data(NUM_DEVICES,NUM_WEBSITES,NUM_ITEMS)

# So now we have a list of matrices for each device
# these matrices all have missing values
# now we want to impute those missing values using tensor completion
# suppose all the devices send their data to a central server
# and that central server stacks the matrices into a tensor
review_tensor = np.stack(device_data,axis = 0)
missing_tensor = np.stack(device_missing,axis = 0)

# Now we have a tensor of size 100 X 20 X 2000
# that records the reviews that 100 users leave on the 2000
# items purchased from 20 different websites
# this is review_tensor

# We also have a binary tensor indicates when an entry of review tensor
# is known (missing_tensor)

# Now we want to use CoSTCo to fill in the missing values
# You can skip the definition of CoSTCo for now, just understand the inputs and outputs 
# once you understand this and the rest of the script
# come back and try to understand how CoSTCo works, you might want to read the paper as well
class CoSTCo(tfk.Model):
    def __init__(self, rank = 10, 
                       shape = (NUM_DEVICES,NUM_WEBSITES,NUM_ITEMS),
                       filters = 64):
        '''
        rank is an important hyperparamater for tensor factorization 
        a higher rank means the model is more powerful but more likely to overfit
        shape is just the shape of the input tensor
        filters is the number of conv filters
        '''

        super(CoSTCo, self).__init__()

        # C_n are the paramater matrices for each axis of the tensor
        # For example C_2 is a matrix of size NUM_WEBSITES by Rank
        # Ex. if you take the 10th row of C_2, it describes the 10th website
        # you can think of this as similar to the latent space of an autoencoder.
        C_init = tf.random_normal_initializer()
        self.C_1 = tf.Variable(
            initial_value=C_init(shape=(shape[0], rank), dtype="float32"),
            trainable=True,
        )
        self.C_2 = tf.Variable(
            initial_value=C_init(shape=(shape[1], rank), dtype="float32"),
            trainable=True,
        )
        self.C_3 = tf.Variable(
            initial_value=C_init(shape=(shape[2], rank), dtype="float32"),
            trainable=True,
        )

  
        # These are the layers that make CoSTCo a non-linear (deep) tensor completion method
        self.conv1 = tfkl.Conv2D(filters,kernel_size = (1,3),activation = 'relu')
        self.conv2 = tfkl.Conv2D(filters,kernel_size = (rank,1),activation = 'relu')
        self.dense1 = tfkl.Dense(filters,activation = 'relu')
        self.dense2 = tfkl.Dense(1,activation = 'sigmoid')


    def call(self,indices):
        '''
        indices is a three-tuple of indices into the tensor
        such as (1,2,3)

        We will return a number between 0-5 representing the prediction for that index
        of the tensor
        Ex. if indices is (1,2,3) this function will return the predicted review rating that the 1st device
        left on the 2nd website for the 3rd item
        '''

        # select the appropiate rows of the factor matrices
        # c_1,c_2 and c_3 will be sets of vectors with the length of rank
        c_1 = tf.gather(self.C_1,indices[...,0])
        c_2 = tf.gather(self.C_2,indices[...,1])
        c_3 = tf.gather(self.C_3,indices[...,2])

        # Because all the vectors are of the same size, we can stack them into a matrix
        # of size Rank X 3
        c = tf.stack([c_1,c_2,c_3],axis = -1)
  
        # Turn this into size Rank X 3 X 1, we will treat this last dimension as the filter dimension
        c = tf.expand_dims(c,axis = -1)

        # Apply the first convolution to this matrix, reducing it to size Rank X 1 X filters
        c = self.conv1(c)
        
        # Apply the first convolution to this matrix, reducing it to size 1 X 1 X filters
        c = self.conv2(c)
  
        # Flatten it to size filters
        c = tfkl.Flatten()(c)

        # Apply one dense layer, resulting in a size of filters
        c = self.dense1(c)

        # Apply a dense layer reducing it to a scalar between 0 and 5
        c = self.dense2(c)*5

        return c

# Make the model
model = CoSTCo(rank = 8,filters = 8)

# Create the data set
X = np.where(missing_tensor == 0)
y = review_tensor[X]
X = np.stack(X,axis = 1)

X_test = np.stack(np.where(missing_tensor == 1),axis = 1)

# X is a traditional feature matrix with three features
# each feature is actually an index into the tensor
# y is a traditional label vector, where the label is
# the value in the tensor corresponding to the indices in X
# X test is the indices of all the missing entrys

# Split the data into a train and val set
index = np.arange(np.shape(y)[0])
np.random.shuffle(index)
train_idx = index[:int(0.8*np.shape(y)[0])]
val_idx = index[int(0.8*np.shape(y)[0]):]

X_train = X[train_idx]
X_val = X[val_idx]
y_train = y[train_idx]
y_val = y[val_idx]


# Compile the CoSTCo
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss = 'mse',optimizer = opt)

# Train CoSTCo
model.fit(X_train,y_train,validation_data = (X_val,y_val),epochs = 100,batch_size = 640)

print("EVALUATE:")
model.evaluate(X_val,y_val,batch_size = 640)












