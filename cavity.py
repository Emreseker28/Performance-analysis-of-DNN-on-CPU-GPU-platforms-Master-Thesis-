#%%
# multivariate multi-headed 1d cnn example
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
'''
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
'''
#import data
#to-do: find a way to get rid of repeating lines
data1 = pd.read_csv('./cavity1.0/0.1/p')
data2 = pd.read_csv('./cavity1.0/0.2/p')
data3 = pd.read_csv('./cavity1.0/0.3/p')
data4 = pd.read_csv('./cavity1.0/0.4/p')
data5 = pd.read_csv('./cavity1.0/0.5/p')
data6 = pd.read_csv('./cavity2.0/0.1/p')
data7 = pd.read_csv('./cavity2.0/0.2/p')
data8 = pd.read_csv('./cavity2.0/0.3/p')
data9 = pd.read_csv('./cavity2.0/0.4/p')
data10 = pd.read_csv('./cavity2.0/0.5/p')
data11 = pd.read_csv('./cavity4.0/0.1/p')
data12 = pd.read_csv('./cavity4.0/0.2/p')
data13 = pd.read_csv('./cavity4.0/0.3/p')
data14 = pd.read_csv('./cavity4.0/0.4/p')
data15 = pd.read_csv('./cavity4.0/0.5/p')
data16 = pd.read_csv('./cavity5.0/0.1/p')
data17 = pd.read_csv('./cavity5.0/0.2/p')
data18 = pd.read_csv('./cavity5.0/0.3/p')
data19 = pd.read_csv('./cavity5.0/0.4/p')
data20 = pd.read_csv('./cavity5.0/0.5/p')

test_data1 = pd.read_csv('./cavity3.0/0.1/p')
test_data2 = pd.read_csv('./cavity3.0/0.2/p')
test_data3 = pd.read_csv('./cavity3.0/0.3/p')
test_data4 = pd.read_csv('./cavity3.0/0.4/p')
test_data5 = pd.read_csv('./cavity3.0/0.5/p')
#add new data with some noise added to it.
#print(data1)
#print(data2)

#take only the values from the files
in_seq1 = data1.iloc[19:419,:].values
in_seq2 = data2.iloc[19:419,:].values
in_seq3 = data3.iloc[19:419,:].values
in_seq4 = data4.iloc[19:419,:].values
in_seq5 = data6.iloc[19:419,:].values
in_seq6 = data7.iloc[19:419,:].values
in_seq7 = data8.iloc[19:419,:].values
in_seq8 = data9.iloc[19:419,:].values
in_seq9 = data11.iloc[19:419,:].values
in_seq10 = data12.iloc[19:419,:].values
in_seq11 = data13.iloc[19:419,:].values
in_seq12 = data14.iloc[19:419,:].values
in_seq13 = data16.iloc[19:419,:].values
in_seq14 = data17.iloc[19:419,:].values
in_seq15 = data18.iloc[19:419,:].values
in_seq16 = data19.iloc[19:419,:].values

test_seq1 = test_data1.iloc[19:419,:].values
test_seq2 = test_data2.iloc[19:419,:].values
test_seq3 = test_data3.iloc[19:419,:].values
test_seq4 = test_data4.iloc[19:419,:].values
test_seq5 = test_data5.iloc[19:419,:].values
#convert values to float type.
in_seq1 = in_seq1.astype(float)
in_seq2 = in_seq2.astype(float)
in_seq3 = in_seq3.astype(float)
in_seq4 = in_seq4.astype(float)
in_seq5 = in_seq5.astype(float)
in_seq6 = in_seq6.astype(float)
in_seq7 = in_seq7.astype(float)
in_seq8 = in_seq8.astype(float)
in_seq9 = in_seq9.astype(float)
in_seq10 = in_seq10.astype(float)
in_seq11 = in_seq11.astype(float)
in_seq12 = in_seq12.astype(float)
in_seq13 = in_seq13.astype(float)
in_seq14 = in_seq14.astype(float)
in_seq15 = in_seq15.astype(float)
in_seq16 = in_seq16.astype(float)

test_seq1 = test_seq1.astype(float)
test_seq2 = test_seq2.astype(float)
test_seq3 = test_seq3.astype(float)
test_seq4 = test_seq4.astype(float)
test_seq5 = test_seq5.astype(float)
# define output sequence
out_seq1 = data5.iloc[19:419,:].values
out_seq2 = data10.iloc[19:419,:].values
out_seq3 = data15.iloc[19:419,:].values
out_seq4 = data20.iloc[19:419,:].values
out_seq1 = out_seq1.astype(float)
out_seq2 = out_seq2.astype(float)
out_seq3 = out_seq3.astype(float)
out_seq4 = out_seq4.astype(float)
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape(1, (len(in_seq1)))
in_seq2 = in_seq2.reshape(1, (len(in_seq2)))
in_seq3 = in_seq3.reshape(1, (len(in_seq3)))
in_seq4 = in_seq4.reshape(1, (len(in_seq4)))
in_seq5 = in_seq5.reshape(1, (len(in_seq5)))
in_seq6 = in_seq6.reshape(1, (len(in_seq6)))
in_seq7 = in_seq7.reshape(1, (len(in_seq7)))
in_seq8 = in_seq8.reshape(1, (len(in_seq8)))
in_seq9 = in_seq9.reshape(1, (len(in_seq9)))
in_seq10 = in_seq10.reshape(1, (len(in_seq10)))
in_seq11 = in_seq11.reshape(1, (len(in_seq11)))
in_seq12 = in_seq12.reshape(1, (len(in_seq12)))
in_seq13 = in_seq13.reshape(1, (len(in_seq13)))
in_seq14 = in_seq14.reshape(1, (len(in_seq14)))
in_seq15 = in_seq15.reshape(1, (len(in_seq15)))
in_seq16 = in_seq16.reshape(1, (len(in_seq16)))
out_seq1 = out_seq1.reshape(1, (len(out_seq1)))
out_seq2 = out_seq2.reshape(1, (len(out_seq2)))
out_seq3 = out_seq3.reshape(1, (len(out_seq3)))
out_seq4 = out_seq4.reshape(1, (len(out_seq4)))

test_seq1 = test_seq1.reshape(1, (len(test_seq1)))
test_seq2 = test_seq2.reshape(1, (len(test_seq2)))
test_seq3 = test_seq3.reshape(1, (len(test_seq3)))
test_seq4 = test_seq4.reshape(1, (len(test_seq4)))
test_seq5 = test_seq5.reshape(1, (len(test_seq5)))
print('in_seq1.shape', in_seq1.shape)
print('in_seq2.shape', in_seq2.shape)
print('in_seq3.shape', in_seq3.shape)
print('out_seq.shape', out_seq1.shape)


# horizontally stack columns
dataset = np.concatenate((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5,
						  in_seq6, in_seq7, in_seq8, in_seq9, in_seq10,
						  in_seq11, in_seq12, in_seq13, in_seq14, in_seq15,
						  in_seq16,
						  out_seq1, out_seq2, out_seq3, out_seq4), axis=0)

test_dataset = np.concatenate((test_seq1, test_seq2, test_seq3,
							   test_seq4, test_seq5), axis=0)
print('dataset.shape', dataset.shape)
print('test_dataset.shape', test_dataset.shape)

# choose a number of time steps
n_steps = 2
# convert into input/output
#X, y = split_sequences(dataset, n_steps)

X = []
y = []
test = []
samples = dataset.shape[0]-n_steps
test_samples = test_dataset.shape[0]-n_steps
for i in range(samples):
	X.append( dataset[i:i+n_steps] )
	y.append( dataset[-1] )
	
for j in range(test_samples):
	test.append(test_dataset[j:j+n_steps])

X = np.array(X)
n_features = X.shape[2]

y = np.array(y).reshape( samples, 1, n_features )
test = np.array(test)
test = test.reshape(-1, 2, n_features)

print('X.shape', X.shape)
print('y.shape', y.shape)
print('n_features', n_features)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#move the model to a function
# first input model
#start from low number of filters, kernels and then start to increase
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=1, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=n_steps)(cnn1)
#cnn1 = Flatten()(cnn1)
# second input model
#visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=1, activation='relu')(cnn1)
#cnn2 = MaxPooling1D(pool_size=2)(cnn2)
#flt = Flatten()(cnn2)
# merge input models
#merge = Concatenate(axis=1)([cnn1, cnn2])
dense = Dense(64, activation='relu')(cnn2)
output = Dense(n_features)(dense)
model = Model(inputs=visible1, outputs=output)
model.compile(optimizer='adam', loss='mse')

print(model.summary())

#tensorflow profiler
#python profiler
#start of the measuring
# fit model
history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))
#end of the measuring


# demonstrate prediction
#take test data from cavity3.0


yhat = model.predict(test, verbose=1)
print(yhat)

#Testing the accuracy of the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Reshaping the test data to fit the output of the program.
test_flattened = test.reshape(-1, test.shape[-1]) # ndarray (6,400)
yhat_flattened = yhat.reshape(-1, yhat.shape[-1]) # ndarray (3, 400)
mse = mean_squared_error(test_flattened[:3, :], yhat_flattened) #taking the first 3 rows to match the sizes
mae = mean_absolute_error(test_flattened[:3, :], yhat_flattened)
print('MSE: ', mse)
print('MAE: ', mae)


#plotting the figures
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

epochs = range(1, 101)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# %%
