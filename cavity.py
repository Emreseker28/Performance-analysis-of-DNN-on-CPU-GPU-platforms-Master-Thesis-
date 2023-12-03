#%%
# Emre Sekeroglu Master Thesis
import pandas as pd
import numpy as np
import cProfile
import re
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#constants
EPOCHS = 100
FILTER_SIZE = 64
DENSE_SIZE = 64

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

def machine_model(n_steps, n_features, filter_size, dense_size):
	# first input model
	#start from low number of filters, kernels and then start to increase
	visible1 = Input(shape=(n_steps, n_features))
	cnn1 = Conv1D(filters=filter_size, kernel_size=1, activation='relu')(visible1)
	cnn1 = MaxPooling1D(pool_size=n_steps)(cnn1)
	#cnn1 = Flatten()(cnn1)
	# second input model
	#visible2 = Input(shape=(n_steps, n_features))
	cnn2 = Conv1D(filters=filter_size, kernel_size=1, activation='relu')(cnn1)
	#cnn2 = MaxPooling1D(pool_size=2)(cnn2)
	#flt = Flatten()(cnn2)
	# merge input models
	#merge = Concatenate(axis=1)([cnn1, cnn2])
	dense = Dense(dense_size, activation='relu')(cnn2)
	output = Dense(n_features)(dense)
	model = Model(inputs=visible1, outputs=output)
	model.compile(optimizer='adam', loss='mse')
	return model

#import data
#to-do: find a way to get rid of repeating lines
#1.0
data1 = pd.read_csv('./cavity1.0/0.1/p')
data2 = pd.read_csv('./cavity1.0/0.2/p')
data3 = pd.read_csv('./cavity1.0/0.3/p')
data4 = pd.read_csv('./cavity1.0/0.4/p')
data5 = pd.read_csv('./cavity1.0/0.5/p')
#1.2
data6 = pd.read_csv('./cavity1.2/0.1/p')
data7 = pd.read_csv('./cavity1.2/0.2/p')
data8 = pd.read_csv('./cavity1.2/0.3/p')
data9 = pd.read_csv('./cavity1.2/0.4/p')
data10 = pd.read_csv('./cavity1.2/0.5/p')
#1.4
data11 = pd.read_csv('./cavity1.4/0.1/p')
data12 = pd.read_csv('./cavity1.4/0.2/p')
data13 = pd.read_csv('./cavity1.4/0.3/p')
data14 = pd.read_csv('./cavity1.4/0.4/p')
data15 = pd.read_csv('./cavity1.4/0.5/p')
#1.6
data16 = pd.read_csv('./cavity1.6/0.1/p')
data17 = pd.read_csv('./cavity1.6/0.2/p')
data18 = pd.read_csv('./cavity1.6/0.3/p')
data19 = pd.read_csv('./cavity1.6/0.4/p')
data20 = pd.read_csv('./cavity1.6/0.5/p')
#1.8
data21 = pd.read_csv('./cavity1.8/0.1/p')
data22 = pd.read_csv('./cavity1.8/0.2/p')
data23 = pd.read_csv('./cavity1.8/0.3/p')
data24 = pd.read_csv('./cavity1.8/0.4/p')
data25 = pd.read_csv('./cavity1.8/0.5/p')
#2.0
data26 = pd.read_csv('./cavity2.0/0.1/p')
data27 = pd.read_csv('./cavity2.0/0.2/p')
data28 = pd.read_csv('./cavity2.0/0.3/p')
data29 = pd.read_csv('./cavity2.0/0.4/p')
data30 = pd.read_csv('./cavity2.0/0.5/p')
#2.2
data31 = pd.read_csv('./cavity2.2/0.1/p')
data32 = pd.read_csv('./cavity2.2/0.2/p')
data33 = pd.read_csv('./cavity2.2/0.3/p')
data34 = pd.read_csv('./cavity2.2/0.4/p')
data35 = pd.read_csv('./cavity2.2/0.5/p')
#2.4
data36 = pd.read_csv('./cavity2.4/0.1/p')
data37 = pd.read_csv('./cavity2.4/0.2/p')
data38 = pd.read_csv('./cavity2.4/0.3/p')
data39 = pd.read_csv('./cavity2.4/0.4/p')
data40 = pd.read_csv('./cavity2.4/0.5/p')
#2.6
data41 = pd.read_csv('./cavity2.6/0.1/p')
data42 = pd.read_csv('./cavity2.6/0.2/p')
data43 = pd.read_csv('./cavity2.6/0.3/p')
data44 = pd.read_csv('./cavity2.6/0.4/p')
data45 = pd.read_csv('./cavity2.6/0.5/p')
#2.8
data46 = pd.read_csv('./cavity2.8/0.1/p')
data47 = pd.read_csv('./cavity2.8/0.2/p')
data48 = pd.read_csv('./cavity2.8/0.3/p')
data49 = pd.read_csv('./cavity2.8/0.4/p')
data50 = pd.read_csv('./cavity2.8/0.5/p')
#4.0
data51 = pd.read_csv('./cavity4.0/0.1/p')
data52 = pd.read_csv('./cavity4.0/0.2/p')
data53 = pd.read_csv('./cavity4.0/0.3/p')
data54 = pd.read_csv('./cavity4.0/0.4/p')
data55 = pd.read_csv('./cavity4.0/0.5/p')
#4.2
data56 = pd.read_csv('./cavity4.2/0.1/p')
data57 = pd.read_csv('./cavity4.2/0.2/p')
data58 = pd.read_csv('./cavity4.2/0.3/p')
data59 = pd.read_csv('./cavity4.2/0.4/p')
data60 = pd.read_csv('./cavity4.2/0.5/p')
#4.4
data61 = pd.read_csv('./cavity4.4/0.1/p')
data62 = pd.read_csv('./cavity4.4/0.2/p')
data63 = pd.read_csv('./cavity4.4/0.3/p')
data64 = pd.read_csv('./cavity4.4/0.4/p')
data65 = pd.read_csv('./cavity4.4/0.5/p')
#4.6
data66 = pd.read_csv('./cavity4.6/0.1/p')
data67 = pd.read_csv('./cavity4.6/0.2/p')
data68 = pd.read_csv('./cavity4.6/0.3/p')
data69 = pd.read_csv('./cavity4.6/0.4/p')
data70 = pd.read_csv('./cavity4.6/0.5/p')
#4.8
data71 = pd.read_csv('./cavity4.8/0.1/p')
data72 = pd.read_csv('./cavity4.8/0.2/p')
data73 = pd.read_csv('./cavity4.8/0.3/p')
data74 = pd.read_csv('./cavity4.8/0.4/p')
data75 = pd.read_csv('./cavity4.8/0.5/p')
#5.0
data76 = pd.read_csv('./cavity5.0/0.1/p')
data77 = pd.read_csv('./cavity5.0/0.2/p')
data78 = pd.read_csv('./cavity5.0/0.3/p')
data79 = pd.read_csv('./cavity5.0/0.4/p')
data80 = pd.read_csv('./cavity5.0/0.5/p')

#test data
#3.0
test_data1 = pd.read_csv('./cavity3.0/0.1/p')
test_data2 = pd.read_csv('./cavity3.0/0.2/p')
test_data3 = pd.read_csv('./cavity3.0/0.3/p')
test_data4 = pd.read_csv('./cavity3.0/0.4/p')
test_data5 = pd.read_csv('./cavity3.0/0.5/p')
#3.2
test_data6 = pd.read_csv('./cavity3.2/0.1/p')
test_data7 = pd.read_csv('./cavity3.2/0.2/p')
test_data8 = pd.read_csv('./cavity3.2/0.3/p')
test_data9 = pd.read_csv('./cavity3.2/0.4/p')
test_data10 = pd.read_csv('./cavity3.2/0.5/p')
#3.4
test_data11 = pd.read_csv('./cavity3.4/0.1/p')
test_data12 = pd.read_csv('./cavity3.4/0.2/p')
test_data13 = pd.read_csv('./cavity3.4/0.3/p')
test_data14 = pd.read_csv('./cavity3.4/0.4/p')
test_data15 = pd.read_csv('./cavity3.4/0.5/p')
#3.5
test_data16 = pd.read_csv('./cavity3.5/0.1/p')
test_data17 = pd.read_csv('./cavity3.5/0.2/p')
test_data18 = pd.read_csv('./cavity3.5/0.3/p')
test_data19 = pd.read_csv('./cavity3.5/0.4/p')
test_data20 = pd.read_csv('./cavity3.5/0.5/p')
#3.6
test_data21 = pd.read_csv('./cavity3.6/0.1/p')
test_data22 = pd.read_csv('./cavity3.6/0.2/p')
test_data23 = pd.read_csv('./cavity3.6/0.3/p')
test_data24 = pd.read_csv('./cavity3.6/0.4/p')
test_data25 = pd.read_csv('./cavity3.6/0.5/p')
#3.7
test_data26 = pd.read_csv('./cavity3.7/0.1/p')
test_data27 = pd.read_csv('./cavity3.7/0.2/p')
test_data28 = pd.read_csv('./cavity3.7/0.3/p')
test_data29 = pd.read_csv('./cavity3.7/0.4/p')
test_data30 = pd.read_csv('./cavity3.7/0.5/p')
#3.8
test_data31 = pd.read_csv('./cavity3.8/0.1/p')
test_data32 = pd.read_csv('./cavity3.8/0.2/p')
test_data33 = pd.read_csv('./cavity3.8/0.3/p')
test_data34 = pd.read_csv('./cavity3.8/0.4/p')
test_data35 = pd.read_csv('./cavity3.8/0.5/p')
#3.9
test_data36 = pd.read_csv('./cavity3.9/0.1/p')
test_data37 = pd.read_csv('./cavity3.9/0.2/p')
test_data38 = pd.read_csv('./cavity3.9/0.3/p')
test_data39 = pd.read_csv('./cavity3.9/0.4/p')
test_data40 = pd.read_csv('./cavity3.9/0.5/p')
#add new data with some noise added to it.
#print(data1)
#print(data2)

#take only the values from the files
in_seq1 = data1.iloc[19:419,:].values
in_seq2 = data2.iloc[19:419,:].values
in_seq3 = data3.iloc[19:419,:].values
in_seq4 = data4.iloc[19:419,:].values
in_seq5 = data5.iloc[19:419,:].values
in_seq6 = data6.iloc[19:419,:].values
in_seq7 = data7.iloc[19:419,:].values
in_seq8 = data8.iloc[19:419,:].values
in_seq9 = data9.iloc[19:419,:].values
in_seq10 = data10.iloc[19:419,:].values
in_seq11 = data11.iloc[19:419,:].values
in_seq12 = data12.iloc[19:419,:].values
in_seq13 = data13.iloc[19:419,:].values
in_seq14 = data14.iloc[19:419,:].values
in_seq15 = data15.iloc[19:419,:].values
in_seq16 = data16.iloc[19:419,:].values
in_seq17 = data17.iloc[19:419,:].values
in_seq18 = data18.iloc[19:419,:].values
in_seq19 = data19.iloc[19:419,:].values
in_seq20 = data20.iloc[19:419,:].values
in_seq21 = data21.iloc[19:419,:].values
in_seq22 = data22.iloc[19:419,:].values
in_seq23 = data23.iloc[19:419,:].values
in_seq24 = data24.iloc[19:419,:].values
in_seq25 = data25.iloc[19:419,:].values
in_seq26 = data26.iloc[19:419,:].values
in_seq27 = data27.iloc[19:419,:].values
in_seq28 = data28.iloc[19:419,:].values
in_seq29 = data29.iloc[19:419,:].values
in_seq30 = data30.iloc[19:419,:].values
in_seq31 = data31.iloc[19:419,:].values
in_seq32 = data32.iloc[19:419,:].values
in_seq33 = data33.iloc[19:419,:].values
in_seq34 = data34.iloc[19:419,:].values
in_seq35 = data35.iloc[19:419,:].values
in_seq36 = data36.iloc[19:419,:].values
in_seq37 = data37.iloc[19:419,:].values
in_seq38 = data38.iloc[19:419,:].values
in_seq39 = data39.iloc[19:419,:].values
in_seq40 = data40.iloc[19:419,:].values
in_seq41 = data41.iloc[19:419,:].values
in_seq42 = data42.iloc[19:419,:].values
in_seq43 = data43.iloc[19:419,:].values
in_seq44 = data44.iloc[19:419,:].values
in_seq45 = data45.iloc[19:419,:].values
in_seq46 = data46.iloc[19:419,:].values
in_seq47 = data47.iloc[19:419,:].values
in_seq48 = data48.iloc[19:419,:].values
in_seq49 = data49.iloc[19:419,:].values
in_seq50 = data50.iloc[19:419,:].values
in_seq51 = data51.iloc[19:419,:].values
in_seq52 = data52.iloc[19:419,:].values
in_seq53 = data53.iloc[19:419,:].values
in_seq54 = data54.iloc[19:419,:].values
in_seq55 = data55.iloc[19:419,:].values
in_seq56 = data56.iloc[19:419,:].values
in_seq57 = data57.iloc[19:419,:].values
in_seq58 = data58.iloc[19:419,:].values
in_seq59 = data59.iloc[19:419,:].values
in_seq60 = data60.iloc[19:419,:].values
in_seq61 = data61.iloc[19:419,:].values
in_seq62 = data62.iloc[19:419,:].values
in_seq63 = data63.iloc[19:419,:].values
in_seq64 = data64.iloc[19:419,:].values
in_seq65 = data65.iloc[19:419,:].values
in_seq66 = data66.iloc[19:419,:].values
in_seq67 = data67.iloc[19:419,:].values
in_seq68 = data68.iloc[19:419,:].values
in_seq69 = data69.iloc[19:419,:].values
in_seq70 = data70.iloc[19:419,:].values
in_seq71 = data71.iloc[19:419,:].values
in_seq72 = data72.iloc[19:419,:].values
in_seq73 = data73.iloc[19:419,:].values
in_seq74 = data74.iloc[19:419,:].values
in_seq75 = data75.iloc[19:419,:].values
in_seq76 = data76.iloc[19:419,:].values
in_seq77 = data77.iloc[19:419,:].values
in_seq78 = data78.iloc[19:419,:].values
in_seq79 = data79.iloc[19:419,:].values
in_seq80 = data80.iloc[19:419,:].values


test_seq1 = test_data1.iloc[19:419,:].values
test_seq2 = test_data2.iloc[19:419,:].values
test_seq3 = test_data3.iloc[19:419,:].values
test_seq4 = test_data4.iloc[19:419,:].values
test_seq5 = test_data5.iloc[19:419,:].values
test_seq6 = test_data6.iloc[19:419,:].values
test_seq7 = test_data7.iloc[19:419,:].values
test_seq8 = test_data8.iloc[19:419,:].values
test_seq9 = test_data9.iloc[19:419,:].values
test_seq10 = test_data10.iloc[19:419,:].values
test_seq11 = test_data11.iloc[19:419,:].values
test_seq12 = test_data12.iloc[19:419,:].values
test_seq13 = test_data13.iloc[19:419,:].values
test_seq14 = test_data14.iloc[19:419,:].values
test_seq15 = test_data15.iloc[19:419,:].values
test_seq16 = test_data16.iloc[19:419,:].values
test_seq17 = test_data17.iloc[19:419,:].values
test_seq18 = test_data18.iloc[19:419,:].values
test_seq19 = test_data19.iloc[19:419,:].values
test_seq20 = test_data20.iloc[19:419,:].values
test_seq21 = test_data21.iloc[19:419,:].values
test_seq22 = test_data22.iloc[19:419,:].values
test_seq23 = test_data23.iloc[19:419,:].values
test_seq24 = test_data24.iloc[19:419,:].values
test_seq25 = test_data25.iloc[19:419,:].values
test_seq26 = test_data26.iloc[19:419,:].values
test_seq27 = test_data27.iloc[19:419,:].values
test_seq28 = test_data28.iloc[19:419,:].values
test_seq29 = test_data29.iloc[19:419,:].values
test_seq30 = test_data30.iloc[19:419,:].values
test_seq31 = test_data31.iloc[19:419,:].values
test_seq32 = test_data32.iloc[19:419,:].values
test_seq33 = test_data33.iloc[19:419,:].values
test_seq34 = test_data34.iloc[19:419,:].values
test_seq35 = test_data35.iloc[19:419,:].values
test_seq36 = test_data36.iloc[19:419,:].values
test_seq37 = test_data37.iloc[19:419,:].values
test_seq38 = test_data38.iloc[19:419,:].values
test_seq39 = test_data39.iloc[19:419,:].values
test_seq40 = test_data40.iloc[19:419,:].values
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
in_seq17 = in_seq17.astype(float)
in_seq18 = in_seq18.astype(float)
in_seq19 = in_seq19.astype(float)
in_seq20 = in_seq20.astype(float)
in_seq21 = in_seq21.astype(float)
in_seq22 = in_seq22.astype(float)
in_seq23 = in_seq23.astype(float)
in_seq24 = in_seq24.astype(float)
in_seq25 = in_seq25.astype(float)
in_seq26 = in_seq26.astype(float)
in_seq27 = in_seq27.astype(float)
in_seq28 = in_seq28.astype(float)
in_seq29 = in_seq29.astype(float)
in_seq30 = in_seq30.astype(float)
in_seq31 = in_seq31.astype(float)
in_seq32 = in_seq32.astype(float)
in_seq33 = in_seq33.astype(float)
in_seq34 = in_seq34.astype(float)
in_seq35 = in_seq35.astype(float)
in_seq36 = in_seq36.astype(float)
in_seq37 = in_seq37.astype(float)
in_seq38 = in_seq38.astype(float)
in_seq39 = in_seq39.astype(float)
in_seq40 = in_seq40.astype(float)
in_seq41 = in_seq41.astype(float)
in_seq42 = in_seq42.astype(float)
in_seq43 = in_seq43.astype(float)
in_seq44 = in_seq44.astype(float)
in_seq45 = in_seq45.astype(float)
in_seq46 = in_seq46.astype(float)
in_seq47 = in_seq47.astype(float)
in_seq48 = in_seq48.astype(float)
in_seq49 = in_seq49.astype(float)
in_seq50 = in_seq50.astype(float)
in_seq51 = in_seq51.astype(float)
in_seq52 = in_seq52.astype(float)
in_seq53 = in_seq53.astype(float)
in_seq54 = in_seq54.astype(float)
in_seq55 = in_seq55.astype(float)
in_seq56 = in_seq56.astype(float)
in_seq57 = in_seq57.astype(float)
in_seq58 = in_seq58.astype(float)
in_seq59 = in_seq59.astype(float)
in_seq60 = in_seq60.astype(float)
in_seq61 = in_seq61.astype(float)
in_seq62 = in_seq62.astype(float)
in_seq63 = in_seq63.astype(float)
in_seq64 = in_seq64.astype(float)
in_seq65 = in_seq65.astype(float)
in_seq66 = in_seq66.astype(float)
in_seq67 = in_seq67.astype(float)
in_seq68 = in_seq68.astype(float)
in_seq69 = in_seq69.astype(float)
in_seq70 = in_seq70.astype(float)
in_seq71 = in_seq71.astype(float)
in_seq72 = in_seq72.astype(float)
in_seq73 = in_seq73.astype(float)
in_seq74 = in_seq74.astype(float)
in_seq75 = in_seq75.astype(float)
in_seq76 = in_seq76.astype(float)
in_seq77 = in_seq77.astype(float)
in_seq78 = in_seq78.astype(float)
in_seq79 = in_seq79.astype(float)
in_seq80 = in_seq80.astype(float)

test_seq1 = test_seq1.astype(float)
test_seq2 = test_seq2.astype(float)
test_seq3 = test_seq3.astype(float)
test_seq4 = test_seq4.astype(float)
test_seq5 = test_seq5.astype(float)
test_seq6 = test_seq6.astype(float)
test_seq7 = test_seq7.astype(float)
test_seq8 = test_seq8.astype(float)
test_seq9 = test_seq9.astype(float)
test_seq10 = test_seq10.astype(float)
test_seq11 = test_seq11.astype(float)
test_seq12 = test_seq12.astype(float)
test_seq13 = test_seq13.astype(float)
test_seq14 = test_seq14.astype(float)
test_seq15 = test_seq15.astype(float)
test_seq16 = test_seq16.astype(float)
test_seq17 = test_seq17.astype(float)
test_seq18 = test_seq18.astype(float)
test_seq19 = test_seq19.astype(float)
test_seq20 = test_seq20.astype(float)
test_seq21 = test_seq21.astype(float)
test_seq22 = test_seq22.astype(float)
test_seq23 = test_seq23.astype(float)
test_seq24 = test_seq24.astype(float)
test_seq25 = test_seq25.astype(float)
test_seq26 = test_seq26.astype(float)
test_seq27 = test_seq27.astype(float)
test_seq28 = test_seq28.astype(float)
test_seq29 = test_seq29.astype(float)
test_seq30 = test_seq30.astype(float)
test_seq31 = test_seq31.astype(float)
test_seq32 = test_seq32.astype(float)
test_seq33 = test_seq33.astype(float)
test_seq34 = test_seq34.astype(float)
test_seq35 = test_seq35.astype(float)
test_seq36 = test_seq36.astype(float)
test_seq37 = test_seq37.astype(float)
test_seq38 = test_seq38.astype(float)
test_seq39 = test_seq39.astype(float)
test_seq40 = test_seq40.astype(float)
# define output sequence
out_seq1 = in_seq5
out_seq2 = in_seq10
out_seq3 = in_seq15
out_seq4 = in_seq20
out_seq5 = in_seq25
out_seq6 = in_seq30
out_seq7 = in_seq35
out_seq8 = in_seq40
out_seq9 = in_seq45
out_seq10 = in_seq50
out_seq11 = in_seq55
out_seq12 = in_seq60
out_seq13 = in_seq65
out_seq14 = in_seq70
out_seq15 = in_seq75
out_seq16 = in_seq80
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
in_seq17 = in_seq17.reshape(1, (len(in_seq17)))
in_seq18 = in_seq18.reshape(1, (len(in_seq18)))
in_seq19 = in_seq19.reshape(1, (len(in_seq19)))
in_seq20 = in_seq20.reshape(1, (len(in_seq20)))
in_seq21 = in_seq21.reshape(1, (len(in_seq21)))
in_seq22 = in_seq22.reshape(1, (len(in_seq22)))
in_seq23 = in_seq23.reshape(1, (len(in_seq23)))
in_seq24 = in_seq24.reshape(1, (len(in_seq24)))
in_seq25 = in_seq25.reshape(1, (len(in_seq25)))
in_seq26 = in_seq26.reshape(1, (len(in_seq26)))
in_seq27 = in_seq27.reshape(1, (len(in_seq27)))
in_seq28 = in_seq28.reshape(1, (len(in_seq28)))
in_seq29 = in_seq29.reshape(1, (len(in_seq29)))
in_seq30 = in_seq30.reshape(1, (len(in_seq30)))
in_seq31 = in_seq31.reshape(1, (len(in_seq31)))
in_seq32 = in_seq32.reshape(1, (len(in_seq32)))
in_seq33 = in_seq33.reshape(1, (len(in_seq33)))
in_seq34 = in_seq34.reshape(1, (len(in_seq34)))
in_seq35 = in_seq35.reshape(1, (len(in_seq35)))
in_seq36 = in_seq36.reshape(1, (len(in_seq36)))
in_seq37 = in_seq37.reshape(1, (len(in_seq37)))
in_seq38 = in_seq38.reshape(1, (len(in_seq38)))
in_seq39 = in_seq39.reshape(1, (len(in_seq39)))
in_seq40 = in_seq40.reshape(1, (len(in_seq40)))
in_seq41 = in_seq41.reshape(1, (len(in_seq41)))
in_seq42 = in_seq42.reshape(1, (len(in_seq42)))
in_seq43 = in_seq43.reshape(1, (len(in_seq43)))
in_seq44 = in_seq44.reshape(1, (len(in_seq44)))
in_seq45 = in_seq45.reshape(1, (len(in_seq45)))
in_seq46 = in_seq46.reshape(1, (len(in_seq46)))
in_seq47 = in_seq47.reshape(1, (len(in_seq47)))
in_seq48 = in_seq48.reshape(1, (len(in_seq48)))
in_seq49 = in_seq49.reshape(1, (len(in_seq49)))
in_seq50 = in_seq50.reshape(1, (len(in_seq50)))
in_seq51 = in_seq51.reshape(1, (len(in_seq51)))
in_seq52 = in_seq52.reshape(1, (len(in_seq52)))
in_seq53 = in_seq53.reshape(1, (len(in_seq53)))
in_seq54 = in_seq54.reshape(1, (len(in_seq54)))
in_seq55 = in_seq55.reshape(1, (len(in_seq55)))
in_seq56 = in_seq56.reshape(1, (len(in_seq56)))
in_seq57 = in_seq57.reshape(1, (len(in_seq57)))
in_seq58 = in_seq58.reshape(1, (len(in_seq58)))
in_seq59 = in_seq59.reshape(1, (len(in_seq59)))
in_seq60 = in_seq60.reshape(1, (len(in_seq60)))
in_seq61 = in_seq61.reshape(1, (len(in_seq61)))
in_seq62 = in_seq62.reshape(1, (len(in_seq62)))
in_seq63 = in_seq63.reshape(1, (len(in_seq63)))
in_seq64 = in_seq64.reshape(1, (len(in_seq64)))
in_seq65 = in_seq65.reshape(1, (len(in_seq65)))
in_seq66 = in_seq66.reshape(1, (len(in_seq66)))
in_seq67 = in_seq67.reshape(1, (len(in_seq67)))
in_seq68 = in_seq68.reshape(1, (len(in_seq68)))
in_seq69 = in_seq69.reshape(1, (len(in_seq69)))
in_seq70 = in_seq70.reshape(1, (len(in_seq70)))
in_seq71 = in_seq71.reshape(1, (len(in_seq71)))
in_seq72 = in_seq72.reshape(1, (len(in_seq72)))
in_seq73 = in_seq73.reshape(1, (len(in_seq73)))
in_seq74 = in_seq74.reshape(1, (len(in_seq74)))
in_seq75 = in_seq75.reshape(1, (len(in_seq75)))
in_seq76 = in_seq76.reshape(1, (len(in_seq76)))
in_seq77 = in_seq77.reshape(1, (len(in_seq77)))
in_seq78 = in_seq78.reshape(1, (len(in_seq78)))
in_seq79 = in_seq79.reshape(1, (len(in_seq79)))
in_seq80 = in_seq80.reshape(1, (len(in_seq80)))
out_seq1 = out_seq1.reshape(1, (len(out_seq1)))
out_seq2 = out_seq2.reshape(1, (len(out_seq2)))
out_seq3 = out_seq3.reshape(1, (len(out_seq3)))
out_seq4 = out_seq4.reshape(1, (len(out_seq4)))
out_seq5 = out_seq5.reshape(1, (len(out_seq5)))
out_seq6 = out_seq6.reshape(1, (len(out_seq6)))
out_seq7 = out_seq7.reshape(1, (len(out_seq7)))
out_seq8 = out_seq8.reshape(1, (len(out_seq8)))
out_seq9 = out_seq9.reshape(1, (len(out_seq9)))
out_seq10 = out_seq10.reshape(1, (len(out_seq10)))
out_seq11 = out_seq11.reshape(1, (len(out_seq11)))
out_seq12 = out_seq12.reshape(1, (len(out_seq12)))
out_seq13 = out_seq13.reshape(1, (len(out_seq13)))
out_seq14 = out_seq14.reshape(1, (len(out_seq14)))
out_seq15 = out_seq15.reshape(1, (len(out_seq15)))
out_seq16 = out_seq16.reshape(1, (len(out_seq16)))

test_seq1 = test_seq1.reshape(1, (len(test_seq1)))
test_seq2 = test_seq2.reshape(1, (len(test_seq2)))
test_seq3 = test_seq3.reshape(1, (len(test_seq3)))
test_seq4 = test_seq4.reshape(1, (len(test_seq4)))
test_seq5 = test_seq5.reshape(1, (len(test_seq5)))
test_seq6 = test_seq6.reshape(1, (len(test_seq6)))
test_seq7 = test_seq7.reshape(1, (len(test_seq7)))
test_seq8 = test_seq8.reshape(1, (len(test_seq8)))
test_seq9 = test_seq9.reshape(1, (len(test_seq9)))
test_seq10 = test_seq10.reshape(1, (len(test_seq10)))
test_seq11 = test_seq11.reshape(1, (len(test_seq11)))
test_seq12 = test_seq12.reshape(1, (len(test_seq12)))
test_seq13 = test_seq13.reshape(1, (len(test_seq13)))
test_seq14 = test_seq14.reshape(1, (len(test_seq14)))
test_seq15 = test_seq15.reshape(1, (len(test_seq15)))
test_seq16 = test_seq16.reshape(1, (len(test_seq16)))
test_seq17 = test_seq17.reshape(1, (len(test_seq17)))
test_seq18 = test_seq18.reshape(1, (len(test_seq18)))
test_seq19 = test_seq19.reshape(1, (len(test_seq19)))
test_seq20 = test_seq20.reshape(1, (len(test_seq20)))
test_seq21 = test_seq21.reshape(1, (len(test_seq21)))
test_seq22 = test_seq22.reshape(1, (len(test_seq22)))
test_seq23 = test_seq23.reshape(1, (len(test_seq23)))
test_seq24 = test_seq24.reshape(1, (len(test_seq24)))
test_seq25 = test_seq25.reshape(1, (len(test_seq25)))
test_seq26 = test_seq26.reshape(1, (len(test_seq26)))
test_seq27 = test_seq27.reshape(1, (len(test_seq27)))
test_seq28 = test_seq28.reshape(1, (len(test_seq28)))
test_seq29 = test_seq29.reshape(1, (len(test_seq29)))
test_seq30 = test_seq30.reshape(1, (len(test_seq30)))
test_seq31 = test_seq31.reshape(1, (len(test_seq31)))
test_seq32 = test_seq32.reshape(1, (len(test_seq32)))
test_seq33 = test_seq33.reshape(1, (len(test_seq33)))
test_seq34 = test_seq34.reshape(1, (len(test_seq34)))
test_seq35 = test_seq35.reshape(1, (len(test_seq35)))
test_seq36 = test_seq36.reshape(1, (len(test_seq36)))
test_seq37 = test_seq37.reshape(1, (len(test_seq37)))
test_seq38 = test_seq38.reshape(1, (len(test_seq38)))
test_seq39 = test_seq39.reshape(1, (len(test_seq39)))
test_seq40 = test_seq40.reshape(1, (len(test_seq40)))



# horizontally stack columns
dataset = np.concatenate((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5,
						  in_seq6, in_seq7, in_seq8, in_seq9, in_seq10,
						  in_seq11, in_seq12, in_seq13, in_seq14, in_seq15,
						  in_seq16, in_seq17, in_seq18, in_seq19, in_seq20,
						  in_seq21, in_seq22, in_seq23, in_seq24, in_seq25,
						  in_seq26, in_seq27, in_seq28, in_seq29, in_seq30,
						  in_seq31, in_seq32, in_seq33, in_seq34, in_seq35,
						  in_seq36, in_seq37, in_seq38, in_seq39, in_seq40,
						  in_seq41, in_seq42, in_seq43, in_seq44, in_seq45,
						  in_seq46, in_seq47, in_seq48, in_seq49, in_seq50,
						  in_seq51, in_seq52, in_seq53, in_seq54, in_seq55,
						  in_seq56, in_seq57, in_seq58, in_seq59, in_seq60,
						  in_seq61, in_seq62, in_seq63, in_seq64, in_seq65,
						  in_seq66, in_seq67, in_seq68, in_seq69, in_seq70,
						  in_seq71, in_seq72, in_seq73, in_seq74, in_seq75,
						  in_seq76, in_seq77, in_seq78, in_seq79, in_seq80,
						  out_seq1, out_seq2, out_seq3, out_seq4, out_seq5,
						  out_seq1, out_seq2, out_seq3, out_seq4, out_seq5,
						  out_seq1, out_seq2, out_seq3, out_seq4, out_seq5,
						  out_seq1, out_seq14, out_seq15, out_seq16
						  ), axis=0)

test_dataset = np.concatenate((test_seq1, test_seq2, test_seq3, test_seq4, test_seq5,
							   test_seq6, test_seq7, test_seq8, test_seq9, test_seq10,
							   test_seq11, test_seq12, test_seq13, test_seq14, test_seq15,
							   test_seq16, test_seq17, test_seq18, test_seq19, test_seq20,
							   test_seq21, test_seq22, test_seq23, test_seq24, test_seq25,
							   test_seq26, test_seq27, test_seq28, test_seq29, test_seq30,
							   test_seq31, test_seq32, test_seq33, test_seq34, test_seq35,
							   test_seq36, test_seq37, test_seq38, test_seq40, test_seq40,
							   ), axis=0)
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


model = machine_model(n_steps=n_steps, n_features=n_features, 
					  filter_size=FILTER_SIZE, dense_size=DENSE_SIZE)

print(model.summary())

#start of the measuring
profiler = cProfile.Profile()
profiler.enable()
# fit model
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
profiler.disable()
profiler.print_stats(sort='cumulative')
profiler.dump_stats('profiler_results.prof')
#end of the measuring


# demonstrate prediction
#take test data from cavity3.0


yhat = model.predict(test, verbose=1)
print(yhat)

#Testing the accuracy of the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Reshaping the test data to fit the output of the program.
test_flattened = test.reshape(-1, test.shape[-1]) # ndarray (76,400)
yhat_flattened = yhat.reshape(-1, yhat.shape[-1]) # ndarray (38, 400)
mse = mean_squared_error(test_flattened[:38, :], yhat_flattened) #taking the first 3 rows to match the sizes
mae = mean_absolute_error(test_flattened[:38, :], yhat_flattened)
print('MSE: ', mse)
print('MAE: ', mae)


#plotting the figures
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

epochs_range = range(1, EPOCHS+1)

plt.plot(epochs_range, loss, 'r', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#pearson correlation
'''
from keras import backend as K
def pearson_corr(y_true, y_pred):
	mx = K.mean(K.constant(y_true), axis=0)
	my = K.mean(K.constant(y_pred), axis=0)
	xm, ym = K.constant(y_true) - mx, K.constant(y_pred) - my
	r_num = K.sum(xm * ym)
	y_true_square_sum = K.sum(xm * xm)
	y_pred_square_sum = K.sum(ym * ym)
	r_den = K.sqrt(y_true_square_sum * y_pred_square_sum)
	r = r_num / r_den
	return K.get_value(K.mean(r))

print("Pearson Correlation: ", pearson_corr(test_flattened[:3, :], yhat_flattened))
'''
from scipy.stats import pearsonr
coef_P, _ = pearsonr(test_flattened[:38, :].flatten(), yhat_flattened.flatten())
print("Pearson Correlation: ", coef_P)

#Spearman correlation
from scipy.stats import spearmanr
coef, p = spearmanr(test_flattened[:38, :], yhat_flattened)
flat_coef, _ = spearmanr(test_flattened[:38, :].flatten(), yhat_flattened.flatten())
print("Spearman Correlation: ", flat_coef)

#Coefficient of determination R
#from sklearn.metrics import r2_score
#R_square = r2_score(test_flattened[:3, :], yhat_flattened)
R_square = coef_P * coef_P
print("Coefficient of Determination: ", R_square)

import pstats
stats = pstats.Stats('profiler_results.prof')
stats.sort_stats('cumulative').print_stats()
# %%
