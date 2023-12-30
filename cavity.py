#%%
# Emre Sekeroglu Master Thesis
import pandas as pd
import numpy as np
import cProfile
import tracemalloc
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



#import data
#to-do: find a way to get rid of repeating lines
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

dataU1 = pd.read_csv('./cavity1.0/0.1/U')
dataU2 = pd.read_csv('./cavity1.0/0.2/U')
dataU3 = pd.read_csv('./cavity1.0/0.3/U')
dataU4 = pd.read_csv('./cavity1.0/0.4/U')
dataU5 = pd.read_csv('./cavity1.0/0.5/U')
#1.2
dataU6 = pd.read_csv('./cavity1.2/0.1/U')
dataU7 = pd.read_csv('./cavity1.2/0.2/U')
dataU8 = pd.read_csv('./cavity1.2/0.3/U')
dataU9 = pd.read_csv('./cavity1.2/0.4/U')
dataU10 = pd.read_csv('./cavity1.2/0.5/U')
#1.4
dataU11 = pd.read_csv('./cavity1.4/0.1/U')
dataU12 = pd.read_csv('./cavity1.4/0.2/U')
dataU13 = pd.read_csv('./cavity1.4/0.3/U')
dataU14 = pd.read_csv('./cavity1.4/0.4/U')
dataU15 = pd.read_csv('./cavity1.4/0.5/U')
#1.6
dataU16 = pd.read_csv('./cavity1.6/0.1/U')
dataU17 = pd.read_csv('./cavity1.6/0.2/U')
dataU18 = pd.read_csv('./cavity1.6/0.3/U')
dataU19 = pd.read_csv('./cavity1.6/0.4/U')
dataU20 = pd.read_csv('./cavity1.6/0.5/U')
#1.8
dataU21 = pd.read_csv('./cavity1.8/0.1/U')
dataU22 = pd.read_csv('./cavity1.8/0.2/U')
dataU23 = pd.read_csv('./cavity1.8/0.3/U')
dataU24 = pd.read_csv('./cavity1.8/0.4/U')
dataU25 = pd.read_csv('./cavity1.8/0.5/U')
#2.0
dataU26 = pd.read_csv('./cavity2.0/0.1/U')
dataU27 = pd.read_csv('./cavity2.0/0.2/U')
dataU28 = pd.read_csv('./cavity2.0/0.3/U')
dataU29 = pd.read_csv('./cavity2.0/0.4/U')
dataU30 = pd.read_csv('./cavity2.0/0.5/U')
#2.2
dataU31 = pd.read_csv('./cavity2.2/0.1/U')
dataU32 = pd.read_csv('./cavity2.2/0.2/U')
dataU33 = pd.read_csv('./cavity2.2/0.3/U')
dataU34 = pd.read_csv('./cavity2.2/0.4/U')
dataU35 = pd.read_csv('./cavity2.2/0.5/U')
#2.4
dataU36 = pd.read_csv('./cavity2.4/0.1/U')
dataU37 = pd.read_csv('./cavity2.4/0.2/U')
dataU38 = pd.read_csv('./cavity2.4/0.3/U')
dataU39 = pd.read_csv('./cavity2.4/0.4/U')
dataU40 = pd.read_csv('./cavity2.4/0.5/U')
#2.6
dataU41 = pd.read_csv('./cavity2.6/0.1/U')
dataU42 = pd.read_csv('./cavity2.6/0.2/U')
dataU43 = pd.read_csv('./cavity2.6/0.3/U')
dataU44 = pd.read_csv('./cavity2.6/0.4/U')
dataU45 = pd.read_csv('./cavity2.6/0.5/U')
#2.8
dataU46 = pd.read_csv('./cavity2.8/0.1/U')
dataU47 = pd.read_csv('./cavity2.8/0.2/U')
dataU48 = pd.read_csv('./cavity2.8/0.3/U')
dataU49 = pd.read_csv('./cavity2.8/0.4/U')
dataU50 = pd.read_csv('./cavity2.8/0.5/U')
#4.0
dataU51 = pd.read_csv('./cavity4.0/0.1/U')
dataU52 = pd.read_csv('./cavity4.0/0.2/U')
dataU53 = pd.read_csv('./cavity4.0/0.3/U')
dataU54 = pd.read_csv('./cavity4.0/0.4/U')
dataU55 = pd.read_csv('./cavity4.0/0.5/U')
#4.2
dataU56 = pd.read_csv('./cavity4.2/0.1/U')
dataU57 = pd.read_csv('./cavity4.2/0.2/U')
dataU58 = pd.read_csv('./cavity4.2/0.3/U')
dataU59 = pd.read_csv('./cavity4.2/0.4/U')
dataU60 = pd.read_csv('./cavity4.2/0.5/U')
#4.4
dataU61 = pd.read_csv('./cavity4.4/0.1/U')
dataU62 = pd.read_csv('./cavity4.4/0.2/U')
dataU63 = pd.read_csv('./cavity4.4/0.3/U')
dataU64 = pd.read_csv('./cavity4.4/0.4/U')
dataU65 = pd.read_csv('./cavity4.4/0.5/U')
#4.6
dataU66 = pd.read_csv('./cavity4.6/0.1/U')
dataU67 = pd.read_csv('./cavity4.6/0.2/U')
dataU68 = pd.read_csv('./cavity4.6/0.3/U')
dataU69 = pd.read_csv('./cavity4.6/0.4/U')
dataU70 = pd.read_csv('./cavity4.6/0.5/U')
#4.8
dataU71 = pd.read_csv('./cavity4.8/0.1/U')
dataU72 = pd.read_csv('./cavity4.8/0.2/U')
dataU73 = pd.read_csv('./cavity4.8/0.3/U')
dataU74 = pd.read_csv('./cavity4.8/0.4/U')
dataU75 = pd.read_csv('./cavity4.8/0.5/U')
#5.0
dataU76 = pd.read_csv('./cavity5.0/0.1/U')
dataU77 = pd.read_csv('./cavity5.0/0.2/U')
dataU78 = pd.read_csv('./cavity5.0/0.3/U')
dataU79 = pd.read_csv('./cavity5.0/0.4/U')
dataU80 = pd.read_csv('./cavity5.0/0.5/U')

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
#3.0
test_dataU1 = pd.read_csv('./cavity3.0/0.1/U')
test_dataU2 = pd.read_csv('./cavity3.0/0.2/U')
test_dataU3 = pd.read_csv('./cavity3.0/0.3/U')
test_dataU4 = pd.read_csv('./cavity3.0/0.4/U')
test_dataU5 = pd.read_csv('./cavity3.0/0.5/U')
#3.2
test_dataU6 = pd.read_csv('./cavity3.2/0.1/U')
test_dataU7 = pd.read_csv('./cavity3.2/0.2/U')
test_dataU8 = pd.read_csv('./cavity3.2/0.3/U')
test_dataU9 = pd.read_csv('./cavity3.2/0.4/U')
test_dataU10 = pd.read_csv('./cavity3.2/0.5/U')
#3.4
test_dataU11 = pd.read_csv('./cavity3.4/0.1/U')
test_dataU12 = pd.read_csv('./cavity3.4/0.2/U')
test_dataU13 = pd.read_csv('./cavity3.4/0.3/U')
test_dataU14 = pd.read_csv('./cavity3.4/0.4/U')
test_dataU15 = pd.read_csv('./cavity3.4/0.5/U')
#3.5
test_dataU16 = pd.read_csv('./cavity3.5/0.1/U')
test_dataU17 = pd.read_csv('./cavity3.5/0.2/U')
test_dataU18 = pd.read_csv('./cavity3.5/0.3/U')
test_dataU19 = pd.read_csv('./cavity3.5/0.4/U')
test_dataU20 = pd.read_csv('./cavity3.5/0.5/U')
#3.6
test_dataU21 = pd.read_csv('./cavity3.6/0.1/U')
test_dataU22 = pd.read_csv('./cavity3.6/0.2/U')
test_dataU23 = pd.read_csv('./cavity3.6/0.3/U')
test_dataU24 = pd.read_csv('./cavity3.6/0.4/U')
test_dataU25 = pd.read_csv('./cavity3.6/0.5/U')
#3.7
test_dataU26 = pd.read_csv('./cavity3.7/0.1/U')
test_dataU27 = pd.read_csv('./cavity3.7/0.2/U')
test_dataU28 = pd.read_csv('./cavity3.7/0.3/U')
test_dataU29 = pd.read_csv('./cavity3.7/0.4/U')
test_dataU30 = pd.read_csv('./cavity3.7/0.5/U')
#3.8
test_dataU31 = pd.read_csv('./cavity3.8/0.1/U')
test_dataU32 = pd.read_csv('./cavity3.8/0.2/U')
test_dataU33 = pd.read_csv('./cavity3.8/0.3/U')
test_dataU34 = pd.read_csv('./cavity3.8/0.4/U')
test_dataU35 = pd.read_csv('./cavity3.8/0.5/U')
#3.9
test_dataU36 = pd.read_csv('./cavity3.9/0.1/U')
test_dataU37 = pd.read_csv('./cavity3.9/0.2/U')
test_dataU38 = pd.read_csv('./cavity3.9/0.3/U')
test_dataU39 = pd.read_csv('./cavity3.9/0.4/U')
test_dataU40 = pd.read_csv('./cavity3.9/0.5/U')
#add new data with some noise added to it.

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
in_seqU1 = np.array([list(map(float, row.strip('()').split())) for row in dataU1.iloc[19:419, 0]], dtype=float)
in_seqU2 = np.array([list(map(float, row.strip('()').split())) for row in dataU2.iloc[19:419, 0]], dtype=float)
in_seqU3 = np.array([list(map(float, row.strip('()').split())) for row in dataU3.iloc[19:419, 0]], dtype=float)
in_seqU4 = np.array([list(map(float, row.strip('()').split())) for row in dataU4.iloc[19:419, 0]], dtype=float)
in_seqU5 = np.array([list(map(float, row.strip('()').split())) for row in dataU5.iloc[19:419, 0]], dtype=float)
in_seqU6 = np.array([list(map(float, row.strip('()').split())) for row in dataU6.iloc[19:419, 0]], dtype=float)
in_seqU7 = np.array([list(map(float, row.strip('()').split())) for row in dataU7.iloc[19:419, 0]], dtype=float)
in_seqU8 = np.array([list(map(float, row.strip('()').split())) for row in dataU8.iloc[19:419, 0]], dtype=float)
in_seqU9 = np.array([list(map(float, row.strip('()').split())) for row in dataU9.iloc[19:419, 0]], dtype=float)
in_seqU10 = np.array([list(map(float, row.strip('()').split())) for row in dataU10.iloc[19:419, 0]], dtype=float)
in_seqU11 = np.array([list(map(float, row.strip('()').split())) for row in dataU11.iloc[19:419, 0]], dtype=float)
in_seqU12 = np.array([list(map(float, row.strip('()').split())) for row in dataU12.iloc[19:419, 0]], dtype=float)
in_seqU13 = np.array([list(map(float, row.strip('()').split())) for row in dataU13.iloc[19:419, 0]], dtype=float)
in_seqU14 = np.array([list(map(float, row.strip('()').split())) for row in dataU14.iloc[19:419, 0]], dtype=float)
in_seqU15 = np.array([list(map(float, row.strip('()').split())) for row in dataU15.iloc[19:419, 0]], dtype=float)
in_seqU16 = np.array([list(map(float, row.strip('()').split())) for row in dataU16.iloc[19:419, 0]], dtype=float)
in_seqU17 = np.array([list(map(float, row.strip('()').split())) for row in dataU17.iloc[19:419, 0]], dtype=float)
in_seqU18 = np.array([list(map(float, row.strip('()').split())) for row in dataU18.iloc[19:419, 0]], dtype=float)
in_seqU19 = np.array([list(map(float, row.strip('()').split())) for row in dataU19.iloc[19:419, 0]], dtype=float)
in_seqU20 = np.array([list(map(float, row.strip('()').split())) for row in dataU20.iloc[19:419, 0]], dtype=float)
in_seqU21 = np.array([list(map(float, row.strip('()').split())) for row in dataU21.iloc[19:419, 0]], dtype=float)
in_seqU22 = np.array([list(map(float, row.strip('()').split())) for row in dataU22.iloc[19:419, 0]], dtype=float)
in_seqU23 = np.array([list(map(float, row.strip('()').split())) for row in dataU23.iloc[19:419, 0]], dtype=float)
in_seqU24 = np.array([list(map(float, row.strip('()').split())) for row in dataU24.iloc[19:419, 0]], dtype=float)
in_seqU25 = np.array([list(map(float, row.strip('()').split())) for row in dataU25.iloc[19:419, 0]], dtype=float)
in_seqU26 = np.array([list(map(float, row.strip('()').split())) for row in dataU26.iloc[19:419, 0]], dtype=float)
in_seqU27 = np.array([list(map(float, row.strip('()').split())) for row in dataU27.iloc[19:419, 0]], dtype=float)
in_seqU28 = np.array([list(map(float, row.strip('()').split())) for row in dataU28.iloc[19:419, 0]], dtype=float)
in_seqU29 = np.array([list(map(float, row.strip('()').split())) for row in dataU29.iloc[19:419, 0]], dtype=float)
in_seqU30 = np.array([list(map(float, row.strip('()').split())) for row in dataU30.iloc[19:419, 0]], dtype=float)
in_seqU31 = np.array([list(map(float, row.strip('()').split())) for row in dataU31.iloc[19:419, 0]], dtype=float)
in_seqU32 = np.array([list(map(float, row.strip('()').split())) for row in dataU32.iloc[19:419, 0]], dtype=float)
in_seqU33 = np.array([list(map(float, row.strip('()').split())) for row in dataU33.iloc[19:419, 0]], dtype=float)
in_seqU34 = np.array([list(map(float, row.strip('()').split())) for row in dataU34.iloc[19:419, 0]], dtype=float)
in_seqU35 = np.array([list(map(float, row.strip('()').split())) for row in dataU35.iloc[19:419, 0]], dtype=float)
in_seqU36 = np.array([list(map(float, row.strip('()').split())) for row in dataU36.iloc[19:419, 0]], dtype=float)
in_seqU37 = np.array([list(map(float, row.strip('()').split())) for row in dataU37.iloc[19:419, 0]], dtype=float)
in_seqU38 = np.array([list(map(float, row.strip('()').split())) for row in dataU38.iloc[19:419, 0]], dtype=float)
in_seqU39 = np.array([list(map(float, row.strip('()').split())) for row in dataU39.iloc[19:419, 0]], dtype=float)
in_seqU40 = np.array([list(map(float, row.strip('()').split())) for row in dataU40.iloc[19:419, 0]], dtype=float)
in_seqU41 = np.array([list(map(float, row.strip('()').split())) for row in dataU41.iloc[19:419, 0]], dtype=float)
in_seqU42 = np.array([list(map(float, row.strip('()').split())) for row in dataU42.iloc[19:419, 0]], dtype=float)
in_seqU43 = np.array([list(map(float, row.strip('()').split())) for row in dataU43.iloc[19:419, 0]], dtype=float)
in_seqU44 = np.array([list(map(float, row.strip('()').split())) for row in dataU44.iloc[19:419, 0]], dtype=float)
in_seqU45 = np.array([list(map(float, row.strip('()').split())) for row in dataU45.iloc[19:419, 0]], dtype=float)
in_seqU46 = np.array([list(map(float, row.strip('()').split())) for row in dataU46.iloc[19:419, 0]], dtype=float)
in_seqU47 = np.array([list(map(float, row.strip('()').split())) for row in dataU47.iloc[19:419, 0]], dtype=float)
in_seqU48 = np.array([list(map(float, row.strip('()').split())) for row in dataU48.iloc[19:419, 0]], dtype=float)
in_seqU49 = np.array([list(map(float, row.strip('()').split())) for row in dataU49.iloc[19:419, 0]], dtype=float)
in_seqU50 = np.array([list(map(float, row.strip('()').split())) for row in dataU50.iloc[19:419, 0]], dtype=float)
in_seqU51 = np.array([list(map(float, row.strip('()').split())) for row in dataU51.iloc[19:419, 0]], dtype=float)
in_seqU52 = np.array([list(map(float, row.strip('()').split())) for row in dataU52.iloc[19:419, 0]], dtype=float)
in_seqU53 = np.array([list(map(float, row.strip('()').split())) for row in dataU53.iloc[19:419, 0]], dtype=float)
in_seqU54 = np.array([list(map(float, row.strip('()').split())) for row in dataU54.iloc[19:419, 0]], dtype=float)
in_seqU55 = np.array([list(map(float, row.strip('()').split())) for row in dataU55.iloc[19:419, 0]], dtype=float)
in_seqU56 = np.array([list(map(float, row.strip('()').split())) for row in dataU56.iloc[19:419, 0]], dtype=float)
in_seqU57 = np.array([list(map(float, row.strip('()').split())) for row in dataU57.iloc[19:419, 0]], dtype=float)
in_seqU58 = np.array([list(map(float, row.strip('()').split())) for row in dataU58.iloc[19:419, 0]], dtype=float)
in_seqU59 = np.array([list(map(float, row.strip('()').split())) for row in dataU59.iloc[19:419, 0]], dtype=float)
in_seqU60 = np.array([list(map(float, row.strip('()').split())) for row in dataU60.iloc[19:419, 0]], dtype=float)
in_seqU61 = np.array([list(map(float, row.strip('()').split())) for row in dataU61.iloc[19:419, 0]], dtype=float)
in_seqU62 = np.array([list(map(float, row.strip('()').split())) for row in dataU62.iloc[19:419, 0]], dtype=float)
in_seqU63 = np.array([list(map(float, row.strip('()').split())) for row in dataU63.iloc[19:419, 0]], dtype=float)
in_seqU64 = np.array([list(map(float, row.strip('()').split())) for row in dataU64.iloc[19:419, 0]], dtype=float)
in_seqU65 = np.array([list(map(float, row.strip('()').split())) for row in dataU65.iloc[19:419, 0]], dtype=float)
in_seqU66 = np.array([list(map(float, row.strip('()').split())) for row in dataU66.iloc[19:419, 0]], dtype=float)
in_seqU67 = np.array([list(map(float, row.strip('()').split())) for row in dataU67.iloc[19:419, 0]], dtype=float)
in_seqU68 = np.array([list(map(float, row.strip('()').split())) for row in dataU68.iloc[19:419, 0]], dtype=float)
in_seqU69 = np.array([list(map(float, row.strip('()').split())) for row in dataU69.iloc[19:419, 0]], dtype=float)
in_seqU70 = np.array([list(map(float, row.strip('()').split())) for row in dataU70.iloc[19:419, 0]], dtype=float)
in_seqU71 = np.array([list(map(float, row.strip('()').split())) for row in dataU71.iloc[19:419, 0]], dtype=float)
in_seqU72 = np.array([list(map(float, row.strip('()').split())) for row in dataU72.iloc[19:419, 0]], dtype=float)
in_seqU73 = np.array([list(map(float, row.strip('()').split())) for row in dataU73.iloc[19:419, 0]], dtype=float)
in_seqU74 = np.array([list(map(float, row.strip('()').split())) for row in dataU74.iloc[19:419, 0]], dtype=float)
in_seqU75 = np.array([list(map(float, row.strip('()').split())) for row in dataU75.iloc[19:419, 0]], dtype=float)
in_seqU76 = np.array([list(map(float, row.strip('()').split())) for row in dataU76.iloc[19:419, 0]], dtype=float)
in_seqU77 = np.array([list(map(float, row.strip('()').split())) for row in dataU77.iloc[19:419, 0]], dtype=float)
in_seqU78 = np.array([list(map(float, row.strip('()').split())) for row in dataU78.iloc[19:419, 0]], dtype=float)
in_seqU79 = np.array([list(map(float, row.strip('()').split())) for row in dataU79.iloc[19:419, 0]], dtype=float)
in_seqU80 = np.array([list(map(float, row.strip('()').split())) for row in dataU80.iloc[19:419, 0]], dtype=float)

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
test_seqU1 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU1.iloc[19:419, 0]], dtype=float)
test_seqU2 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU2.iloc[19:419, 0]], dtype=float)
test_seqU3 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU3.iloc[19:419, 0]], dtype=float)
test_seqU4 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU4.iloc[19:419, 0]], dtype=float)
test_seqU5 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU5.iloc[19:419, 0]], dtype=float)
test_seqU6 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU6.iloc[19:419, 0]], dtype=float)
test_seqU7 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU7.iloc[19:419, 0]], dtype=float)
test_seqU8 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU8.iloc[19:419, 0]], dtype=float)
test_seqU9 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU9.iloc[19:419, 0]], dtype=float)
test_seqU10 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU10.iloc[19:419, 0]], dtype=float)
test_seqU11 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU11.iloc[19:419, 0]], dtype=float)
test_seqU12 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU12.iloc[19:419, 0]], dtype=float)
test_seqU13 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU13.iloc[19:419, 0]], dtype=float)
test_seqU14 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU14.iloc[19:419, 0]], dtype=float)
test_seqU15 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU15.iloc[19:419, 0]], dtype=float)
test_seqU16 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU16.iloc[19:419, 0]], dtype=float)
test_seqU17 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU17.iloc[19:419, 0]], dtype=float)
test_seqU18 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU18.iloc[19:419, 0]], dtype=float)
test_seqU19 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU19.iloc[19:419, 0]], dtype=float)
test_seqU20 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU20.iloc[19:419, 0]], dtype=float)
test_seqU21 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU21.iloc[19:419, 0]], dtype=float)
test_seqU22 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU22.iloc[19:419, 0]], dtype=float)
test_seqU23 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU23.iloc[19:419, 0]], dtype=float)
test_seqU24 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU24.iloc[19:419, 0]], dtype=float)
test_seqU25 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU25.iloc[19:419, 0]], dtype=float)
test_seqU26 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU26.iloc[19:419, 0]], dtype=float)
test_seqU27 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU27.iloc[19:419, 0]], dtype=float)
test_seqU28 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU28.iloc[19:419, 0]], dtype=float)
test_seqU29 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU29.iloc[19:419, 0]], dtype=float)
test_seqU30 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU30.iloc[19:419, 0]], dtype=float)
test_seqU31 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU31.iloc[19:419, 0]], dtype=float)
test_seqU32 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU32.iloc[19:419, 0]], dtype=float)
test_seqU33 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU33.iloc[19:419, 0]], dtype=float)
test_seqU34 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU34.iloc[19:419, 0]], dtype=float)
test_seqU35 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU35.iloc[19:419, 0]], dtype=float)
test_seqU36 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU36.iloc[19:419, 0]], dtype=float)
test_seqU37 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU37.iloc[19:419, 0]], dtype=float)
test_seqU38 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU38.iloc[19:419, 0]], dtype=float)
test_seqU39 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU39.iloc[19:419, 0]], dtype=float)
test_seqU40 = np.array([list(map(float, row.strip('()').split())) for row in test_dataU40.iloc[19:419, 0]], dtype=float)
#convert values to float type.
in_seq1 = in_seq1.astype(float).reshape(1, (len(in_seq1)))
in_seq2 = in_seq2.astype(float).reshape(1, (len(in_seq2)))
in_seq3 = in_seq3.astype(float).reshape(1, (len(in_seq3)))
in_seq4 = in_seq4.astype(float).reshape(1, (len(in_seq4)))
in_seq5 = in_seq5.astype(float).reshape(1, (len(in_seq5)))
in_seq6 = in_seq6.astype(float).reshape(1, (len(in_seq6)))
in_seq7 = in_seq7.astype(float).reshape(1, (len(in_seq7)))
in_seq8 = in_seq8.astype(float).reshape(1, (len(in_seq8)))
in_seq9 = in_seq9.astype(float).reshape(1, (len(in_seq9)))
in_seq10 = in_seq10.astype(float).reshape(1, (len(in_seq10)))
in_seq11 = in_seq11.astype(float).reshape(1, (len(in_seq11)))
in_seq12 = in_seq12.astype(float).reshape(1, (len(in_seq12)))
in_seq13 = in_seq13.astype(float).reshape(1, (len(in_seq13)))
in_seq14 = in_seq14.astype(float).reshape(1, (len(in_seq14)))
in_seq15 = in_seq15.astype(float).reshape(1, (len(in_seq15)))
in_seq16 = in_seq16.astype(float).reshape(1, (len(in_seq16)))
in_seq17 = in_seq17.astype(float).reshape(1, (len(in_seq17)))
in_seq18 = in_seq18.astype(float).reshape(1, (len(in_seq18)))
in_seq19 = in_seq19.astype(float).reshape(1, (len(in_seq19)))
in_seq20 = in_seq20.astype(float).reshape(1, (len(in_seq20)))
in_seq21 = in_seq21.astype(float).reshape(1, (len(in_seq21)))
in_seq22 = in_seq22.astype(float).reshape(1, (len(in_seq22)))
in_seq23 = in_seq23.astype(float).reshape(1, (len(in_seq23)))
in_seq24 = in_seq24.astype(float).reshape(1, (len(in_seq24)))
in_seq25 = in_seq25.astype(float).reshape(1, (len(in_seq25)))
in_seq26 = in_seq26.astype(float).reshape(1, (len(in_seq26)))
in_seq27 = in_seq27.astype(float).reshape(1, (len(in_seq27)))
in_seq28 = in_seq28.astype(float).reshape(1, (len(in_seq28)))
in_seq29 = in_seq29.astype(float).reshape(1, (len(in_seq29)))
in_seq30 = in_seq30.astype(float).reshape(1, (len(in_seq30)))
in_seq31 = in_seq31.astype(float).reshape(1, (len(in_seq31)))
in_seq32 = in_seq32.astype(float).reshape(1, (len(in_seq32)))
in_seq33 = in_seq33.astype(float).reshape(1, (len(in_seq33)))
in_seq34 = in_seq34.astype(float).reshape(1, (len(in_seq34)))
in_seq35 = in_seq35.astype(float).reshape(1, (len(in_seq35)))
in_seq36 = in_seq36.astype(float).reshape(1, (len(in_seq36)))
in_seq37 = in_seq37.astype(float).reshape(1, (len(in_seq37)))
in_seq38 = in_seq38.astype(float).reshape(1, (len(in_seq38)))
in_seq39 = in_seq39.astype(float).reshape(1, (len(in_seq39)))
in_seq40 = in_seq40.astype(float).reshape(1, (len(in_seq40)))
in_seq41 = in_seq41.astype(float).reshape(1, (len(in_seq41)))
in_seq42 = in_seq42.astype(float).reshape(1, (len(in_seq42)))
in_seq43 = in_seq43.astype(float).reshape(1, (len(in_seq43)))
in_seq44 = in_seq44.astype(float).reshape(1, (len(in_seq44)))
in_seq45 = in_seq45.astype(float).reshape(1, (len(in_seq45)))
in_seq46 = in_seq46.astype(float).reshape(1, (len(in_seq46)))
in_seq47 = in_seq47.astype(float).reshape(1, (len(in_seq47)))
in_seq48 = in_seq48.astype(float).reshape(1, (len(in_seq48)))
in_seq49 = in_seq49.astype(float).reshape(1, (len(in_seq49)))
in_seq50 = in_seq50.astype(float).reshape(1, (len(in_seq50)))
in_seq51 = in_seq51.astype(float).reshape(1, (len(in_seq51)))
in_seq52 = in_seq52.astype(float).reshape(1, (len(in_seq52)))
in_seq53 = in_seq53.astype(float).reshape(1, (len(in_seq53)))
in_seq54 = in_seq54.astype(float).reshape(1, (len(in_seq54)))
in_seq55 = in_seq55.astype(float).reshape(1, (len(in_seq55)))
in_seq56 = in_seq56.astype(float).reshape(1, (len(in_seq56)))
in_seq57 = in_seq57.astype(float).reshape(1, (len(in_seq57)))
in_seq58 = in_seq58.astype(float).reshape(1, (len(in_seq58)))
in_seq59 = in_seq59.astype(float).reshape(1, (len(in_seq59)))
in_seq60 = in_seq60.astype(float).reshape(1, (len(in_seq60)))
in_seq61 = in_seq61.astype(float).reshape(1, (len(in_seq61)))
in_seq62 = in_seq62.astype(float).reshape(1, (len(in_seq62)))
in_seq63 = in_seq63.astype(float).reshape(1, (len(in_seq63)))
in_seq64 = in_seq64.astype(float).reshape(1, (len(in_seq64)))
in_seq65 = in_seq65.astype(float).reshape(1, (len(in_seq65)))
in_seq66 = in_seq66.astype(float).reshape(1, (len(in_seq66)))
in_seq67 = in_seq67.astype(float).reshape(1, (len(in_seq67)))
in_seq68 = in_seq68.astype(float).reshape(1, (len(in_seq68)))
in_seq69 = in_seq69.astype(float).reshape(1, (len(in_seq69)))
in_seq70 = in_seq70.astype(float).reshape(1, (len(in_seq70)))
in_seq71 = in_seq71.astype(float).reshape(1, (len(in_seq71)))
in_seq72 = in_seq72.astype(float).reshape(1, (len(in_seq72)))
in_seq73 = in_seq73.astype(float).reshape(1, (len(in_seq73)))
in_seq74 = in_seq74.astype(float).reshape(1, (len(in_seq74)))
in_seq75 = in_seq75.astype(float).reshape(1, (len(in_seq75)))
in_seq76 = in_seq76.astype(float).reshape(1, (len(in_seq76)))
in_seq77 = in_seq77.astype(float).reshape(1, (len(in_seq77)))
in_seq78 = in_seq78.astype(float).reshape(1, (len(in_seq78)))
in_seq79 = in_seq79.astype(float).reshape(1, (len(in_seq79)))
in_seq80 = in_seq80.astype(float).reshape(1, (len(in_seq80)))
in_seqU1 = in_seqU1.reshape(3, (len(in_seqU1)))
in_seqU2 = in_seqU2.reshape(3, (len(in_seqU2)))
in_seqU3 = in_seqU3.reshape(3, (len(in_seqU3)))
in_seqU4 = in_seqU4.reshape(3, (len(in_seqU4)))
in_seqU5 = in_seqU5.reshape(3, (len(in_seqU5)))
in_seqU6 = in_seqU6.reshape(3, (len(in_seqU6)))
in_seqU7 = in_seqU7.reshape(3, (len(in_seqU7)))
in_seqU8 = in_seqU8.reshape(3, (len(in_seqU8)))
in_seqU9 = in_seqU9.reshape(3, (len(in_seqU9)))
in_seqU10 = in_seqU10.reshape(3, (len(in_seqU10)))
in_seqU11 = in_seqU11.reshape(3, (len(in_seqU11)))
in_seqU12 = in_seqU12.reshape(3, (len(in_seqU12)))
in_seqU13 = in_seqU13.reshape(3, (len(in_seqU13)))
in_seqU14 = in_seqU14.reshape(3, (len(in_seqU14)))
in_seqU15 = in_seqU15.reshape(3, (len(in_seqU15)))
in_seqU16 = in_seqU16.reshape(3, (len(in_seqU16)))
in_seqU17 = in_seqU17.reshape(3, (len(in_seqU17)))
in_seqU18 = in_seqU18.reshape(3, (len(in_seqU18)))
in_seqU19 = in_seqU19.reshape(3, (len(in_seqU19)))
in_seqU20 = in_seqU20.reshape(3, (len(in_seqU20)))
in_seqU21 = in_seqU21.reshape(3, (len(in_seqU21)))
in_seqU22 = in_seqU22.reshape(3, (len(in_seqU22)))
in_seqU23 = in_seqU23.reshape(3, (len(in_seqU23)))
in_seqU24 = in_seqU24.reshape(3, (len(in_seqU24)))
in_seqU25 = in_seqU25.reshape(3, (len(in_seqU25)))
in_seqU26 = in_seqU26.reshape(3, (len(in_seqU26)))
in_seqU27 = in_seqU27.reshape(3, (len(in_seqU27)))
in_seqU28 = in_seqU28.reshape(3, (len(in_seqU28)))
in_seqU29 = in_seqU29.reshape(3, (len(in_seqU29)))
in_seqU30 = in_seqU30.reshape(3, (len(in_seqU30)))
in_seqU31 = in_seqU31.reshape(3, (len(in_seqU31)))
in_seqU32 = in_seqU32.reshape(3, (len(in_seqU32)))
in_seqU33 = in_seqU33.reshape(3, (len(in_seqU33)))
in_seqU34 = in_seqU34.reshape(3, (len(in_seqU34)))
in_seqU35 = in_seqU35.reshape(3, (len(in_seqU35)))
in_seqU36 = in_seqU36.reshape(3, (len(in_seqU36)))
in_seqU37 = in_seqU37.reshape(3, (len(in_seqU37)))
in_seqU38 = in_seqU38.reshape(3, (len(in_seqU38)))
in_seqU39 = in_seqU39.reshape(3, (len(in_seqU39)))
in_seqU40 = in_seqU40.reshape(3, (len(in_seqU40)))
in_seqU41 = in_seqU41.reshape(3, (len(in_seqU41)))
in_seqU42 = in_seqU42.reshape(3, (len(in_seqU42)))
in_seqU43 = in_seqU43.reshape(3, (len(in_seqU43)))
in_seqU44 = in_seqU44.reshape(3, (len(in_seqU44)))
in_seqU45 = in_seqU45.reshape(3, (len(in_seqU45)))
in_seqU46 = in_seqU46.reshape(3, (len(in_seqU46)))
in_seqU47 = in_seqU47.reshape(3, (len(in_seqU47)))
in_seqU48 = in_seqU48.reshape(3, (len(in_seqU48)))
in_seqU49 = in_seqU49.reshape(3, (len(in_seqU49)))
in_seqU50 = in_seqU50.reshape(3, (len(in_seqU50)))
in_seqU51 = in_seqU51.reshape(3, (len(in_seqU51)))
in_seqU52 = in_seqU52.reshape(3, (len(in_seqU52)))
in_seqU53 = in_seqU53.reshape(3, (len(in_seqU53)))
in_seqU54 = in_seqU54.reshape(3, (len(in_seqU54)))
in_seqU55 = in_seqU55.reshape(3, (len(in_seqU55)))
in_seqU56 = in_seqU56.reshape(3, (len(in_seqU56)))
in_seqU57 = in_seqU57.reshape(3, (len(in_seqU57)))
in_seqU58 = in_seqU58.reshape(3, (len(in_seqU58)))
in_seqU59 = in_seqU59.reshape(3, (len(in_seqU59)))
in_seqU60 = in_seqU60.reshape(3, (len(in_seqU60)))
in_seqU61 = in_seqU61.reshape(3, (len(in_seqU61)))
in_seqU62 = in_seqU62.reshape(3, (len(in_seqU62)))
in_seqU63 = in_seqU63.reshape(3, (len(in_seqU63)))
in_seqU64 = in_seqU64.reshape(3, (len(in_seqU64)))
in_seqU65 = in_seqU65.reshape(3, (len(in_seqU65)))
in_seqU66 = in_seqU66.reshape(3, (len(in_seqU66)))
in_seqU67 = in_seqU67.reshape(3, (len(in_seqU67)))
in_seqU68 = in_seqU68.reshape(3, (len(in_seqU68)))
in_seqU69 = in_seqU69.reshape(3, (len(in_seqU69)))
in_seqU70 = in_seqU70.reshape(3, (len(in_seqU70)))
in_seqU71 = in_seqU71.reshape(3, (len(in_seqU71)))
in_seqU72 = in_seqU72.reshape(3, (len(in_seqU72)))
in_seqU73 = in_seqU73.reshape(3, (len(in_seqU73)))
in_seqU74 = in_seqU74.reshape(3, (len(in_seqU74)))
in_seqU75 = in_seqU75.reshape(3, (len(in_seqU75)))
in_seqU76 = in_seqU76.reshape(3, (len(in_seqU76)))
in_seqU77 = in_seqU77.reshape(3, (len(in_seqU77)))
in_seqU78 = in_seqU78.reshape(3, (len(in_seqU78)))
in_seqU79 = in_seqU79.reshape(3, (len(in_seqU79)))
in_seqU80 = in_seqU80.reshape(3, (len(in_seqU80)))

test_seq1 = test_seq1.astype(float).reshape(1, (len(test_seq1)))
test_seq2 = test_seq2.astype(float).reshape(1, (len(test_seq2)))
test_seq3 = test_seq3.astype(float).reshape(1, (len(test_seq3)))
test_seq4 = test_seq4.astype(float).reshape(1, (len(test_seq4)))
test_seq5 = test_seq5.astype(float).reshape(1, (len(test_seq5)))
test_seq6 = test_seq6.astype(float).reshape(1, (len(test_seq6)))
test_seq7 = test_seq7.astype(float).reshape(1, (len(test_seq7)))
test_seq8 = test_seq8.astype(float).reshape(1, (len(test_seq8)))
test_seq9 = test_seq9.astype(float).reshape(1, (len(test_seq9)))
test_seq10 = test_seq10.astype(float).reshape(1, (len(test_seq10)))
test_seq11 = test_seq11.astype(float).reshape(1, (len(test_seq11)))
test_seq12 = test_seq12.astype(float).reshape(1, (len(test_seq12)))
test_seq13 = test_seq13.astype(float).reshape(1, (len(test_seq13)))
test_seq14 = test_seq14.astype(float).reshape(1, (len(test_seq14)))
test_seq15 = test_seq15.astype(float).reshape(1, (len(test_seq15)))
test_seq16 = test_seq16.astype(float).reshape(1, (len(test_seq16)))
test_seq17 = test_seq17.astype(float).reshape(1, (len(test_seq17)))
test_seq18 = test_seq18.astype(float).reshape(1, (len(test_seq18)))
test_seq19 = test_seq19.astype(float).reshape(1, (len(test_seq19)))
test_seq20 = test_seq20.astype(float).reshape(1, (len(test_seq20)))
test_seq21 = test_seq21.astype(float).reshape(1, (len(test_seq21)))
test_seq22 = test_seq22.astype(float).reshape(1, (len(test_seq22)))
test_seq23 = test_seq23.astype(float).reshape(1, (len(test_seq23)))
test_seq24 = test_seq24.astype(float).reshape(1, (len(test_seq24)))
test_seq25 = test_seq25.astype(float).reshape(1, (len(test_seq25)))
test_seq26 = test_seq26.astype(float).reshape(1, (len(test_seq26)))
test_seq27 = test_seq27.astype(float).reshape(1, (len(test_seq27)))
test_seq28 = test_seq28.astype(float).reshape(1, (len(test_seq28)))
test_seq29 = test_seq29.astype(float).reshape(1, (len(test_seq29)))
test_seq30 = test_seq30.astype(float).reshape(1, (len(test_seq30)))
test_seq31 = test_seq31.astype(float).reshape(1, (len(test_seq31)))
test_seq32 = test_seq32.astype(float).reshape(1, (len(test_seq32)))
test_seq33 = test_seq33.astype(float).reshape(1, (len(test_seq33)))
test_seq34 = test_seq34.astype(float).reshape(1, (len(test_seq34)))
test_seq35 = test_seq35.astype(float).reshape(1, (len(test_seq35)))
test_seq36 = test_seq36.astype(float).reshape(1, (len(test_seq36)))
test_seq37 = test_seq37.astype(float).reshape(1, (len(test_seq37)))
test_seq38 = test_seq38.astype(float).reshape(1, (len(test_seq38)))
test_seq39 = test_seq39.astype(float).reshape(1, (len(test_seq39)))
test_seq40 = test_seq40.astype(float).reshape(1, (len(test_seq40)))
test_seqU1 = test_seqU1.reshape(3, (len(test_seqU1)))
test_seqU2 = test_seqU2.reshape(3, (len(test_seqU2)))
test_seqU3 = test_seqU3.reshape(3, (len(test_seqU3)))
test_seqU4 = test_seqU4.reshape(3, (len(test_seqU4)))
test_seqU5 = test_seqU5.reshape(3, (len(test_seqU5)))
test_seqU6 = test_seqU6.reshape(3, (len(test_seqU6)))
test_seqU7 = test_seqU7.reshape(3, (len(test_seqU7)))
test_seqU8 = test_seqU8.reshape(3, (len(test_seqU8)))
test_seqU9 = test_seqU9.reshape(3, (len(test_seqU9)))
test_seqU10 = test_seqU10.reshape(3, (len(test_seqU10)))
test_seqU11 = test_seqU11.reshape(3, (len(test_seqU11)))
test_seqU12 = test_seqU12.reshape(3, (len(test_seqU12)))
test_seqU13 = test_seqU13.reshape(3, (len(test_seqU13)))
test_seqU14 = test_seqU14.reshape(3, (len(test_seqU14)))
test_seqU15 = test_seqU15.reshape(3, (len(test_seqU15)))
test_seqU16 = test_seqU16.reshape(3, (len(test_seqU16)))
test_seqU17 = test_seqU17.reshape(3, (len(test_seqU17)))
test_seqU18 = test_seqU18.reshape(3, (len(test_seqU18)))
test_seqU19 = test_seqU19.reshape(3, (len(test_seqU19)))
test_seqU20 = test_seqU20.reshape(3, (len(test_seqU20)))
test_seqU21 = test_seqU21.reshape(3, (len(test_seqU21)))
test_seqU22 = test_seqU22.reshape(3, (len(test_seqU22)))
test_seqU23 = test_seqU23.reshape(3, (len(test_seqU23)))
test_seqU24 = test_seqU24.reshape(3, (len(test_seqU24)))
test_seqU25 = test_seqU25.reshape(3, (len(test_seqU25)))
test_seqU26 = test_seqU26.reshape(3, (len(test_seqU26)))
test_seqU27 = test_seqU27.reshape(3, (len(test_seqU27)))
test_seqU28 = test_seqU28.reshape(3, (len(test_seqU28)))
test_seqU29 = test_seqU29.reshape(3, (len(test_seqU29)))
test_seqU30 = test_seqU30.reshape(3, (len(test_seqU30)))
test_seqU31 = test_seqU31.reshape(3, (len(test_seqU31)))
test_seqU32 = test_seqU32.reshape(3, (len(test_seqU32)))
test_seqU33 = test_seqU33.reshape(3, (len(test_seqU33)))
test_seqU34 = test_seqU34.reshape(3, (len(test_seqU34)))
test_seqU35 = test_seqU35.reshape(3, (len(test_seqU35)))
test_seqU36 = test_seqU36.reshape(3, (len(test_seqU36)))
test_seqU37 = test_seqU37.reshape(3, (len(test_seqU37)))
test_seqU38 = test_seqU38.reshape(3, (len(test_seqU38)))
test_seqU39 = test_seqU39.reshape(3, (len(test_seqU39)))
test_seqU40 = test_seqU40.reshape(3, (len(test_seqU40)))
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
out_seqU1 = in_seqU5
out_seqU2 = in_seqU10
out_seqU3 = in_seqU15
out_seqU4 = in_seqU20
out_seqU5 = in_seqU25
out_seqU6 = in_seqU30
out_seqU7 = in_seqU35
out_seqU8 = in_seqU40
out_seqU9 = in_seqU45
out_seqU10 = in_seqU50
out_seqU11 = in_seqU55
out_seqU12 = in_seqU60
out_seqU13 = in_seqU65
out_seqU14 = in_seqU70
out_seqU15 = in_seqU75
out_seqU16 = in_seqU80


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
						  out_seq6, out_seq7, out_seq8, out_seq9, out_seq10,
						  out_seq11, out_seq12, out_seq13, out_seq14, out_seq15,
						  out_seq16), axis=0)
datasetU = np.concatenate((in_seqU1, in_seqU2, in_seqU3, in_seqU4, in_seqU5,
						  in_seqU6, in_seqU7, in_seqU8, in_seqU9, in_seqU10,
						  in_seqU11, in_seqU12, in_seqU13, in_seqU14, in_seqU15,
						  in_seqU16, in_seqU17, in_seqU18, in_seqU19, in_seqU20,
						  in_seqU21, in_seqU22, in_seqU23, in_seqU24, in_seqU25,
						  in_seqU26, in_seqU27, in_seqU28, in_seqU29, in_seqU30,
						  in_seqU31, in_seqU32, in_seqU33, in_seqU34, in_seqU35,
						  in_seqU36, in_seqU37, in_seqU38, in_seqU39, in_seqU40,
						  in_seqU41, in_seqU42, in_seqU43, in_seqU44, in_seqU45,
						  in_seqU46, in_seqU47, in_seqU48, in_seqU49, in_seqU50,
						  in_seqU51, in_seqU52, in_seqU53, in_seqU54, in_seqU55,
						  in_seqU56, in_seqU57, in_seqU58, in_seqU59, in_seqU60,
						  in_seqU61, in_seqU62, in_seqU63, in_seqU64, in_seqU65,
						  in_seqU66, in_seqU67, in_seqU68, in_seqU69, in_seqU70,
						  in_seqU71, in_seqU72, in_seqU73, in_seqU74, in_seqU75,
						  in_seqU76, in_seqU77, in_seqU78, in_seqU79, in_seqU80,
						  out_seqU1, out_seqU2, out_seqU3, out_seqU4, out_seqU5,
						  out_seqU6, out_seqU7, out_seqU8, out_seqU9, out_seqU10,
						  out_seqU11, out_seqU12, out_seqU13, out_seqU14, out_seqU15,
						  out_seqU16), axis=0)

test_datasetU = np.concatenate((test_seqU1, test_seqU2, test_seqU3, test_seqU4, test_seqU5,
							   test_seqU6, test_seqU7, test_seqU8, test_seqU9, test_seqU10,
							   test_seqU11, test_seqU12, test_seqU13, test_seqU14, test_seqU15,
							   test_seqU16, test_seqU17, test_seqU18, test_seqU19, test_seqU20,
							   test_seqU21, test_seqU22, test_seqU23, test_seqU24, test_seqU25,
							   test_seqU26, test_seqU27, test_seqU28, test_seqU29, test_seqU30,
							   test_seqU31, test_seqU32, test_seqU33, test_seqU34, test_seqU35,
							   test_seqU36, test_seqU37, test_seqU38, test_seqU40, test_seqU40,
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

#constants
EPOCHS = 100
FILTER_SIZE = 64
DENSE_SIZE = 64
BATCH_SIZE = 32 #Default is 32
ITERATION = 1

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
	output = Dense(n_features, activation='relu')(dense)
	model = Model(inputs=visible1, outputs=output)
	model.compile(optimizer='adam', loss='mse')
	return model

# choose a number of time steps
n_steps = 2
X = []
y = []
y2 = []
test = []
samples = dataset.shape[0]-n_steps
test_samples = test_dataset.shape[0]-n_steps
for i in range(samples):
	X.append( dataset[i:i+n_steps] )
	X.append(datasetU[i:i+n_steps])
	y.append( dataset[-1] )

for z in range(samples):
	y2.append( datasetU[-1] )
	
for j in range(test_samples):
	test.append(test_dataset[j:j+n_steps])
	test.append(test_datasetU[j:j+n_steps])

X = np.array(X)
n_features = X.shape[2]

y = np.concatenate((y,y2), axis=0)
y = np.array(y).reshape( samples*2, 1, n_features )
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

#GPU
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#tf.debugging.set_log_device_placement(True)

#with tf.device('/GPU:0'):
#start of the measuring
#time measuring in python
profiler = cProfile.Profile()
profiler.enable()
tracemalloc.start()
# fit model
#add batch size
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE , epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
print("Memory Usage: ")
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
profiler.disable()
#profiler.print_stats(sort='cumulative')
profiler.dump_stats(f'profiler_results_epochs{EPOCHS}_batch_{BATCH_SIZE}_iteration_{ITERATION}.prof')
#end of the measuring


# demonstrate prediction
#here as well
yhat = model.predict(test, verbose=1)
print(yhat)

#Testing the accuracy of the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Reshaping the test data to fit the output of the program.
test_flattened = test.reshape(-1, test.shape[-1]) # ndarray (76,400)
yhat_flattened = yhat.reshape(-1, yhat.shape[-1]) # ndarray (38, 400)
mse = mean_squared_error(test_flattened[:76, :], yhat_flattened) #taking the first 3 rows to match the sizes
mae = mean_absolute_error(test_flattened[:76, :], yhat_flattened)
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
from scipy.stats import pearsonr
coef_P, _ = pearsonr(test_flattened[:76, :].flatten(), yhat_flattened.flatten())
print("Pearson Correlation: ", coef_P)

#Spearman correlation
from scipy.stats import spearmanr
coef, p = spearmanr(test_flattened[:76, :], yhat_flattened)
flat_coef, _ = spearmanr(test_flattened[:76, :].flatten(), yhat_flattened.flatten())
print("Spearman Correlation: ", flat_coef)

#Coefficient of determination R
R_square = np.absolute(coef_P * coef_P) #R^2 cannot be a negative value
print("Coefficient of Determination: ", R_square)

import pstats
stats = pstats.Stats(f'profiler_results_epochs{EPOCHS}_batch_{BATCH_SIZE}_iteration_{ITERATION}.prof')
stats.sort_stats('cumulative').print_stats()
# %%
