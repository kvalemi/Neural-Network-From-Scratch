## Libraries ##
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import itertools
from sklearn.utils import shuffle
import sys

## Relative import ##
import nnet

## Starting Point ##
def main():

	## Data Preparation ##

    # Read in the dataset
	df = pd.read_csv('wisconsin-cancer-dataset.csv', header = None)

	# Normalize the response variable to 0-1 scale
	df.iloc[:, 10].replace(2, 0, inplace = True)
	df.iloc[:, 10].replace(4, 1, inplace = True)

	# Remove expl vars with unknown values
	df = df[~df[6].isin(['?'])]

	# Set DF type
	df = df.astype(float)

	# Apply min-max normalization
	names = df.columns[0:10]
	scaler = MinMaxScaler()
	scaled_df = scaler.fit_transform(df.iloc[:, 0:10])
	scaled_df = pd.DataFrame(scaled_df, columns = names)

	# Split the data up into test and training
	scaled_df = shuffle(scaled_df).reset_index()

	# training data
	x=scaled_df.iloc[0:500,1:10].values.transpose()
	y=df.iloc[0:500,10:].values.transpose()

	# test data
	xval=scaled_df.iloc[501:683,1:10].values.transpose()
	yval=df.iloc[501:683,10:].values.transpose()


	## Training ##

	# Create NN instance and train it
	nn = nnet.dlnet(x, y)
	nn.dims = [9, int(sys.argv[1]), 1]
	nn.lr = float(sys.argv[2])

	print("--> Training NN with dimensions (9, ", sys.argv[1], ", 1) and LR = ", sys.argv[2])

	# 50000 iterations of back propogation
	nn.gd(x, y, iter = 50000)

	# Training misclassification rate
	print("\n")
	print("Training misclassification rate:")
	pred_train = nn.pred(x, y)

	# Test misclassification rate
	print("Test misclassification rate:")
	pred_test = nn.pred(xval, yval)

	# print the training and test set predictions
	nn.X = xval
	nn.Y = yval

	yvalh, loss = nn.forward()

	print("\n")

	print("Predicted training values:")
	print("\ny",np.around(yval[:,0:50,], decimals=0).astype(np.int), "\n")       

	print("Predicted test values:")
	print("\nyh",np.around(yvalh[:,0:50,], decimals=0).astype(np.int),"\n")


if __name__ == "__main__":
    main()

