## Project Description
In this project I built a one layer neural network completely from scratch, without the help of any ML packages. I first looked up the steps behind training a one layer neural network and then proceeded to program the math in a systematic way. After writing the code for the NN, I then trained it on the Wisconsin Breast Cancer Dataset (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). More information about the dataset:


- There are 699 rows in total, belonging to 699 patients.

- The first column is an ID that identifies each patient.

- The following 9 columns are features that express different types of information connected to the detected tumors. They represent data related to: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli and Mitoses.

- The last column is the class of the tumor and it has two possible values: 2 means that the tumor was found to be benign. 4 means that it was found to be malignant.

- We are told as well that there are a few rows that contain missing data. The missing data is represented in the data-set with the ? character.

- Out of the 699 patients in the dataset, the class distribution is: Benign: 458 (65.5%) and Malignant: 241 (34.5%).


The initial go at this project can be seen in the Jupyter Notebook. After successfully training my NN scratch and making predictions on the dataset, I then packaged the code into two scripts, `Self_Optimizer` and `You_Optimize`. `Self_Optimizer` tunes the node count of the first layer itself and returns the most optimal NN model. We are simply defining most optimal as the model with the lowest validation/test misclassification rate.

## How to Build

1) **You_Optimize:** This script takes as input the NN parameters you feed it, trains the NN 50,000 times and returns the most optimal misclassification rate. The first parameter you feed it has to be the node count of the first layer of the network, and the second parameter has to be the learning rate of the network. Example: `Python3 You_Optimize.py 5 0.05`

2) **Self_Optimize:** This script automatically tunes the network so no input is required from the user. Example: `Python3 Self_Optimizer.py`
