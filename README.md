# Neural_computing
Comparative analysis of two algorithms Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM) for classification of cardiotocogram (CTG) to determine fetal state.

This is how to run the files for comparative analysis of two algorithms Multi-Layer Perceptron
(MLP) and Support Vector Machine (SVM) for classification of cardiotocogram (CTG).

Initial data from UCI - http://archive.ics.uci.edu/ml/datasets/Cardiotocography

For testing final models only run (these files are saved with best hyperparameters from
training):
1 -MLP_CTG_final.m
2- SVM_CTG_final.m

Other Files and workflow:
1. data.csv (initial data from UCI used in data_processing.m)
2. ADASYN.m (Function from
https://www.mathworks.com/matlabcentral/fileexchange/50541-adasyn-improves-classbalance-extension-of-smote, used in data_processing.m)
3. CTG_processing.m (for data cleaning and preprocessing)
4. processed_Data.csv (the cleaned data created and saved by the data_processing.m file)
5. split_train_testdata.m (for splitting the data into training and testing. Saves the training
and testing data in the 'splitted_data.mat' file)
6. splitted_data.mat (training and testing data splitted into classes and features)
7. MLP_trainingmdl.m (grid search and final MLP model, takes about 57 minutes)
5. SVM_trainingmdl.m (grid search and final SVM model, takes about 3 hours)

