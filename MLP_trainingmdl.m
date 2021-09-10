clear all;
rng(19);

% Load data
load('splitted_data.mat')

%% GRID SEARCH for Training the Network %%
 % It was tried to use 2 different the training function
                        % 'trainscg' and 'trainlm' - the last one produces
                        % better results and runned faster
                      
% Training Function:
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation

%trainFcn = 'trainscg';  % Scaled Conjugate Gradient backpropagation
 %For trying the 'trainscg' training funcion one must uncomment the line for 'trainscg' (line 15) 
                       
% Hyperparameters:
learning_Rate = [0.05, 0.2, 0.5, 0.9];   
momentum_Param = [0.3, 0.6, 0.8, 0.9];
hidden_Layer_Size = [8, 10, 12, 15];

% Table to store the results of the grid search in (average of all 10
% iterations of the cross validation stored)
Grid_search_tbl = [];
updateGrid = [];

% Accuracy/Error/whole Table (each value of each of the 10 iterations
% within the Cross Validation is included)
All_Acc = [];
All_error = []; 
CV_tbl_updated = [];
CV_tbl = [];

counter = 0;

% 10-fold cross validation
indices = crossvalind('Kfold', classes_Training,10);

% Grid Search:
%Loop for i, j, k 10 times, on 9 different training sets and tested on changing 1 validation set

for i = learning_Rate  % 1st loop goes throughfour different learning rates 
    
    for j = momentum_Param % 2nd loop iterates through four different numbers for momentum
        
      for k = hidden_Layer_Size % 3dr loop goes through different numbers of neurons for the four hidden layers 
        
          Acc = [];
          Error = [];
          counter = counter+1;
          
        for x = 1:10
           
            % index for validation
            validation_idx = (indices == x);
            train_idx = ~validation_idx;
           
            % network with four hidden layers:
            net = fitnet([k,k,k,k],trainFcn);
            net.trainParam.epochs = 200;  % Max. number of epochs (stops after 200 epochs)
      
        %net.trainParam.max_fail = 100  %Different values was tested but 6 validation checks is mantained  
   
            net.performFcn = 'mse';       % default Error function

            net.trainParam.lr = i;        % Learning rate 
            net.trainParam.mc = j;        % Momentum parameter 
            net.layers{1}.transferFcn = 'tansig';   % Activation Function for the four Layers
            net.layers{2}.transferFcn = 'tansig';
            net.layers{3}.transferFcn = 'tansig';   
            net.layers{4}.transferFcn = 'tansig';

            % training the network
            [net,tr] = train(net,features_Training(train_idx,:)', classes_Training(train_idx,:)');
           
            % y = output of the network
            y = net(features_Training(validation_idx,:)');
           
            % performance of the network (error)
            performance = perform(net,y,classes_Training(validation_idx,:)');

            fprintf('fitnet, performance: %f\n', performance);
            fprintf('number of epochs: %d, stop: %s\n', tr.num_epochs, tr.stop);

            %create the confusion matrix    
            CM_mdl= confusionmat(classes_Training(validation_idx,:)',round(y));
            
            % calculate the accuracy of the model using the confusion matrix 
            Acc_mdl = 100*sum(diag(CM_mdl))./sum(CM_mdl(:));
           
           
            % store the model accuracy values in an array
            Acc = [Acc; Acc_mdl]; 
            All_Acc = [All_Acc;Acc_mdl];
           
            % store the model performance values in an array
            Error = [Error; performance];
            All_error =[All_error; performance];

            CV_tbl_updated = [counter i j k performance Acc_mdl];
            CV_tbl = [CV_tbl ; CV_tbl_updated];
           
        end
           
           meanAcc = sum(Acc)/10;
           meanError = sum(Error)/10;
           updateGrid = [counter i j k meanError meanAcc]
           Grid_search_tbl = [Grid_search_tbl ; updateGrid]
           
      end
      
    end
    
end

%% Finding best hyperparameters %%
% Index with highest Accuracy in tableGrid
[~,ModelIdx1]=max(Grid_search_tbl(:,6));

% Highest Accuracy
max_Accuracy = Grid_search_tbl(ModelIdx1,6);
Val_Max_Acc = Grid_search_tbl(ModelIdx1,:);

% Index with lowest Error in tableGrid
[~,ModelIdx2]=min(Grid_search_tbl(:,5));

% Lowest Error
min_Error = Grid_search_tbl(ModelIdx2,5);
ValuesMinError = Grid_search_tbl(ModelIdx2,:);

% Find the best parameters that result in the highest Accuracy
Val_Max_Acc = array2table(Val_Max_Acc);
Best_Learning_Rate = Val_Max_Acc{:,2} ; 
Best_Momentum_Param = Val_Max_Acc{:,3};
Best_Hidden_LayerSize = Val_Max_Acc{:,4};



%% BEST MODEL %%
% Train a new model using the "best" parameters 
netBest = fitnet([Best_Hidden_LayerSize,Best_Hidden_LayerSize,Best_Hidden_LayerSize,Best_Hidden_LayerSize],trainFcn);
netBest.trainParam.epochs = 200;               % Num. epochs (after 200 epochs the training stops)
netBest.performFcn = 'mse';                    % Error function
netBest.trainParam.lr = Best_Learning_Rate;      % Learning rate 
netBest.trainParam.mc = Best_Momentum_Param; % Momentum parameter
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression',  'plotconfusion'}; %Plot graphs to improve training when necessary
% Train network with the training data
[netBest,tr2] = train(netBest,features_Training', classes_Training');


%% TRAINING ACCURACY %%
% Test network with the training data in order to get the training accuracy (training outcome)
y_Training = netBest(features_Training');

% Performance Ttraining
performanceTraining = perform(netBest,y_Training,classes_Training');

% Confustion matrix training
trainingConfusion= confusionmat(classes_Training',round(y_Training));

% Accuracy Training
trainingAccuracy = 100*sum(diag(trainingConfusion))./sum(trainingConfusion(:));

figure(1);
confusionchart(trainingConfusion)
title("MLP Confusion Matrix of Training Data")

% Training Metrics - Precision | Recall | F1-Score
% Class 1
training_precision_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(2,1)+trainingConfusion(3,1)))

training_recall_class1 = trainingConfusion(1,1) / ...
    (trainingConfusion(1,1) + (trainingConfusion(1,2)+trainingConfusion(1,3)))

training_F1Score_class1 = 2 * ((training_precision_class1*training_recall_class1)/(training_precision_class1+training_recall_class1))

%Class 2
training_precision_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(1,2)+trainingConfusion(3,2)))

training_recall_class2 = trainingConfusion(2,2) / ...
    (trainingConfusion(2,2) + (trainingConfusion(2,1)+trainingConfusion(2,3)))

training_F1Score_class2 = 2 * ((training_precision_class2*training_recall_class2)/(training_precision_class2+training_recall_class2))

%Class 3
training_precision_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(1,3)+trainingConfusion(2,3)))

training_recall_class3 = trainingConfusion(3,3) / ...
    (trainingConfusion(3,3) + (trainingConfusion(3,1)+trainingConfusion(3,2)))

training_F1Score_class3 = 2 * ((training_precision_class3*training_recall_class3)/(training_precision_class3+training_recall_class3))

%Final Table TRAINING - Precision | Recall | F1-Score
trainingMetrics = [training_precision_class1, training_recall_class1, training_F1Score_class1; training_precision_class2, ...
    training_recall_class2, training_F1Score_class2; training_precision_class3, training_recall_class3, training_F1Score_class3]
trainingMetrics = array2table(trainingMetrics)
trainingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}



%% TEST ACCURACY %%
% Test network with the test data in order to get the test accuracy (unseen data)
y_Testing = netBest(features_Testing');

% Performance Test
performanceBest = perform(netBest,y_Testing,classes_Testing');

fprintf('fitnet, performance: %f\n', performanceBest);
fprintf('number of epochs: %d, stop: %s\n', tr2.num_epochs, tr2.stop);

% Confustion matrix test
testingConfusion= confusionmat(classes_Testing',round(y_Testing));
testingAccuracy = 100*sum(diag(testingConfusion))./sum(testingConfusion(:));

% Confustion chart (test data)
figure(2);
confusionchart(testingConfusion)
title("MLP Confusion Matrix of Testing Data")

% Testing Metrics - Precision | Recall | F1-Score

%Class 1
testing_precision_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(2,1)+testingConfusion(3,1)))

testing_recall_class1 = testingConfusion(1,1) / ...
    (testingConfusion(1,1) + (testingConfusion(1,2)+testingConfusion(1,3)))

testing_F1Score_class1 = 2 * ((testing_precision_class1*testing_recall_class1)/ ...
    (testing_precision_class1+testing_recall_class1))

% Class 2
testing_precision_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(1,2)+testingConfusion(3,2)))

testing_recall_class2 = testingConfusion(2,2) / ...
    (testingConfusion(2,2) + (testingConfusion(2,1)+testingConfusion(2,3)))

testing_F1Score_class2 = 2 * ((testing_precision_class2*testing_recall_class2)/...
    (testing_precision_class2+testing_recall_class2))

% Class 3
testing_precision_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(1,3)+testingConfusion(2,3)))

testing_recall_class3 = testingConfusion(3,3) / ...
    (testingConfusion(3,3) + (testingConfusion(3,1)+testingConfusion(3,2)))

testing_F1Score_class3 = 2 * ((testing_precision_class3*testing_recall_class3)/...
    (testing_precision_class3+testing_recall_class3))

%Final Table TESTING - Precision | Recall | F1-Score
testingMetrics = [testing_precision_class1, testing_recall_class1, testing_F1Score_class1; testing_precision_class2, ...
    testing_recall_class2, testing_F1Score_class2; testing_precision_class3, testing_recall_class3, testing_F1Score_class3]
testingMetrics = array2table(testingMetrics)
testingMetrics.Properties.VariableNames = {'Precision', 'Recall', 'F1'}





%% GRAPHS %%
%%Check network performance and determine if any changes need 
%to be made to the training process
figure (3)
plotperf(tr)
%Check gradient coefficient
figure(4)
plottrainstate(tr)
% Hyperparameters vs. Accuracy
figure(5);
% Momentum vs accuracy
% take max accuracy per momentum
M_vs_Accuracy = splitapply(@max,Grid_search_tbl(:,6),groupsM);
plot(M_vs_Accuracy)
title("Momentum vs. Accuracy")
xlabel('Momentum')
ylabel('Accuracy (%)')

% Hidden layer size vs accuracy
figure(6);
% group the different values for hidden layer size that were tested
groupsHL=findgroups(Grid_search_tbl(:,4));
% take max accuracy per hidden layer size
HL_vs_Accuracy = splitapply(@max,Grid_search_tbl(:,6),groupsHL);
plot(HL_vs_Accuracy)
title("Hidden Layer Size vs. Accuracy")
xlabel('Hidden Layer Size')
ylabel('Accuracy (%)')
% Learning rate vs accuracy
figure(7);
% group the different learning rates that were tested
groupsL=findgroups(Grid_search_tbl(:,2));
% take max accuracy per learning rate
LR_vs_Accuracy = splitapply(@max,Grid_search_tbl(:,6),groupsL);
plot(LR_vs_Accuracy)
title("Learning Rate vs. Accuracy")
xlabel('Learning Rate')
ylabel('Accuracy (%)')



