%Pre-processing of CTG dataset

%clear all; 

CTG_data = readtable('data.csv');


%Find missing data
FNan = ismissing(CTG_data, {NaN, 'NA'});
CTG_data(any(FNan,2),:)

%Delete the rows which contain NaN 
dataTable = CTG_data(~any(FNan,2),:);

%Make into an array 
dataArray = table2array(dataTable); 
dataArray(:,25) = categorical(dataArray(:,25));

%Count the target class 
class_labels = categorical(dataArray(:,25));
counts_class = countcats(class_labels); 

%Summary 
summaries = summary(dataTable);

%Summaries grouped by class 
summaryGroups = grpstats(dataTable, 'Class');


%% Implement resampling ADASYN %%
%This function can be found in the link below
%https://uk.mathworks.com/matlabcentral/fileexchange/50541-adasyn-improves-class-balance-extension-of-smote

%features class 1  - Normal %
class1Features = dataArray((dataArray(:,25)==1),:);
%features class 2 - Suspect %
class2Features = dataArray((dataArray(:,25)==2),:);
%Features class 3 - Pathologic %
class3Features = dataArray((dataArray(:,25)==3),:);

%Class labels %
%class 1 becomes 0 %
class1_0 = false([1655 1]);

%class 2 becomes 1 %
class2_1 = true ([295 1]);

%class 3 becomes 1 %
class3_1 = true ([176 1]);

%Adasyn the data for class 2 %
adasyn_features                 = [class1Features(:,1:24); class2Features(:,1:24)];
adasyn_labels                  = [class1_0  ; class2_1];
adasyn_beta                     = [0.75];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization
    
[adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);

%Adasyn the data for class 3 % 
adasyn_features                 = [class1Features(:,1:24); class3Features(:,1:24)];
adasyn_labels                  = [class1_0  ; class3_1];
adasyn_beta                     = [0.75];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization


[adasyn_featuresSyn2, adasyn_labelsSyn2] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);

%Put all the data together 
twos = repelem(2,1010);
twos = twos';
newClass2 = [adasyn_featuresSyn, twos];

threes = repelem(3,1109);
threes = threes';
newClass3 = [adasyn_featuresSyn2, threes];

processed_data = [dataArray;newClass2;newClass3];

%Normalise Data 
normalize_data = normalize(processed_data(:,1:24));
finalprocessed = [normalize_data,processed_data(:,25)];

%Write to csv file 

csvwrite('processed_data.csv', finalprocessed)
