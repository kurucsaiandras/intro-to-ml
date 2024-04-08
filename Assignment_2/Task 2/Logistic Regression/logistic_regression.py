from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

##### Load data

dataset = pd.read_csv("../../../Life-Expectancy-Data.csv")

classCategories = np.asarray(dataset["Economy_status_Developed"])

dataset = dataset.drop(["Economy_status_Developed", 'Economy_status_Developing', 'Country'], axis=1)





##### Preprocessing


with_region = True

cols = [col for col in dataset.columns if col not in ['Economy_status_developing']]


if with_region:
    region_encoder = OneHotEncoder()
    region_encoded = region_encoder.fit_transform(dataset[['Region']])

    region_encoded_df = pd.DataFrame(region_encoded.toarray(), columns=region_encoder.get_feature_names_out())

    raw_data = pd.concat([dataset, region_encoded_df], axis=1)

    raw_data = raw_data.drop(['Region'], axis=1)
    cols.remove('Region')

    X = raw_data[cols].values
    y = classCategories
else:
    dataset = dataset.drop(['Region'], axis=1)
    X = dataset[cols].values
    y = classCategories

cols = range(0, len(dataset.columns))
attributeNames = np.asarray(dataset.columns[cols])



##### Run cross validation

best_accuracy = 0
best_models = []

use_stratified = False



K = 10
outer_kfold = (KFold(n_splits=K, shuffle=True, random_state=0) 
                if not use_stratified else StratifiedKFold(n_splits=K, shuffle=True, random_state=0))

for i,(train_idx, val_idx) in enumerate(outer_kfold.split(X,y)):
    

    X_train = X[train_idx]
    X_test = X[val_idx]
    y_train = y[train_idx]
    y_test = y[val_idx]
    
    trainStd = np.std(X_train, axis=0)
    trainMean = np.mean(X_train, axis=0)

    testNorm = (X_test - trainMean) / trainStd 
    trainNorm = (X_train - trainMean) / trainStd 

    
    innerK = 10
    lambda_interval = np.logspace(-8, 2, 25)


    inner_accuracy = np.zeros(len(lambda_interval))
    inner_model = []
    gen_e_lambda = np.zeros(len(lambda_interval))

    for j, lamb in enumerate(lambda_interval):

        
        inner_kfold = (KFold(n_splits=innerK, shuffle=True, random_state=0) 
                if not use_stratified 
                else  StratifiedKFold(n_splits=innerK, shuffle=True, random_state=0))

        gen_e = 0
        
        model = LogisticRegression(penalty="l2",solver="saga",max_iter=5000, C=1/lamb, random_state=0)
        scaler = StandardScaler()
        
        pipe = Pipeline(steps=[('scaler', scaler), ('model', model)])
        res = cross_validate(pipe, trainNorm, y_train, cv=inner_kfold, n_jobs=8,groups=y_train)
        gen_e_lambda[j] = np.mean(res['test_score'])
        # for i_train_idx, i_val_idx in inner_kfold.split(X_train,y_train):
        #     iX_train = X_train[i_train_idx]
        #     iX_test = X_train[i_val_idx]
        #     iy_train = y_train[i_train_idx]
        #     iy_test = y_train[i_val_idx]

        #     iX_trainStd = np.std(iX_train, axis=0)
        #     iX_trainMean = np.mean(iX_train, axis=0)

        #     iX_test = (iX_test - iX_trainMean) / iX_trainStd 
        #     iX_train = (iX_train - iX_trainMean) / iX_trainStd 


        #     model = LogisticRegression(penalty="l2",solver="lbfgs",max_iter=50000, C=1/lamb, random_state=0)
        #     model.fit(iX_train, iy_train)
        #     y_pred = model.predict(iX_test)
        #     gen_e += sum(y_pred != iy_test)/len(iy_test)



            
        
        #print("prediction: ", prediction, "lambda: {:.10f}".format(lamb))

    
    
    model_index = np.argmax(gen_e_lambda)
    best_lambda = lambda_interval[model_index]
    best_model = LogisticRegression(penalty="l2",solver="lbfgs", max_iter=50000, C=1/best_lambda, random_state=0)
    
    trainStd = np.std(X_train, axis=0)
    trainMean = np.mean(X_train, axis=0)

    testNorm = (X_test - trainMean) / trainStd 
    trainNorm = (X_train - trainMean) / trainStd 

    best_model.fit(trainNorm, y_train)
    y_pred = best_model.predict(testNorm)
    prediction = sum(y_pred != y_test)/len(y_test)
    
    print("Fold:", i, ": generalized Error: {:.5f}".format(gen_e_lambda[model_index]), ": validation: {:.5f}".format(prediction),  ": lambda: {:.5f}".format(lambda_interval[model_index]))
    best_models.append([best_model, gen_e_lambda[model_index] , lambda_interval[model_index]])


        

