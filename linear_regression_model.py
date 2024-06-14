import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_squared_error
from model import *


def linear_regression_model():

    #save the data in df
    df = import_dataset()
    

    # Select the columns to be encoded
    columns_to_encode = ['Neighborhood', 'Street', 'Month']

    # Perform one-hot encoding
    encoded_df = pd.get_dummies(df, columns=columns_to_encode)
    encoded_df=encoded_df.replace(False,0)
    encoded_df=encoded_df.replace(True,1)

    # Now the DataFrame 'encoded_df' contains the one-hot encoded columns

    encoded_df=encoded_df.drop(columns=["Date"])
    encoded_df = encoded_df.astype(int)#ממיר כל נתון לסוג int
    print(encoded_df.head())#return the first 5 rows of the data and the desteminisions of the data
   
    #split the data into train, validation and test
    model_df=encoded_df
    masked_train=model_df.loc[:,'Year']<=2021
    masked_test=model_df.loc[:,'Year']>=2022
    train,test=model_df.loc[masked_train,:],model_df.loc[masked_test,:]
    print(f"train size : {len(train)}  \n test size : {len(test)}")


    # define x and y variables
    x_train,y_train=train.drop(columns=['Apartment_Price']),train["Apartment_Price"]
    x_test,y_test=test.drop(columns=['Apartment_Price']),test["Apartment_Price"]

    #build linear regression base model 
    reg=Lasso(alpha=1)
    reg.fit(x_train,y_train)
    y_prediction_train=reg.predict(x_train)
    y_prediction_test=reg.predict(x_test)

    ## use 2 metrics rmse + r2 + add it to dictionary result={{base score: r2=1,mse=0.8,}}
    summary={}
    summary["base_model"]={}
    summary["base_model"]["train"]={}
    summary["base_model"]["test"]={}
    summary["base_model"]["train"]["rmse"]=mean_squared_error(y_train,y_prediction_train,squared=False)
    summary["base_model"]["test"]["rmse"]=mean_squared_error(y_test,y_prediction_test,squared=False)

    summary["base_model"]["train"]["r2"]=r2_score(y_train,y_prediction_train)
    summary["base_model"]["test"]["r2"]=r2_score(y_test,y_prediction_test)
    
    print(summary)

    return