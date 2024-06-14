    ##------- model - deep learning regressor model - train and validation only--------##
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import r2_score, mean_squared_error
from model import *


def best_model_train_and_validation():
    #dataset
    df = import_dataset()

    print(df)

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


    #split to x and y
    x=encoded_df.drop(columns=['Apartment_Price'])
    y=encoded_df["Apartment_Price"]

    standard_scaler = StandardScaler()
    scaled_x = pd.DataFrame(standard_scaler.fit_transform(x),columns=x.columns)
    print(scaled_x.head())
    scaled_x["Year"]


    model_df=encoded_df
    masked_train=model_df.loc[:,'Year']<=2019
    masked_val=((model_df.loc[:,'Year']>2019)&(model_df.loc[:,'Year']<=2021))
    masked_test=model_df.loc[:,'Year']>=2022
    train,val,test=model_df.loc[masked_train,:],model_df.loc[masked_val,:],model_df.loc[masked_test,:]
    print(f"train size : {len(train)} \n validation size : {len(val)} \n test size : {len(test)}")

    # define x and y variables
    x_train,y_train=train.drop(columns=['Apartment_Price']),train["Apartment_Price"]
    x_val,y_val=val.drop(columns=['Apartment_Price']),val["Apartment_Price"]

    #scaling 
    standard_scaler = StandardScaler()
    scaled_x_train = pd.DataFrame(standard_scaler.fit_transform(x_train),columns=x_train.columns)
    scaled_x_val = pd.DataFrame(standard_scaler.transform(x_val),columns=x_val.columns)

    
    ##------- model 3 - deep learning regressor model --------##
    input_shape=scaled_x_train.shape[1]
    model = Sequential([
        Dense(256,activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)),
        Dropout(0.1),
        Dense(256,activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)),
        Dropout(0.1),
        Dense(256,activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)),
        Dropout(0.1),
        Dense(32,activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)),
        Dropout(0.1),
        Dense(256,activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)),
        Dropout(0.1),
        Dense(1)]
        )
    opt= Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss='mse')
    model.summary()
    history=model.fit(scaled_x_train,y_train,epochs=200,batch_size=32,verbose=0)
    y_prediction_train=model.predict(scaled_x_train)
    y_prediction_val=model.predict(scaled_x_val)

    summary={}
    summary["dl_reg"]={}
    summary["dl_reg"]["train"]={}
    summary["dl_reg"]["val"]={}
    summary["dl_reg"]["train"]["rmse"]=mean_squared_error(y_train,y_prediction_train,squared=False)
    summary["dl_reg"]["val"]["rmse"]=mean_squared_error(y_val,y_prediction_val,squared=False)

    summary["dl_reg"]["train"]["r2"]=r2_score(y_train,y_prediction_train)
    summary["dl_reg"]["val"]["r2"]=r2_score(y_val,y_prediction_val)

    print(summary)
    
    return