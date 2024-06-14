import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from graphs import *




def model(layer_params,learning_rate,epoch_number):

    #1. import dataset
    df=import_dataset()

    #2. data processing
    scaled_x_train,y_train,scaled_x_test,y_test,masked_train,masked_test=data_processing(df)

    #3. build the model architecture
    model=deep_learning_reg(scaled_x_train,y_train,scaled_x_test,y_test,layer_params,learning_rate)
    
    #4. train and test the model
    results,result_model_df,history,result_test,model=fit_predict_model(model,scaled_x_train,y_train,scaled_x_test,y_test,df,masked_train,masked_test,epoch_number)
    
    # 5.evaluate the model
    evaluation_model(results,result_test,history,model,result_model_df)

    return 




def import_dataset():
    df = pd.read_csv(r'D:\project_visual_studio_code\Final_Project_Apartment_Prices.csv', encoding='unicode_escape')
    return df

def data_processing(df):
        
    # Select the columns to be encoded
    columns_to_encode = ['Neighborhood', 'Street', 'Month']

    # one-hot encoding
    encoded_df = pd.get_dummies(df, columns=columns_to_encode)
    encoded_df=encoded_df.replace(False,0)
    encoded_df=encoded_df.replace(True,1)

    #remove irrelvent columns
    encoded_df=encoded_df.drop(columns=["Date"])

    #cahnge datatype to int
    encoded_df = encoded_df.astype(int)#ממיר כל נתון לסוג flaot

    #split to train and test datasets
    masked_train=encoded_df.loc[:,'Year']<=2021
    masked_test=encoded_df.loc[:,'Year']>=2022
    train,test=encoded_df.loc[masked_train,:],encoded_df.loc[masked_test,:]
    print(f"train size : {len(train)}  \n test size : {len(test)}")

    # define x and y variables
    x_train,y_train=train.drop(columns=['Apartment_Price']),train["Apartment_Price"]
    x_test,y_test=test.drop(columns=['Apartment_Price']),test["Apartment_Price"]

    #scaling 
    standard_scaler = StandardScaler()
    scaled_x_train = pd.DataFrame(standard_scaler.fit_transform(x_train),columns=x_train.columns)
    scaled_x_test = pd.DataFrame(standard_scaler.transform(x_test),columns=x_test.columns)

    return scaled_x_train,y_train,scaled_x_test,y_test,masked_train,masked_test

def deep_learning_reg(scaled_x_train,y_train,scaled_x_test,y_test,layer_params,learning_rate):
    """
    the function create dynamically deep learning regression model 
    layer_params : list contains dynamicaly size of the archtecture layers
    
    """
    input_shape=scaled_x_train.shape[1]
    model = Sequential()
    model.add(Dense(layer_params[0],activation='relu',input_shape=(input_shape,),kernel_regularizer=l1_l2(0.0001)))
    model.add(Dropout(0.1))
    model.add(Dense(layer_params[1],activation='relu',kernel_regularizer=l1_l2(0.0001)))
    model.add(Dropout(0.1))

    for param in layer_params[2:]:
        model.add(Dense(param,activation='relu',kernel_regularizer=l1_l2(0.0001)))
        model.add(Dropout(0.1))
    model.add(Dense(1))
    

    opt= Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,loss='mse')
    model.summary()


    return model

def fit_predict_model(model,scaled_x_train,y_train,scaled_x_test,y_test,df,masked_train,masked_test,epoch_number):
    
    results={}
    #fit the model
    start_train=datetime.datetime.now()
    history=model.fit(scaled_x_train,y_train,epochs=epoch_number,batch_size=32,verbose=0)
    end_train=datetime.datetime.now()
    training_time=end_train-start_train
    results["train time"]=training_time

    #predict y train
    y_prediction_train=model.predict(scaled_x_train)
    results["y_train_prediction"]=y_prediction_train

    #preedict y test
    y_prediction_test=model.predict(scaled_x_test)
    test_time=(end_train-start_train)/len(y_prediction_test)
    results["predcition time (sec/sample)"]=test_time
    results["y_test_prediction"]=y_prediction_test

    #store evaluation metrics
    results={}
    results["dl_reg"]={}
    results["dl_reg"]["train"]={}
    results["dl_reg"]["test"]={}
    results["dl_reg"]["train"]["rmse"]=np.round(mean_squared_error(y_train,y_prediction_train,squared=False),0)
    results["dl_reg"]["test"]["rmse"]=np.round(mean_squared_error(y_test,y_prediction_test,squared=False),0)

    results["dl_reg"]["train"]["r2"]=np.round(r2_score(y_train,y_prediction_train),2)
    results["dl_reg"]["test"]["r2"]=np.round(r2_score(y_test,y_prediction_test),2)


    # create csv dataset with the predicted prices
    result_test=df.loc[masked_test,:]
    result_test["predicted_price"]=y_prediction_test
    result_train=df.loc[masked_train,:]
    result_train["predicted_price"]=y_prediction_train
    result_model_df=pd.concat([result_train,result_test])
    
    return results, result_model_df,history,result_test,model