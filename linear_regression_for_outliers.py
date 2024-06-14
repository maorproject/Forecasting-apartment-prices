######-- model 1 - linear regresssion model - drop outliers--######
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from model import *


def outliers_detection():
    """
    The funciton import the row dataset, predict price using simple linear regression and save the dataset with the predicte price
    the aim is to detect (manually) outliers
    """
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

    #split to x and y
    x=encoded_df.drop(columns=['Apartment_Price'])
    y=encoded_df["Apartment_Price"]

    standard_scaler = StandardScaler()
    scaled_x = pd.DataFrame(standard_scaler.fit_transform(x),columns=x.columns)
    ## linear regression  - find outliers 
    reg=LinearRegression()
    reg.fit(scaled_x,y)
    y_score=reg.predict(scaled_x)

    price_validation_df=df
    price_validation_df["predicted_price_filter_outliers"]=y_score
    price_validation_df["ratio predicted and real"]=price_validation_df["predicted_price_filter_outliers"]/price_validation_df["Apartment_Price"]
    price_validation_df.to_csv("validation_price_v2.csv")

    return print("new dataset with predcited price (for outlier detection) was saved as validation_price_v2")
