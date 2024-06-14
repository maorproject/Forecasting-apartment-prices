import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
from graphs import *



def evaluation_model(results,result_test,history,model,result_model_df):

    #1. save the result to csv
    result_model_df.to_csv("D:/project_visual_studio_code/result_model_df.csv")

    #### visualisation

    #2. architecture of the model
    plot_model(model, to_file="D:/project_visual_studio_code/model_plot.png", show_shapes=True, show_layer_names=True)

    #3. create figure with 2 plots
    price_compare = result_test[["Apartment_Price", "predicted_price"]]
    # Define the range for the ideal prediction line based on the actual data
    min_price = min(price_compare["Apartment_Price"].min(), price_compare["predicted_price"].min())
    max_price = max(price_compare["Apartment_Price"].max(), price_compare["predicted_price"].max())
    x = np.linspace(min_price, max_price, 100)
    y = x
    loss = np.log(history.history['loss'][:200])
    epochs = range(1, len(loss) + 1)
    # Create a DataFrame for the loss data
    data_loss = pd.DataFrame({"Epochs": epochs, "Log Loss": loss})
    # Plot the loss over epochs

    fig,axs=plt.subplots(1,2,figsize=(15,5))    # Plot the ideal prediction line

    # Plot 3.1: Comparison of actual and predicted prices
    sns.lineplot(x=x, y=y, color='red', label='Ideal Prediction', ax=axs[0])
    sns.scatterplot(x=price_compare["Apartment_Price"], y=price_compare["predicted_price"], label='Predicted vs Actual', ax=axs[0])
    axs[0].set_xlabel('Actual Apartment Price')
    axs[0].set_ylabel('Predicted Apartment Price')
    axs[0].set_title('Comparison of Actual and Predicted Apartment Prices')
    axs[0].legend()

    # Plot 3.2: Logarithm of training MSE loss over epochs
    sns.lineplot(data=data_loss, x='Epochs', y='Log Loss', ax=axs[1])
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Log MSE Loss')
    axs[1].set_title('Logarithm of Training MSE Loss over Epochs')

    # Display the plots
    plt.tight_layout()
    plt.show()


    #print evaluation metrics
    rmse_train=results["dl_reg"]["train"]["rmse"]
    rmse_test=results["dl_reg"]["test"]["rmse"]
    r2_train=results["dl_reg"]["train"]["r2"]
    r2_test=results["dl_reg"]["test"]["r2"]


    print(f"Train:  rmse: {rmse_train}, r2 score: {r2_train}")

    print(f"Test:   rmse: {rmse_test}, r2 score: {r2_test}")
