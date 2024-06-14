from model import model
from linear_regression_for_outliers import outliers_detection
from linear_regression_model import linear_regression_model
from graphs import *
from model import *
from linear_regression_for_outliers import *
from linear_regression_model import *
from best_model_train_validation import *


def get_layer_parameters():
    """
    The function prompts the user to input the number of parameters for each of four layers in a model.
    
    Parameters: 
    None
    
    Output:
    Returns a tuple containing the number of parameters for each layer.
    """
    num_layers=int(input("Enter the number of layers you want to configure (3,4,5): "))
    while num_layers not in [3,4,5]:
        num_layers=int(input("Enter again the number of layers you want to configure (3,4,5): "))

    print("\nPlease enter the number of parameters for each layer:")
    parameters = []
    for i in range(1,num_layers+1):
        param = input(f"Enter the number of parameters for layer {i}: ")
        while not param.isdigit():
            print("Sorry, this is not a valid input. Please enter a positive integer.")
            param = input(f"Enter the number of parameters for layer {i} again: ")
        parameters.append(int(param))

    print("\nPlease enter the learning rate:")
    learning_rate =float( input())

    print("\nPlease enter the number of epoch:")
    epoch_number = int(input())



    return parameters,learning_rate,epoch_number

def main():
    print("Welcome to the Neural Network Configuration Tool")
    # Add more options or functionality here as needed
    while True:
        print("\nChoose an option:")
        print("1. Enter layer parameters")
        print("2. Get the best model")
        print("3. Detect outliers by linear regression")
        print("4. Get linear regression model results")
        print("5. Get the best model results when it's split into train and validation")
        print("6. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            layer_params ,learning_rate,epoch_number= get_layer_parameters()
            print("Layer parameters:", layer_params)
            model( layer_params ,learning_rate,epoch_number)
        if choice=="2":
            layer_params ,learning_rate,epoch_number=([256,256,256,32,256],0.0001,200)
            print(f"""the best model params:
                  layers: {layer_params}
                  learning rate: {learning_rate}
                  epoch numbers: {epoch_number}""")
            model( layer_params ,learning_rate,epoch_number)
        if choice=='3':
            outliers_detection()
        if choice=='4':
            linear_regression_model()
        if choice=='5':
            best_model_train_and_validation()
        elif choice == "6":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please select a valid option.")

if __name__=="__main__":
    main()