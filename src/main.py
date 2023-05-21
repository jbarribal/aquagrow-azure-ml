import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
#from sklearn.model_selection import cross_val_score
#from sklearn.datasets import load_iris
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import classification_report, r2_score 
from sklearn.model_selection import train_test_split

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    aquagrow_df = pd.read_csv(args.data)

    mlflow.log_metric("num_samples", aquagrow_df.shape[0])
    mlflow.log_metric("num_features", aquagrow_df.shape[1] - 1)

    #Split train and test datasets
    train_df, test_df = train_test_split(
        aquagrow_df,
        test_size=args.test_train_ratio,
    )
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    # Extracting the label column
    y_train = train_df.pop("Weight")

    # convert the dataframe values to array
    X_train = train_df.values

    # Extracting the label column
    y_test = test_df.pop("Weight")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    # clf = GradientBoostingClassifier(
    #     n_estimators=args.n_estimators, learning_rate=args.learning_rate
    # )
    # clf.fit(X_train, y_train)

    #y_pred = clf.predict(X_test)

    rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr_model.fit(X_train, y_train)
    y_pred = rfr_model.predict(X_test)

    print(r2_score(y_test, y_pred))
    ###################
    #</train the model>
    ###################

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model= rfr_model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=rfr_model,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
