import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hyperopt import hp, fmin, tpe, Trials
import pickle
import os
import json

seed = 42


#####################################################
# * Function to set up XGboost hyperopt Experiment
####################################################


def xgboost_hyperopt(
    model_check_point, X_train, y_train, X_valid, y_valid, epochs=100, iterations=20
):
    ##* writing objective function

    def objective(params):
        model = xgb.XGBRegressor(
            n_estimators=epochs,  # Number of trees (epochs)
            learning_rate=params["learning_rate"],
            max_depth=int(params["max_depth"]),
            subsample=params["subsample"],
            gamma=params["gamma"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=int(params["min_child_weight"]),
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
        )

        # Train the model on the training data with early stopping
        eval_set = [(X_valid, y_valid)]
        model.fit(
            X_train,
            y_train,
            early_stopping_rounds=10,
            eval_metric="rmse",
            eval_set=eval_set,
            verbose=False,
        )

        # Get the best iteration based on early stopping
        best_iteration = model.best_iteration

        # Score the model on the validation data using the best iteration
        y_pred = model.predict(X_valid, ntree_limit=best_iteration)
        mse = mean_squared_error(y_valid, y_pred)
        score = mse

        return score

    ##* Define the search space for hyperparameters
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "subsample": hp.uniform("subsample", 0.7, 1.0),
        "gamma": hp.uniform("gamma", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1.0),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-10), np.log(1.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-10), np.log(1.0)),
    }

    ##* running hyperopt trials
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=iterations,  # Number of iterations
        trials=trials,
    )

    ##* outputting trial results

    print("Best Parameters: {}".format(best))

    with open((model_check_point + "/best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, default=int)

    return best


def xgboost_retrain_test(
    best_params,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    model_check_point,
    epochs=100,
):
    # Create an XGBoost model with the best hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=epochs,  # Choose an appropriate number of trees (epochs)
        learning_rate=best_params["learning_rate"],
        max_depth=int(best_params["max_depth"]),
        subsample=best_params["subsample"],
        gamma=best_params["gamma"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=int(best_params["min_child_weight"]),
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
    )

    # Train the model on the training data with early stopping
    eval_set = [(X_valid, y_valid)]
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=False,
    )

    # Get the best iteration based on early stopping
    best_iteration = model.best_iteration

    ##* saving the model
    file_name = os.path.join(model_check_point, "xgboost_model.pkl")
    # save
    pickle.dump(best_iteration, open(file_name, "wb"))

    # Score the model on the validation data using the best iteration
    y_pred = model.predict(X_test, ntree_limit=best_iteration)
    y_true = y_test

    r2_metric = r2_score(y_true, y_pred)
    print(f"R2 score: {r2_metric}")

    mse_score = mean_squared_error(y_true, y_pred, squared=True)
    print(f"MSE Score: {mse_score}")

    rmse_score = mean_squared_error(y_true, y_pred, squared=False)
    print(f"RMSE Score: {rmse_score}")

    mae_score = mean_absolute_error(y_true, y_pred)
    print(f"MAE Score: {mae_score}")

    score_dict = {
        "R2 Score": r2_metric,
        "MSE Score": mse_score,
        "RMSE Score": rmse_score,
        "MAE Score": mae_score,
    }

    with open((model_check_point + "/test_scores.json"), "w", encoding="utf-8") as f:
        json.dump(score_dict, f, default=int)

    df = pd.DataFrame({"actual_RT": y_true, "pred_RT": y_pred})
    print(df.head())
    pred_df_path = model_check_point + "/test_preds.csv"
    df.to_csv(pred_df_path, index=False)
    print(f"saved predictions to {pred_df_path}")

    print("\n*** Hyperparameter Optimization is done ***\n")
