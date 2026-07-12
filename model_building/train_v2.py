
# =========================
# Imports
# =========================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import xgboost as xgb
import mlflow

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


# =========================
# Config
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_REPO = "bhattrushikesh97/tourism-mlops-app"
MODEL_REPO = "bhattrushikesh97/tourism-mlops-app"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism Package Prediction")

api = HfApi(token=HF_TOKEN)


# =========================
# Load Dataset from HF
# =========================
def fetch_data(file_name):
    return pd.read_csv(
        hf_hub_download(
            repo_id=DATA_REPO,
            filename=file_name,
            repo_type="dataset",
            token=HF_TOKEN
        )
    )


def load_all_data():
    X_train = fetch_data("Xtrain.csv")
    X_test = fetch_data("Xtest.csv")
    y_train = fetch_data("ytrain.csv").squeeze()
    y_test = fetch_data("ytest.csv").squeeze()

    print("Data fetched from Hugging Face successfully")
    return X_train, X_test, y_train, y_test


# =========================
# Feature Groups
# =========================
def get_feature_lists():

    categorical_cols = [
        "TypeofContact", "CityTier", "Occupation", "Gender",
        "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
        "Passport", "PitchSatisfactionScore", "OwnCar", "Designation"
    ]

    numerical_cols = [
        "Age", "DurationOfPitch", "NumberOfPersonVisiting",
        "NumberOfFollowups", "NumberOfTrips",
        "NumberOfChildrenVisiting", "MonthlyIncome"
    ]

    return numerical_cols, categorical_cols


# =========================
# Preprocessing
# =========================
def create_transformer(num_cols, cat_cols):

    transformer = ColumnTransformer([
        ("num_block", StandardScaler(), num_cols),
        ("cat_block", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return transformer


# =========================
# Model Pipeline
# =========================
def build_model_pipeline(transformer, weight):

    model = xgb.XGBClassifier(
        scale_pos_weight=weight,
        random_state=42,
        eval_metric="logloss"
    )

    pipe = Pipeline([
        ("feature_engineering", transformer),
        ("classifier", model)
    ])

    return pipe


# =========================
# Training Function
# =========================
def run_training(X_train, y_train, X_test, y_test, pipeline):

    param_grid = {
        "classifier__n_estimators": [60, 80],
        "classifier__max_depth": [3, 4],
        "classifier__learning_rate": [0.03, 0.05],
        "classifier__colsample_bytree": [0.5, 0.6],
        "classifier__reg_lambda": [0.5, 1]
    }

    with mlflow.start_run():

        tuner = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1
        )

        tuner.fit(X_train, y_train)

        # Log all runs
        for idx, params in enumerate(tuner.cv_results_["params"]):
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric(
                    "cv_mean_score",
                    tuner.cv_results_["mean_test_score"][idx]
                )

        best_model = tuner.best_estimator_

        mlflow.log_params(tuner.best_params_)

        # =====================
        # Evaluation
        # =====================
        threshold = 0.45

        train_prob = best_model.predict_proba(X_train)[:, 1]
        test_prob = best_model.predict_proba(X_test)[:, 1]

        train_pred = (train_prob > threshold).astype(int)
        test_pred = (test_prob > threshold).astype(int)

        train_metrics = classification_report(y_train, train_pred, output_dict=True)
        test_metrics = classification_report(y_test, test_pred, output_dict=True)

        mlflow.log_metrics({
            "train_acc": train_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_precision": test_metrics["1"]["precision"],
            "test_recall": test_metrics["1"]["recall"],
            "test_f1": test_metrics["1"]["f1-score"]
        })

        # Save Model
        model_file = "Tourism_Package_Prediction.joblib"
        joblib.dump(best_model, model_file)

        mlflow.log_artifact(model_file, artifact_path="model")

        print("Model saved locally")

        return model_file


# =========================
# Upload to Hugging Face
# =========================
def upload_model(model_path):

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print("Model repo exists")
    except RepositoryNotFoundError:
        print("Creating new HF model repo...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO,
        repo_type="model"
    )

    print("Model uploaded to Hugging Face")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_all_data()

    num_cols, cat_cols = get_feature_lists()

    # Handle imbalance
    class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    transformer = create_transformer(num_cols, cat_cols)

    pipeline = build_model_pipeline(transformer, class_weight)

    model_path = run_training(X_train, y_train, X_test, y_test, pipeline)

    upload_model(model_path)

    print("Training pipeline completed successfully 🚀")
