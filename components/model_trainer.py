import os
import sys

from exception.exception import NetworkSecurityException 
from logger.logger import logging

from entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from entity.config_entity import ModelTrainerConfig

from xgboost import XGBClassifier

from utils.ml_utils.model.estimator import NetworkModel
from utils.main_utils.utils import save_object,load_object
from utils.main_utils.utils import load_numpy_array_data
from utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.model_selection import GridSearchCV



class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def perform_hyper_parameter_tunig(self, model, param_grid, X_train, y_train):
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the model with the training data
        grid_search.fit(X_train, y_train)

        # Get the best estimator and parameters
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_

        return best_estimator, best_params
    

    def train_model(self,x_train,y_train):
        try:
            param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
            }
             # Initialize the XGBClassifier
            xgb_clf = XGBClassifier()

            # Initialize GridSearchCV with the model and parameter grid
            grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

            # Fit the model with the training data
            grid_search.fit(x_train, y_train)

            # Get the best estimator
            best_xgb_clf = grid_search.best_estimator_

            # Log the best parameters
            logging.info(f"Best Parameters: {grid_search.best_params_}")

            return best_xgb_clf
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise e
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                #raise Exception("Trained model is not good to provide expected accuracy")
                print("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)


            # Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            Network_Model = NetworkModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)