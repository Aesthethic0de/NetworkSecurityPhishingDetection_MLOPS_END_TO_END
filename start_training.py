from exception.exception import NetworkSecurityException
from logger.logger import logging
import os
import sys
from pipeline.training_pipeline import TrainingPipeline
from pipeline.batch_prediction import start_batch_prediction

def start_training():
    try:
        logging.info("training has started!!")
        model_training = TrainingPipeline()
        model_training.run_pipeline()

    except Exception as e:
        logging.error(f"training has not started due to {str(e)}")
        raise NetworkSecurityException(e, sys)
    
if __name__ == "__main__":
    # start_training()
    start_batch_prediction(input_file_path=r"C:\Users\mrsin\OneDrive\Desktop\NetworkSecurityPhisingDetection_MLOPS_END_TO_END\Artifacts\09_03_2024_11_50_38\data_ingestion\ingested\test.csv")