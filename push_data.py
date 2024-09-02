from exception.exception import NetworkSecurityException
from logger.logger import logging
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import sys
import pandas as pd
import json


load_dotenv()



MONGO_URL = os.getenv("MONGO_DB_URL")
class ExportDataToMongo:
    def __init__(self) -> None:
        self.client = MongoClient(MONGO_URL)
    
    def ping(self):
        try:
            self.client.admin.command("ping")
            logging.info("successfully pinged MongoDB!!")
        except Exception as e:
            logging.error("Unable to ping MongoDB!!")
            raise NetworkSecurityException(e,sys)
        
    def csv_to_json_converter(self, filepath):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def pushing_data_to_mongodb(self, database, collection, records):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            self.database = self.client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def start_pushing(self, filepath, database, collection):
        try:
            self.ping()
            records = self.csv_to_json_converter(filepath=filepath)
            length_of_pushed_data = self.pushing_data_to_mongodb(database=database,
                                                                collection=collection, records=records)
            logging.info(f"successfully pushed data to MongoDB!! -- {length_of_pushed_data}")
        except Exception as e:
            logging.error("Error while pushing data to MongoDB!!")
            raise NetworkSecurityException(e,sys)

if __name__ == "__main__":
    test = ExportDataToMongo()
    test.start_pushing(filepath=r"C:\Users\mrsin\OneDrive\Desktop\NetworkSecurityPhisingDetection_MLOPS_END_TO_END\notebook\NetworkData.csv",
                       database="NetworkSecurity",
                       collection="data")

