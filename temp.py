paths:
  data:
    table_metadata: "path/to/cstone_table_metadata.parquet"
    column_metadata: "path/to/cstone_column_metadata.parquet"
    cru_meta: "path/to/CRU_meta.xlsx"
    proprietary_tables: "path/to/proprietary_tables.xlsx"
    foreign_keys: "path/to/foreign_key.csv"
  vectorstore:
    save_path: "path/to/vectorstore/index.faiss"
  model:
    tokenizer: "path/to/tokenizer"
    model: "path/to/model"




# src/utils/config_loader.py
import yaml

class ConfigLoader:
    @staticmethod
    def load_config(config_path="config/config.yml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config



# src/indexing/data_loader.py
import pandas as pd
import os
from src.utils.config_loader import ConfigLoader

class DataLoader:
    def __init__(self):
        self.config = ConfigLoader.load_config()

    def load(self):
        print("Loading data from configuration paths...")

        # Load paths from config
        table_metadata_path = self.config["paths"]["data"]["table_metadata"]
        column_metadata_path = self.config["paths"]["data"]["column_metadata"]
        cru_meta_path = self.config["paths"]["data"]["cru_meta"]
        proprietary_tables_path = self.config["paths"]["data"]["proprietary_tables"]
        foreign_keys_path = self.config["paths"]["data"]["foreign_keys"]

        # Read files
        df_table = pd.read_parquet(table_metadata_path)
        df_column = pd.read_parquet(column_metadata_path)
        df_cru_meta = pd.read_excel(cru_meta_path)
        df2 = pd.read_excel(proprietary_tables_path)
        df_foreign_keys = pd.read_csv(foreign_keys_path)

        return df_table, df_column, df_cru_meta, df2, df_foreign_keys
