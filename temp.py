import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from config import global_config


class DataLoader:
    def __init__(self):
        # Load environment variables from `.env` file
        load_dotenv(find_dotenv())
        self.asset_path = global_config["paths"]["asset_path"]

    def load_parquet_files(self):
        """
        Loads table and column metadata from parquet files.
        """
        print("Loading parquet files...")
        try:
            df_table = pd.read_parquet(os.path.join(self.asset_path, "cstone_table_metadata.parquet"))
            df_column = pd.read_parquet(os.path.join(self.asset_path, "cstone_column_metadata.parquet"))
            return df_table, df_column
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading parquet files: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error while loading parquet files: {e}")

    def load_excel_files(self):
        """
        Loads Excel files for CRU metadata and proprietary tables.
        """
        print("Loading Excel files...")
        try:
            df_cru_meta = pd.read_excel(os.path.join(self.asset_path, "CRU_meta.xlsx"))
            df_proprietary_tables = pd.read_excel(os.path.join(self.asset_path, "proprietary_tables.xlsx"))
            return df_cru_meta, df_proprietary_tables
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading Excel files: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error while loading Excel files: {e}")

    def load_csv_files(self):
        """
        Loads the foreign key mapping from a CSV file.
        """
        print("Loading CSV files...")
        try:
            df_foreign_keys = pd.read_csv(os.path.join(self.asset_path, "foreign_key.csv"))
            return df_foreign_keys
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading CSV files: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error while loading CSV files: {e}")

    def load_all_data(self):
        """
        Loads all required data files (parquet, Excel, and CSV).
        """
        print("Loading all data...")
        df_table, df_column = self.load_parquet_files()
        df_cru_meta, df_proprietary_tables = self.load_excel_files()
        df_foreign_keys = self.load_csv_files()

        return {
            "table_metadata": df_table,
            "column_metadata": df_column,
            "cru_metadata": df_cru_meta,
            "proprietary_tables": df_proprietary_tables,
            "foreign_keys": df_foreign_keys,
        }
