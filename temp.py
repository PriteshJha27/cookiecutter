import pandas as pd


class Preprocessor:
    def __init__(self):
        pass

    def filter_tables(self, df_table, list_of_tables):
        """
        Filters the table metadata to include only the relevant tables.
        """
        print("Filtering tables based on the list of tables...")
        filtered_table = df_table[df_table["table_name"].isin(list_of_tables)]
        return filtered_table

    def merge_metadata(self, df_table, df_col, df_cru_meta, df_proprietary_tables):
        """
        Merges table, column, and additional metadata into a unified DataFrame.
        """
        print("Merging table and column metadata...")
        merged_df = df_table.merge(df_col, on="table_name", how="outer")
        
        # Merging with CRU metadata and proprietary tables
        print("Adding CRU metadata and proprietary tables...")
        additional_metadata = pd.concat([df_proprietary_tables, df_cru_meta], axis=0, ignore_index=True)
        combined_df = pd.concat([merged_df, additional_metadata], axis=0, ignore_index=True)
        
        return combined_df

    def process_key_columns(self, df, key_cols_column="key_cols", partition_cols_column="partition_cols"):
        """
        Processes key and partition columns, creating a unified representation.
        """
        print("Processing key and partition columns...")
        key_cols = df[key_cols_column].fillna("").apply(lambda x: x.split(","))
        partition_cols = df[partition_cols_column].fillna("").apply(lambda x: x.split(","))
        
        # Combine key and partition columns
        keys_df = pd.concat([key_cols, partition_cols], axis=1, ignore_index=True)
        keys_df.columns = ["key_cols", "partition_cols"]
        df_keys = pd.concat([df, pd.DataFrame(keys_df)], axis=1)
        
        return df_keys

    def merge_foreign_keys(self, df_keys, df_foreign_keys):
        """
        Adds foreign key information by merging with foreign keys metadata.
        """
        print("Merging with foreign key metadata...")
        foreign_keys_combined = df_keys.merge(df_foreign_keys, how="left", left_on="table_name", right_on="table_name")
        return foreign_keys_combined

    def process_foreign_key_columns(self, foreign_keys_df):
        """
        Processes foreign keys, creating a dictionary mapping table-column relationships.
        """
        print("Processing foreign key table-column dictionary...")
        foreign_key_selected = foreign_keys_df[["table1", "column1", "table2", "column2"]]
        foreign_key_selected.columns = ["table", "column", "foreign_table", "foreign_column"]
        
        # Create a foreign key dictionary
        foreign_key_dict = (
            foreign_key_selected.groupby("table")["column"]
            .apply(list)
            .to_dict()
        )
        
        return foreign_key_dict
