
# router/router.py
from src.indexing.data_loader import DataLoader
from src.indexing.preprocessing import Preprocessor
from src.indexing.schema_linking import SchemaLinker
from src.indexing.save_index_vectorDB import IndexSaver

def router():
    # Starting Indexing Pipeline
    print("=== Starting Indexing Pipeline ===")
    
    try:
        # Step 1: Load Data
        data_loader = DataLoader()
        print("Step 1: Loading data...")
        df_table, df_column, df_cru_meta, df2, df_foreign_keys = data_loader.load()

        # Step 2: Preprocess Data
        preprocessor = Preprocessor()
        print("Step 2: Preprocessing data...")
        processed_data = preprocessor.process(df_table, df_column, df_cru_meta, df2, df_foreign_keys)

        # Step 3: Schema Linking
        schema_linker = SchemaLinker()
        print("Step 3: Linking schema...")
        schema_details = schema_linker.link(processed_data)

        # Step 4: Save Index
        index_saver = IndexSaver()
        print("Step 4: Saving index...")
        index_saver.save(schema_details)

        print("=== Indexing Pipeline Completed Successfully ===")

    except Exception as e:
        print(f"Error in Indexing Pipeline: {str(e)}")
