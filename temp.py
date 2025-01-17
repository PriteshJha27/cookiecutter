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
