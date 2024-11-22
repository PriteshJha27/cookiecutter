```python
from langchain.tools import Tool
from dotenv import load_dotenv
from typing import List
from typing_extensions import Doc, Annotated, Iterable, Literal
from langchain.core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import faiss
from pathlib import Path
import json
import sqlite3
from sqlalchemy import create_engine, text

@dataclass
class TableSchema:
    name: str
    description: str
    columns: Dict[str, str]
    relationships: Dict[str, str]
    key_metrics: List[str]
    sample_queries: List[str]

class LocalEmbeddingModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)

class UnderwritingRAG:
    def __init__(self, model_path: str):
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.schema_store = {}
        self.dataframes = {}
        self._initialize_schemas()

    def save_vectorstore(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(save_path / 'index.faiss'))

        schema_data = {}
        for name, schema in self.schema_store.items():
            schema_data[name] = {
                'name': schema.name,
                'description': schema.description,
                'columns': schema.columns,
                'relationships': schema.relationships,
                'key_metrics': schema.key_metrics,
                'sample_queries': schema.sample_queries
            }
        
        with open(save_path / 'schema_store.json', 'w') as f:
            json.dump(schema_data, f, indent=2)

        print(f"Vectorstore saved to {save_path}")

    @classmethod
    def load_vectorstore(cls, model_path: str, load_path: str):
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"Path {load_path} does not exist")

        instance = cls(model_path)

        index_path = load_path / 'index.faiss'
        if index_path.exists():
            instance.index = faiss.read_index(str(index_path))
        else:
            raise ValueError("No FAISS index found")

        schema_path = load_path / 'schema_store.json'
        if schema_path.exists():
            with open(schema_path) as f:
                schema_data = json.load(f)
            
            instance.schema_store = {
                name: TableSchema(
                    name=data['name'],
                    description=data['description'],
                    columns=data['columns'],
                    relationships=data['relationships'],
                    key_metrics=data['key_metrics'],
                    sample_queries=data['sample_queries']
                )
                for name, data in schema_data.items()
            }

        print(f"Vectorstore loaded from {load_path}")
        return instance
```
def _initialize_schemas(self):
       self.schema_store = {
           "risk_assessment": TableSchema(
               name="Risk Assessment",
               description="Contains risk assessment details and recommendations for borrowers",
               columns={
                   "assessment_id": "Unique identifier for risk assessment",
                   "borrower_id": "Foreign key to Borrower Profile",
                   "risk_score": "Overall risk score (0-100)",
                   "market_risk": "Market risk level (Low/Moderate/High)",
                   "credit_risk": "Credit risk level (Low/Moderate/High)", 
                   "operational_risk": "Operational risk level (Low/Moderate/High)",
                   "regulatory_risk": "Regulatory risk level (Low/Moderate/High)",
                   "recommendation": "Final recommendation (Approve/Reject/Further Review)",
                   "assessment_date": "Date of risk assessment",
                   "reviewed_by": "Name of reviewing officer",
                   "next_review_date": "Next scheduled review date"
               },
               relationships={
                   "borrower_id": "borrower_profile.borrower_id"
               },
               key_metrics=[
                   "risk_score",
                   "recommendation",
                   "risk_levels"
               ],
               sample_queries=[
                   "What is the risk assessment for borrower TCS-1?",
                   "Show all high credit risk assessments",
                   "List rejected applications with risk scores above 50"
               ]
           ),
           
           "borrower_profile": TableSchema(
               name="Borrower Profile", 
               description="Contains company profile and credit rating information",
               columns={
                   "borrower_id": "Unique identifier for borrower",
                   "company_name": "Name of the company",
                   "industry_sector": "Industry sector of the company", 
                   "incorporation_date": "Date of incorporation",
                   "headquarters_location": "Company headquarters location",
                   "business_structure": "Legal structure of business",
                   "ownership": "Ownership details",
                   "credit_rating": "Current credit rating",
                   "last_rating_update": "Last credit rating update date"
               },
               relationships={},
               key_metrics=[
                   "credit_rating",
                   "business_age",
                   "rating_trend"
               ],
               sample_queries=[
                   "What is TCS-1's credit rating?",
                   "List all AAA rated borrowers",
                   "Show recently incorporated companies"
               ]
           ),
           
           "credit_history": TableSchema(
               name="Credit History",
               description="Loan and credit history details",
               columns={
                   "credit_id": "Unique identifier for credit record",
                   "borrower_id": "Foreign key to Borrower Profile",
                   "credit_type": "Type of credit facility",
                   "loan_amount": "Original loan amount",
                   "loan_issued_date": "Date loan was issued",
                   "loan_maturity_date": "Loan maturity date",
                   "repayment_status": "Current repayment status",
                   "default_history": "Previous default history",
                   "interest_rate": "Current interest rate",
                   "outstanding_balance": "Current outstanding balance"
               },
               relationships={
                   "borrower_id": "borrower_profile.borrower_id"
               },
               key_metrics=[
                   "default_rate",
                   "average_loan_amount", 
                   "repayment_performance"
               ],
               sample_queries=[
                   "Show default history for borrower TCS-1",
                   "List all active loans above $40000",
                   "What is the average interest rate for term loans?"
               ]
           ),

           "financial_ratios": TableSchema(
               name="Financial Ratios",
               description="Key financial ratios and performance metrics",
               columns={
                   "ratio_id": "Unique identifier for ratio record",
                   "borrower_id": "Foreign key to Borrower Profile",
                   "reporting_period": "Period of financial reporting",
                   "current_ratio": "Current assets / Current liabilities",
                   "quick_ratio": "Quick assets / Current liabilities",
                   "debt_to_equity": "Total debt / Total equity",
                   "interest_coverage_ratio": "EBIT / Interest expenses",
                   "gross_margin_ratio": "Gross profit / Revenue",
                   "return_on_assets": "Net income / Total assets",
                   "return_on_equity": "Net income / Shareholder equity"
               },
               relationships={
                   "borrower_id": "borrower_profile.borrower_id"
               },
               key_metrics=[
                   "solvency_metrics",
                   "profitability_metrics",
                   "efficiency_metrics"
               ],
               sample_queries=[
                   "What are the current ratios below 1.5?",
                   "Show companies with high debt-to-equity",
                   "List top 10 by return on equity"
               ]
           ),

           "financial_statements": TableSchema(
               name="Financial Statements",
               description="Detailed financial statement data",
               columns={
                   "statement_id": "Unique identifier for financial statement",
                   "borrower_id": "Foreign key to Borrower Profile",
                   "reporting_period": "Period of financial reporting",
                   "total_assets": "Total assets value",
                   "total_liabilities": "Total liabilities value",
                   "shareholder_equity": "Total shareholder equity",
                   "total_revenue": "Total revenue",
                   "net_income": "Net income",
                   "cash_flow_operations": "Cash flow from operations",
                   "cash_flow_investments": "Cash flow from investments",
                   "cash_flow_financing": "Cash flow from financing",
                   "gross_margin": "Gross margin percentage",
                   "operating_margin": "Operating margin percentage",
                   "net_profit_margin": "Net profit margin percentage"
               },
               relationships={
                   "borrower_id": "borrower_profile.borrower_id"
               },
               key_metrics=[
                   "size_metrics",
                   "profitability_metrics",
                   "cash_flow_metrics"
               ],
               sample_queries=[
                   "Show companies with negative operating cash flow",
                   "List top 10 by revenue",
                   "What is the average profit margin?"
               ]
           )
       }

def create_schema_embedding(self, table_name: str) -> str:
       schema = self.schema_store[table_name]
       
       schema_text = f"Table: {schema.name}\n"
       schema_text += f"Description: {schema.description}\n\n"
       
       schema_text += "Columns:\n"
       for col, desc in schema.columns.items():
           schema_text += f"- {col}: {desc}\n"
       
       if schema.relationships:
           schema_text += "\nRelationships:\n"
           for col, rel in schema.relationships.items():
               schema_text += f"- {col} relates to {rel}\n"
       
       schema_text += "\nKey Metrics:\n"
       for metric in schema.key_metrics:
           schema_text += f"- {metric}\n"
       
       schema_text += "\nTypical Queries:\n"
       for query in schema.sample_queries:
           schema_text += f"- {query}\n"
           
       return schema_text

   def add_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
       self.dataframes[table_name] = df
       
       if self.index is None:
           schema_text = self.create_schema_embedding(table_name)
           embedding = self.embedding_model.encode([schema_text])[0]
           self.index = faiss.IndexFlatL2(embedding.shape[0])
           self.index.add(np.array([embedding]).astype('float32'))
       else:
           schema_text = self.create_schema_embedding(table_name)
           embedding = self.embedding_model.encode([schema_text])[0]
           self.index.add(np.array([embedding]).astype('float32'))

   def query(self, query: str, k: int = 2) -> List[Dict]:
       query_vector = self.embedding_model.encode([query])
       query_vector = np.array(query_vector).astype('float32')
       
       distances, indices = self.index.search(query_vector, k)
       
       relevant_tables = []
       for idx, distance in enumerate(distances[0]):
           table_name = list(self.schema_store.keys())[indices[0][idx]]
           schema = self.schema_store[table_name]
           
           relevant_tables.append({
               'table_name': schema.name,
               'relevance_score': float(distance),
               'schema': schema,
               'sample_data': self.dataframes[table_name].head(3) if table_name in self.dataframes else None
           })
           
       return relevant_tables

class SQLiteQueryExecutor:
   def __init__(self):
       self.engine = create_engine('sqlite:///:memory:')
       
   def load_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
       for table_name, df in dataframes.items():
           df.to_sql(table_name.lower(), self.engine, if_exists='replace', index=False)
   
   def extract_sql_query(self, docstring: str) -> Optional[str]:
       pattern = r"'''[\s]*sql(.*?)'''+"
       matches = re.search(pattern, docstring, re.DOTALL)
       
       if not matches:
           pattern = r"```sql\s*(.*?)```"
           matches = re.search(pattern, docstring, re.DOTALL)
           if not matches:
               return None
       
       sql_query = matches.group(1).strip()
       return sql_query
   
   def execute_query(self, query: str) -> Tuple[pd.DataFrame, str]:
       try:
           df = pd.read_sql_query(text(query), self.engine)
           formatted_query = query.strip()
           return df, formatted_query
       except Exception as e:
           raise Exception(f"Error executing query: {str(e)}")

def format_query_results(results: List[Dict]) -> str:
   context = "Relevant tables for your query:\n\n"
   
   for result in results:
       context += f"Table: {result['table_name']}\n"
       context += f"Relevance Score: {result['relevance_score']:.4f}\n"
       schema = result['schema']
       
       context += "Key Columns:\n"
       for col, desc in schema.columns.items():
           context += f"- {col}: {desc}\n"
       
       if result['sample_data'] is not None:
           context += "\nSample Data:\n"
           context += result['sample_data'].to_string()
       
       context += "\n\n"
   
   return context

def initialize_rag_system(
   model_path: str,
   risk_df,
   borrower_df,
   credit_df,
   ratios_df,
   statements_df,
   save_path: str = None
):
   rag = UnderwritingRAG(model_path)
   
   rag.add_dataframe(risk_df, "risk_assessment")
   rag.add_dataframe(borrower_df, "borrower_profile")
   rag.add_dataframe(credit_df, "credit_history")
   rag.add_dataframe(ratios_df, "financial_ratios")
   rag.add_dataframe(statements_df, "financial_statements")
   
   if save_path:
       rag.save_vectorstore(save_path)
   
   return rag

def load_csv_files(csv_path: str) -> Dict[str, pd.DataFrame]:
   dataframes = {
       'risk_assessment': pd.read_csv(csv_path + "Risk_Assessment.csv"),
       'borrower_profile': pd.read_csv(csv_path + "Borrower_Profile.csv"),
       'credit_history': pd.read_csv(csv_path + "Credit_History.csv"),
       'financial_ratios': pd.read_csv(csv_path + "Financial_Ratios.csv"),
       'financial_statements': pd.read_csv(csv_path + "Financial_Statements.csv")
   }
   return dataframes

def retrieval_llm(query):
   results = rag.query(query)
   context = format_query_results(results)
   
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a SQL master expert capable of writing complex SQL query. "
                  "Based on the following underwriting data context {context}, answer the query as a SQL query based on the context provided."
                  "Please construct a SQL query using ONLY the context and the query provided above."
                  "When joining tables, employ type coercion to guarantee data type consistency for the join columns."
                  "\n\n"
                  "IMPORTANT: Use ONLY the column names mentioned in context. DO NOT USE any other column names outside of this."
                  "IMPORTANT: Associate column name mentioned in context only to the table_name specified under context."
                  "NOTE: Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed."
                  ),
       ("human", "{query}")
   ])

   chain = prompt | model | StrOutputParser()
   
   final_result = chain.invoke({
       "context": context,
       "query": query
   })
   
   return results, context, final_result

if __name__ == "__main__":
   load_dotenv()
   
   model_path = os.getenv("MODEL_PATH")
   csv_path = os.getenv("CSV_PATH")
   vectorstore_path = "vectorstores/underwriting"

   model = ChatAmexLlama(
       base_url=os.getenv("LLAMA_API_URL"),
       auth_url=os.getenv("LLAMA_AUTH_URL"),
       user_id=os.getenv("LLAMA_USER_ID"),
       pwd=os.getenv("LLAMA_PASSWORD"),
       cert_path=os.getenv("CERT_PATH")
   )

   # Either initialize new or load existing vectorstore
   if not Path(vectorstore_path).exists():
       # First time setup
       dataframes = load_csv_files(csv_path)
       rag = initialize_rag_system(
           model_path=model_path,
           risk_df=dataframes['risk_assessment'],
           borrower_df=dataframes['borrower_profile'],
           credit_df=dataframes['credit_history'],
           ratios_df=dataframes['financial_ratios'],
           statements_df=dataframes['financial_statements'],
           save_path=vectorstore_path
       )
   else:
       # Load existing vectorstore
       rag = UnderwritingRAG.load_vectorstore(
           model_path=model_path,
           load_path=vectorstore_path
       )
       dataframes = load_csv_files(csv_path)

   # Initialize SQLite executor
   executor = SQLiteQueryExecutor()
   executor.load_dataframes(dataframes)

   # Example query
   query = "Fetch basic borrower profile details for TCS"
   results, context, final_result = retrieval_llm(query)
   sql_query = executor.extract_sql_query(final_result)
   if sql_query:
       answer = executor.execute_query(sql_query)
       print(answer[0])
