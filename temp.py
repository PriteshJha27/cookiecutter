import re
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
from typing import Optional, Tuple, Dict
import os

class SQLiteQueryExecutor:
    def __init__(self):
        """Initialize in-memory SQLite database"""
        # Create SQLite in-memory database
        self.engine = create_engine('sqlite:///:memory:')
        
    def load_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """
        Load DataFrames into SQLite database
        
        Args:
            dataframes: Dictionary of table_name: DataFrame pairs
        """
        # Load each DataFrame into SQLite
        for table_name, df in dataframes.items():
            # Convert DataFrame to SQL table
            df.to_sql(table_name.lower(), self.engine, if_exists='replace', index=False)
            
    def extract_sql_query(self, docstring: str) -> Optional[str]:
        """
        Extract SQL query from docstring between ''' sql ''' markers
        
        Args:
            docstring: Function docstring containing SQL query
        
        Returns:
            Extracted SQL query or None if not found
        """
        # Pattern to match SQL between ''' sql ''' markers
        pattern = r"'''[\s]*sql(.*?)'''+"
        
        # Find all matches using regex
        matches = re.findall(pattern, docstring, re.DOTALL)
        
        if not matches:
            return None
            
        # Get the first match and clean it
        sql_query = matches[0].strip()
        return sql_query

    def execute_query(self, query: str) -> Tuple[pd.DataFrame, str]:
        """
        Execute SQL query and return results
        
        Args:
            query: SQL query to execute
        
        Returns:
            Tuple of (result DataFrame, formatted query)
        """
        try:
            # Create DataFrame from query
            df = pd.read_sql_query(text(query), self.engine)
            
            # Format query for display
            formatted_query = query.strip()
            
            return df, formatted_query
            
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def process_function(self, func) -> Tuple[pd.DataFrame, str]:
        """
        Process function docstring to extract and execute SQL query
        
        Args:
            func: Function containing SQL query in docstring
        
        Returns:
            Tuple of (result DataFrame, formatted query)
        """
        docstring = func.__doc__
        if not docstring:
            raise ValueError("Function has no docstring")
            
        query = self.extract_sql_query(docstring)
        if not query:
            raise ValueError("No SQL query found in docstring")
            
        return self.execute_query(query)

# Example usage
def load_csv_files() -> Dict[str, pd.DataFrame]:
    """Load all CSV files from the current directory"""
    dataframes = {
        'risk_assessment': pd.read_csv('Risk_Assessment.csv'),
        'borrower_profile': pd.read_csv('Borrower_Profile.csv'),
        'credit_history': pd.read_csv('Credit_History.csv'),
        'financial_ratios': pd.read_csv('Financial_Ratios.csv'),
        'financial_statements': pd.read_csv('Financial_Statements.csv')
    }
    return dataframes

def get_high_risk_borrowers():
    '''
    Get borrowers with high risk scores and poor financial ratios
    
    ''' sql
    WITH HighRiskAssessments AS (
        SELECT 
            ra.borrower_id,
            ra.risk_score,
            ra.market_risk,
            ra.credit_risk
        FROM risk_assessment ra
        WHERE ra.risk_score > 70
        AND (ra.market_risk = 'High' OR ra.credit_risk = 'High')
    ),
    PoorRatios AS (
        SELECT 
            fr.borrower_id,
            fr.current_ratio,
            fr.debt_to_equity,
            fr.interest_coverage_ratio
        FROM financial_ratios fr
        WHERE fr.current_ratio < 1.5
        OR fr.debt_to_equity > 2.0
        OR fr.interest_coverage_ratio < 1.5
    )
    SELECT 
        hr.borrower_id,
        bp.company_name,
        bp.credit_rating,
        hr.risk_score,
        hr.market_risk,
        hr.credit_risk,
        pr.current_ratio,
        pr.debt_to_equity,
        pr.interest_coverage_ratio
    FROM HighRiskAssessments hr
    JOIN borrower_profile bp ON hr.borrower_id = bp.borrower_id
    JOIN PoorRatios pr ON hr.borrower_id = pr.borrower_id
    ORDER BY hr.risk_score DESC;
    '''
    pass

def main():
    # Initialize executor
    executor = SQLiteQueryExecutor()
    
    # Load CSV files into DataFrames
    print("Loading CSV files...")
    dataframes = load_csv_files()
    
    # Load DataFrames into SQLite
    print("Creating in-memory database...")
    executor.load_dataframes(dataframes)
    
    # Execute query
    print("\nExecuting query...")
    results, query = executor.process_function(get_high_risk_borrowers)
    
    print("\nExtracted Query:")
    print(query)
    
    print("\nResults:")
    print(results)
    
    return results

if __name__ == "__main__":
    results_df = main()
