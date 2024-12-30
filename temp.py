import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class DatabaseMetadataSearch:
    def __init__(self):
        # Initialize NLP components
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stop words
        self.domain_stop_words = {
            'table', 'column', 'database', 'schema', 'id', 'key', 'partition',
            'type', 'varchar', 'int', 'float', 'date', 'timestamp'
        }
        self.stop_words.update(self.domain_stop_words)
        
        # Initialize vectorizers
        self.table_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(self.stop_words),
            min_df=1,
            max_df=0.95,
            analyzer='word'
        )
        
        self.column_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(self.stop_words),
            min_df=1,
            max_df=0.95,
            analyzer='word'
        )
        
        # Store vectorized matrices
        self.table_matrix = None
        self.column_matrix = None
        
        # Store feature names
        self.table_features = None
        self.column_features = None
        
        # Store original dataframe
        self.df = None
        
    def preprocess_text(self, text):
        """Preprocess text by cleaning and normalizing"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def prepare_table_descriptions(self, df):
        """
        Process table-level metadata and create TF-IDF matrix
        """
        self.df = df.copy()
        
        # Combine table metadata
        self.df['table_combined'] = self.df.apply(
            lambda row: f"{row['table_name']} {row['table_description']} {row['keys']} {row['partition_keys']}",
            axis=1
        )
        
        # Preprocess combined text
        processed_texts = self.df['table_combined'].apply(self.preprocess_text)
        
        # Create TF-IDF matrix
        self.table_matrix = self.table_vectorizer.fit_transform(processed_texts)
        self.table_features = self.table_vectorizer.get_feature_names_out()
        
        # Get top keywords for each table
        self.df['table_keywords'] = self._get_top_keywords(
            self.table_matrix, 
            self.table_features, 
            n_keywords=5
        )
        
        return self.table_matrix
    
    def prepare_column_descriptions(self):
        """
        Process column-level metadata and create TF-IDF matrix
        """
        # Combine column metadata
        self.df['column_combined'] = self.df.apply(
            lambda row: f"{row['column_name']} {row['column_description']}",
            axis=1
        )
        
        # Preprocess combined text
        processed_texts = self.df['column_combined'].apply(self.preprocess_text)
        
        # Create TF-IDF matrix
        self.column_matrix = self.column_vectorizer.fit_transform(processed_texts)
        self.column_features = self.column_vectorizer.get_feature_names_out()
        
        # Get top keywords for each column
        self.df['column_keywords'] = self._get_top_keywords(
            self.column_matrix, 
            self.column_features, 
            n_keywords=3
        )
        
        return self.column_matrix
    
    def _get_top_keywords(self, tfidf_matrix, feature_names, n_keywords=5):
        """Extract top keywords based on TF-IDF scores"""
        top_keywords = []
        
        for i in range(tfidf_matrix.shape[0]):
            # Get scores for this document
            scores = tfidf_matrix[i].toarray()[0]
            
            # Create tuples of (keyword, score)
            keyword_scores = list(zip(feature_names, scores))
            
            # Sort by score and get top n
            sorted_keywords = sorted(
                keyword_scores, 
                key=lambda x: x[1], 
                reverse=True
            )[:n_keywords]
            
            # Store only keywords with non-zero scores
            keywords = [k for k, s in sorted_keywords if s > 0]
            top_keywords.append(keywords)
        
        return top_keywords
    
    def search(self, query, top_n=5):
        """
        Search for relevant tables and columns based on query
        """
        if self.table_matrix is None or self.column_matrix is None:
            raise ValueError("Please prepare table and column descriptions first")
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Vectorize query for tables
        query_vec_table = self.table_vectorizer.transform([processed_query])
        
        # Vectorize query for columns
        query_vec_column = self.column_vectorizer.transform([processed_query])
        
        # Calculate similarities
        table_similarities = cosine_similarity(query_vec_table, self.table_matrix)[0]
        column_similarities = cosine_similarity(query_vec_column, self.column_matrix)[0]
        
        # Get top matching tables
        table_matches = []
        for idx in table_similarities.argsort()[-top_n:][::-1]:
            if table_similarities[idx] > 0:
                table_matches.append({
                    'table_name': self.df.iloc[idx]['table_name'],
                    'similarity': table_similarities[idx],
                    'keywords': self.df.iloc[idx]['table_keywords']
                })
        
        # Get top matching columns
        column_matches = []
        for idx in column_similarities.argsort()[-top_n:][::-1]:
            if column_similarities[idx] > 0:
                column_matches.append({
                    'table_name': self.df.iloc[idx]['table_name'],
                    'column_name': self.df.iloc[idx]['column_name'],
                    'similarity': column_similarities[idx],
                    'keywords': self.df.iloc[idx]['column_keywords']
                })
        
        return {
            'table_matches': table_matches,
            'column_matches': column_matches
        }

# Example usage:
def main():
    # Sample data
    data = {
        'table_name': ['customer_orders', 'product_inventory'],
        'table_description': ['Contains customer order details and history', 'Product stock and inventory management'],
        'column_name': ['order_date', 'stock_level'],
        'column_description': ['Date when order was placed', 'Current inventory level'],
        'keys': ['customer_id, order_id', 'product_id'],
        'partition_keys': ['order_date', 'update_date']
    }
    
    df = pd.DataFrame(data)
    
    # Initialize the search engine
    search_engine = DatabaseMetadataSearch()
    
    # Prepare matrices
    search_engine.prepare_table_descriptions(df)
    search_engine.prepare_column_descriptions()
    
    # Example search
    query = "customer orders history"
    results = search_engine.search(query)
    
    # Print results
    print("\nMatching Tables:")
    for match in results['table_matches']:
        print(f"Table: {match['table_name']}")
        print(f"Similarity: {match['similarity']:.3f}")
        print(f"Keywords: {match['keywords']}\n")
    
    print("\nMatching Columns:")
    for match in results['column_matches']:
        print(f"Table: {match['table_name']}")
        print(f"Column: {match['column_name']}")
        print(f"Similarity: {match['similarity']:.3f}")
        print(f"Keywords: {match['keywords']}\n")

if __name__ == "__main__":
    main()


# Initialize
search_engine = DatabaseMetadataSearch()

# Prepare your data
search_engine.prepare_table_descriptions(your_dataframe)
search_engine.prepare_column_descriptions()

# Search
results = search_engine.search("your search query")
