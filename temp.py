import torch
from transformers import AutoTokenizer, AutoModel
from typing import List


class EmbeddingGenerator:
    def __init__(self, model_path: str):
        """
        Initializes the embedding generator with a Hugging Face model.
        Args:
            model_path (str): Path to the pre-trained Hugging Face model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling of token embeddings, considering attention mask.
        """
        token_embeddings = model_output[0]  # First element is token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """
        Generates sentence embeddings for a list of sentences.
        Args:
            sentences (List[str]): List of input sentences.
        Returns:
            torch.Tensor: Normalized sentence embeddings.
        """
        print("Generating embeddings...")
        # Tokenize inputs
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=2048)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


import faiss
import numpy as np
from typing import List


class IndexSaver:
    def __init__(self, vectorstore_path: str, embedding_dimension: int = 768):
        """
        Initializes the FAISS index saver.
        Args:
            vectorstore_path (str): Path to save the FAISS index.
            embedding_dimension (int): Dimension of the embeddings.
        """
        self.vectorstore_path = vectorstore_path
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(self.embedding_dimension)

    def save_embeddings_to_index(self, embeddings: List[np.ndarray], save_name: str = "index.faiss"):
        """
        Adds embeddings to the FAISS index and saves it.
        Args:
            embeddings (List[np.ndarray]): List of embeddings to save.
            save_name (str): Name of the saved FAISS index file.
        """
        print("Saving embeddings to FAISS index...")
        all_embeddings = np.vstack(embeddings).astype("float32")
        self.index.add(all_embeddings)
        save_dir = f"{self.vectorstore_path}/{save_name}"
        faiss.write_index(self.index, save_dir)
        print(f"FAISS index saved at: {save_dir}")

    def load_index(self, index_name: str = "index.faiss"):
        """
        Loads an existing FAISS index.
        Args:
            index_name (str): Name of the FAISS index file to load.
        Returns:
            faiss.Index: Loaded FAISS index.
        """
        print(f"Loading FAISS index from {self.vectorstore_path}/{index_name}...")
        return faiss.read_index(f"{self.vectorstore_path}/{index_name}")
