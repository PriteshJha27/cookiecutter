
import os
import logging
from typing import Dict, List, Any
from unstructured.partition.pdf import partition_pdf
from utils.config_management import load_config
from exceptions.exceptions_loader import DataLoaderError

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles document loading from various sources using unstructured.io."""
    
    def __init__(self):
        self.config = load_config()
        self.data_folder = self.config['data']['data_folder']

    def load_document(self) -> Dict[str, List[Any]]:
        """Load documents from the configured data folder."""
        if not self.data_folder:
            logger.error("Data folder is not specified in the configuration file.")
            raise DataLoaderError("Data folder is not defined in the configuration.")

        logger.info(f"Data folder set to: {self.data_folder}")

        data = {}
        try:
            for filename in sorted(os.listdir(self.data_folder), reverse=True):
                if filename.endswith(".pdf"):
                    data[filename] = self._load_pdf(filename)
            return data
        except Exception as e:
            logger.error(f"Error in document loading: {e}")
            raise DataLoaderError(f"Failed to load documents: {str(e)}")

    def _load_pdf(self, filename: str) -> List[Any]:
        """
        Load a PDF file and its pages using unstructured.io.
        
        Args:
            filename (str): Name of the PDF file to load
            
        Returns:
            List[Any]: List of page elements with metadata
        """
        try:
            logger.info(f"Processing file: {filename}")
            filepath = os.path.join(self.data_folder, filename)
            
            # Use unstructured to partition the PDF
            elements = partition_pdf(
                filepath,
                include_metadata=True,
                strategy="hi_res"  # Use high-resolution strategy for better extraction
            )
            
            # Transform elements to match the expected format
            pages = []
            for idx, element in enumerate(elements):
                # Create a page-like structure to maintain compatibility
                page_data = {
                    'content': str(element),
                    'metadata': {
                        'source': filename,
                        'file_path': filepath,
                        'page_number': element.metadata.page_number if hasattr(element.metadata, 'page_number') else idx + 1
                    }
                }
                pages.append(page_data)
            
            return pages
            
        except Exception as e:
            logger.error(f"Error loading PDF {filename}: {e}")
            raise DataLoaderError(f"Failed to load PDF {filename}: {str(e)}")

    @staticmethod
    def validate_document(document: Dict[str, Any]) -> bool:
        """Validate loaded document structure."""
        try:
            if not isinstance(document, dict):
                return False
                
            required_metadata = ['source', 'file_path']
            for pages in document.values():
                for page in pages:
                    if not hasattr(page, 'metadata'):
                        return False
                    if not all(key in page.metadata for key in required_metadata):
                        return False
                        
            return True
        except Exception:
            return False
___________________________________________________________________________________________________________________________




___________________________________________________________________________________________________________________________
