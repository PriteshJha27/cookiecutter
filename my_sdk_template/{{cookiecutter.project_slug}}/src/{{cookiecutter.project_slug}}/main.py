from typing import Any, Dict
from .router import Router

def process_chain(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for processing chains
    """
    router = Router()
    return router.process(input_data)