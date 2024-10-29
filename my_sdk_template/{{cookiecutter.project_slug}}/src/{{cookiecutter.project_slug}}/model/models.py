
# {{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/model/models.py
import os
from langchain_openai import ChatOpenAI
from ..utils.helpers import load_config

def get_model() -> ChatOpenAI:
    """
    Initialize and return the ChatOpenAI model
    """
    config = load_config()
    return ChatOpenAI(
        model_name=config["model"]["name"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"]
    )