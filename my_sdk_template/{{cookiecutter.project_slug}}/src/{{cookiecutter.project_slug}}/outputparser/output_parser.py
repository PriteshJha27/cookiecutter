# {{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/outputparser/output_parser.py
from typing import Optional, Type
from langchain.schema.output_parser import BaseOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser, JSONOutputParser

def create_output_parser(parser_type: str) -> Optional[BaseOutputParser]:
    """
    Create an output parser based on the specified type
    """
    parser_map = {
        "json": JSONOutputParser(),
        "list": CommaSeparatedListOutputParser(),
    }
    return parser_map.get(parser_type)