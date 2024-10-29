# {{cookiecutter.project_slug}}/setup.py
from setuptools import setup, find_packages

setup(
    name="{{cookiecutter.project_slug}}",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "pydantic>=2.0.0",
        "python-dotenv>=0.19.0",
        "PyYAML>=5.4.1"
    ],
    author="{{cookiecutter.author_name}}",
    description="{{cookiecutter.description}}",
)