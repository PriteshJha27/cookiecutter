#!/bin/bash

# Define base directory name
BASE_DIR="cookiecutter-sdk-template"

echo "🚀 Creating Langchain SDK Template Structure..."

# Interactive prompt for project details
echo "Please provide the following details for your project:"
echo "---------------------------------------------------"

# Get project details with default values
read -p "Project Name (default 'Langchain Project'): " project_name
project_name=${project_name:-"Langchain Project"}

read -p "Author Name (default 'Your Name'): " author_name
author_name=${author_name:-"Your Name"}

read -p "Project Description (default 'A Langchain-based SDK for handling prompts, chains, and LLM interactions'): " description
description=${description:-"A Langchain-based SDK for handling prompts, chains, and LLM interactions"}

read -p "OpenAI Model Name (default 'gpt-4o-mini'): " model_name
model_name=${model_name:-"gpt-4o-mini"}

echo "---------------------------------------------------"
echo "Creating directory structure..."

# Create the directory structure
mkdir -p $BASE_DIR/{hooks,"{{cookiecutter.project_slug}}"/{src/"{{cookiecutter.project_slug}}"/{services,model,prompts,outputparser,utils},tests}}

# Create cookiecutter.json with user inputs
cat > $BASE_DIR/cookiecutter.json << EOF
{
    "project_name": "$project_name",
    "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
    "author_name": "$author_name",
    "description": "$description",
    "openai_model_name": "$model_name"
}
EOF

# Create empty files
touch $BASE_DIR/hooks/post_gen_project.py
touch "$BASE_DIR/{{cookiecutter.project_slug}}/README.md"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/config.yml"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/requirements.txt"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/setup.py"

# Create Python files
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/main.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/router.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/services/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/services/lcel.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/model/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/model/models.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/prompts/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/prompts/prompt_generator.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/outputparser/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/outputparser/output_parser.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/utils/__init__.py"
touch "$BASE_DIR/{{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/utils/helpers.py"

touch "$BASE_DIR/{{cookiecutter.project_slug}}/tests/__init__.py"

echo "✅ Directory structure created successfully!"
echo ""
echo "📁 Structure created:"
tree $BASE_DIR

echo ""
echo "Project Details Summary:"
echo "---------------------------------------------------"
echo "Project Name: $project_name"
echo "Author: $author_name"
echo "Description: $description"
echo "OpenAI Model: $model_name"
echo "---------------------------------------------------"
echo ""
echo "Next steps:"
echo "1. Copy your Python code into the respective files"
echo "2. Review cookiecutter.json configuration in $BASE_DIR/cookiecutter.json"
echo "3. Add your README.md content"
echo "4. Add requirements.txt dependencies"