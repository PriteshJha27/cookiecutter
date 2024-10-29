# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key

### Environment Setup
1. Create a `.env` file in your project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

2. Install the package:
```bash
pip install -e .
```

## Quick Start

```python
from {{cookiecutter.project_slug}} import process_chain

# Example 1: Simple text completion
result = process_chain({
    "prompt_template": "List 5 ways to stay healthy: {format_instructions}",
    "output_format": "list"
})
print(result)

# Example 2: JSON output
result = process_chain({
    "prompt_template": "Create a character profile for {name} with the following attributes: age, occupation, hobbies",
    "name": "John Doe",
    "output_format": "json"
})
print(result)
```

## Features

### 1. Prompt Templates
Create dynamic prompts using ChatPromptTemplate:
```python
result = process_chain({
    "prompt_template": "Translate the following text to {language}: {text}",
    "language": "French",
    "text": "Hello, how are you?",
})
```

### 2. Output Parsers
Support for different output formats:
- JSON (`"output_format": "json"`)
- Comma-separated list (`"output_format": "list"`)

### 3. LCEL (LangChain Expression Language)
Built-in support for LCEL chains:
- Prompt → Model → Output Parser pipeline
- Easily extensible for custom chains

## Configuration

Update `config.yml` to modify:

```yaml
model:
  name: gpt-3.5-turbo      # Model name
  temperature: 0.7         # Creativity (0.0 - 1.0)
  max_tokens: 1000         # Maximum response length

prompts:
  default_system_prompt: "You are a helpful AI assistant."
```

## Project Structure
```
src/{{cookiecutter.project_slug}}/
├── __init__.py
├── main.py              # Main entry point
├── router.py            # Central routing logic
├── services/           
│   └── lcel.py         # LCEL chain builder
├── model/
│   └── models.py       # LLM model setup
├── prompts/
│   └── prompt_generator.py  # Prompt template creation
├── outputparser/
│   └── output_parser.py    # Output parsing utilities
└── utils/
    └── helpers.py      # Utility functions
```

## Advanced Usage

### Custom System Prompts
```python
result = process_chain({
    "prompt_template": "Explain {concept} like I'm five",
    "concept": "photosynthesis",
    "system_prompt": "You are a friendly teacher who explains complex topics in simple terms"
})
```

### Handling Complex Outputs
```python
# JSON output with specific structure
result = process_chain({
    "prompt_template": "Analyze the sentiment of: {text}",
    "text": "I love this product but the delivery was slow",
    "output_format": "json",
    "output_schema": {
        "sentiment": "str",
        "score": "float",
        "aspects": "list"
    }
})
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
