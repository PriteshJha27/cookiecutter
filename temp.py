
import json
import pandas as pd

class CustomJSONLoader:
    def __init__(self, json_file, schema='.', text_content=False, json_lines=True):
        self.json_file = json_file
        self.schema = schema
        self.text_content = text_content
        self.json_lines = json_lines

    def load(self):
        # Handle the JSON lines (newline-delimited JSON) case
        if self.json_lines:
            with open(self.json_file, 'r') as file:
                data = [json.loads(line) for line in file]
        else:
            # Handle standard JSON file
            with open(self.json_file, 'r') as file:
                data = json.load(file)
        
        # Apply schema (if required, add logic to process schema; here we assume schema=".")
        if self.schema == '.':
            # If schema is ".", no transformation is applied; return raw data
            return pd.DataFrame(data)
        else:
            raise NotImplementedError("Custom schema processing is not implemented.")

# Example usage:
json_file = "your_file.json"
loader = CustomJSONLoader(json_file, schema='.', text_content=False, json_lines=True)
df = loader.load()

print(df)


loader = CustomJSONLoader("your_file.json", schema='.', text_content=False, json_lines=True)
doc = loader.load()
