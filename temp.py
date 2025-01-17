def generate_instruction_prompt():
    """
    Generates a generic instruction prompt for SQL generation.
    Returns:
        str: Instruction prompt string.
    """
    instruction_prompt = """
    **Instruction**:
    - Generate a Hive SQL query step by step based on the given schema and user question.
    - Ensure the query structure is valid and uses relevant tables and columns.
    - Provide reasoning for the chosen tables, columns, and joins.
    """
    return instruction_prompt


def generate_few_shot_example_prompt(few_shot_examples=None):
    """
    Generates a few-shot example prompt for guiding the model.
    Args:
        few_shot_examples (list): List of few-shot examples, if available.
    Returns:
        str: Few-shot example prompt string.
    """
    examples = few_shot_examples if few_shot_examples else []
    return f"""
    **Few-Shot Examples**:
    {examples}
    """


def generate_comment_prompt(question, knowledge=None):
    """
    Generates a prompt with user query and additional knowledge context.
    Args:
        question (str): User query or question.
        knowledge (str, optional): Additional domain knowledge or context.
    Returns:
        str: Combined comment prompt string.
    """
    knowledge_prompt = f"**Knowledge**: {knowledge}" if knowledge else ""
    combined_prompt = f"""
    **Question**: {question}
    {knowledge_prompt}
    """
    return combined_prompt


def cru_instruction_prompt():
    """
    Generates a CRU-specific instruction prompt for SQL generation.
    Returns:
        str: CRU instruction prompt string.
    """
    return """
    **CRU Domain Knowledge**:
    - Include CRU-specific details in the query generation.
    """


def generate_cot_prompt():
    """
    Generates a "chain-of-thought" (CoT) prompt for step-by-step SQL reasoning.
    Returns:
        str: CoT prompt string.
    """
    cot_prompt = """
    Generate the Hive SQL query step by step. Below I provide an example:
    - Identify the required tables from the schema.
    - Select relevant columns for filtering, grouping, and aggregations.
    - Ensure valid joins and conditions based on the schema relationships.
    """
    return cot_prompt


def generate_schema_prompt_all(table_schema, keyword_results, foreign_key_table_column_dict=True):
    """
    Generates a schema-based prompt with table schema and keywords.
    Args:
        table_schema (str): Schema details in string format.
        keyword_results (dict): Results with relevant tables and keywords.
        foreign_key_table_column_dict (bool): Whether to include foreign key relationships.
    Returns:
        str: Schema prompt string.
    """
    return f"""
    **Schema Information**:
    Below is the table schema related to the question:
    **Database Info**:
    {table_schema}

    On searching through the schema, the following tables came out to be most significant:
    {keyword_results}
    """


def generate_combined_prompts_one(
    question,
    schema,
    keyword_results,
    few_shot_examples=None,
    knowledge=None,
    foreign_key_table_column_dict=None,
):
    """
    Combines all the generated prompts into one comprehensive prompt.
    Args:
        question (str): User query or question.
        schema (str): Schema details in string format.
        keyword_results (dict): Results with relevant tables and keywords.
        few_shot_examples (list, optional): Few-shot examples, if available.
        knowledge (str, optional): Additional domain knowledge or context.
        foreign_key_table_column_dict (bool, optional): Foreign key relationships.
    Returns:
        str: Combined prompt string.
    """
    few_shot_prompt = generate_few_shot_example_prompt(few_shot_examples) if few_shot_examples else ""
    schema_prompt = generate_schema_prompt_all(schema, keyword_results, foreign_key_table_column_dict)
    cot_prompt = generate_cot_prompt()
    instruction_prompt = generate_instruction_prompt()
    comment_prompt = generate_comment_prompt(question, knowledge)
    cru_prompt = cru_instruction_prompt()

    combined_prompts = "\n\n".join(
        [
            "Given a database schema, question, and knowledge, generate the correct Hive SQL query for the question.",
            few_shot_prompt,
            cot_prompt,
            schema_prompt,
            comment_prompt,
            cru_prompt,
            instruction_prompt,
        ]
    )

    return combined_prompts
