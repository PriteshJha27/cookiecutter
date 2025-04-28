
define user greet
"hello"
"hi"
"hey"
"good morning"
"good evening"

define user ask_fact_check
"is it true that"
"can you verify"
"please fact check"
"confirm whether"

define user ask_general_question
"what is"
"how to"
"explain"

define flow handle_user_input
  user greet
    bot "Hello! How can I assist you today?"

  user ask_fact_check
    bot "Let me fact check that for you..."

  user ask_general_question
    bot "Sure, let me explain that to you..."







define user ask_about_unrelated_topics
"tell me a joke"
"who is your creator"
"what's your name"
"do you like pizza"

define flow handle_off_topic
  user ask_about_unrelated_topics
    bot "I'm here to assist you with important queries. Let's stay focused!"








define user fact_check_request
"is it true that"
"can you verify"
"fact check"

define flow handle_fact_check
  user fact_check_request
    bot "I will retrieve trusted sources and validate the information for you."





from nemoguardrails import RailsConfig, LLMRails
from langchain.chat_models import ChatOpenAI

config = RailsConfig(
    models=[{
        "type": "main",
        "engine": "openai",
        "model": "gpt-4o"
    }],
    colang_files=[
        "D:/Pritesh/VS Code Workspace/DSA/test/guardrails/user_intent.co",
        "D:/Pritesh/VS Code Workspace/DSA/test/guardrails/factcheck.co",
        "D:/Pritesh/VS Code Workspace/DSA/test/guardrails/off-topic.co"
    ],
    prompt_files=[
        "D:/Pritesh/VS Code Workspace/DSA/test/guardrails/prompts.yml"
    ]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
rails = LLMRails(config=config, llm=llm)
