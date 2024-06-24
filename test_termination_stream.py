import os

# THIS TESTS: TWO AGENTS WITH TERMINATION, STREAMING

# HOW TO USE:
# 1. CHANGE THIS KEY TO MATCH THE CLIENT API KEY KEYWORD AND VALUE
os.environ["COHERE_API_KEY"] = ""

# 2. CHANGE THE PARAMETERS TO MATCH YOUR CLIENT CONFIG
altmodel_llm_config = {
    "config_list":
    [
        {
            "api_type": "cohere",
            "model": "command-r-plus",
            "api_key": os.getenv("COHERE_API_KEY"),
            "stream": True,
            "cache_seed": None
        }
    ]
}

from autogen import ConversableAgent

jack = ConversableAgent(
    "Jack",
    llm_config=altmodel_llm_config,
    system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
    is_termination_msg=lambda x: True if "FINISH" in x["content"] else False
)
emma = ConversableAgent(
    "Emma",
    llm_config=altmodel_llm_config,
    system_message="Your name is Emma and you are a comedian in two-person comedy show. Say the word FINISH ONLY AFTER you've heard 2 of Jack's jokes.",
    is_termination_msg=lambda x: True if "FINISH" in x["content"] else False
)

chat_result = jack.initiate_chat(emma, message="Emma, tell me a joke about goldfish and peanut butter.", max_turns=10)