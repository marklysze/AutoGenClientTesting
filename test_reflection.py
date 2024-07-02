from typing_extensions import Annotated

import autogen
import os

# THIS TESTS: LLM Reflection

# HOW TO USE:
# 1. CHANGE THIS KEY TO MATCH THE CLIENT API KEY KEYWORD AND VALUE
os.environ["COHERE_API_KEY"] = ""

# 2. CHANGE THE PARAMETERS TO MATCH YOUR CLIENT CONFIG
config_list = [
    {
        "api_type": "cohere",
        "model": "command-r-plus",
        "api_key": os.getenv("COHERE_API_KEY"),
        "cache_seed": None
    }
]

llm_config = {"config_list": config_list}

task = """Write a concise but engaging blogpost about Navida."""

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a professional writer, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    You should imporve the quality of the content based on the feedback from the user.
    """,
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config={"config_list": config_list},
    system_message="""
    You are a critic, known for your thoroughness and commitment to standards.
    Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
    all materials align with required guidelines.
    For code
    """,
)

def reflection_message(recipient, messages, sender, config):
    print("Reflecting...", "yellow")
    return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


user_proxy.register_nested_chats(
    [{"recipient": critic, "message": reflection_message, "summary_method": "last_msg", "max_turns": 1}],
    trigger=writer,
)

res = user_proxy.initiate_chat(recipient=writer, message=task, max_turns=2, summary_method="last_msg")
