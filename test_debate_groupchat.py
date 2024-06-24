import json
import os
from dotenv import load_dotenv
import autogen
from autogen import ConversableAgent, AssistantAgent
os.system("clear")

# THIS TESTS: GROUP CHAT

# HOW TO USE:
# 1. CHANGE THESE TO MATCH THE CLIENT API KEY KEYWORD AND VALUE
os.environ["COHERE_API_KEY"] = ""
api_key = os.environ["COHERE_API_KEY"]

# 2. CHANGE THE PARAMETERS TO MATCH YOUR CLIENT CONFIG
altmodel_llm_config = {
    "config_list":
    [
        {
            "api_type": "cohere",
            "model": "command-r-plus",
            "api_key": os.getenv("COHERE_API_KEY"),
            "cache_seed": None
        }
    ]
}

class debate:
    def __init__(self):
        import os

        self.llm = altmodel_llm_config["config_list"][0]["model"]
        tdict={"model":self.llm,"api_key":api_key}
        os.environ["OAI_CONFIG_LIST"] = "[" + json.dumps(tdict) + "]"
        self.config_file_or_env = "OAI_CONFIG_LIST"
        filepath = os.path.join(os.path.dirname(__file__), "test_debate_team.json")
        self.saved_team = filepath
        self.llm_config = altmodel_llm_config
        os.environ["AUTOGEN_USE_DOCKER"] = "False"

    def load_team(self):
        #from urllib.parse import urlparse
        #from autogen.agentchat.contrib.agent_builder import AgentBuilder

        file_name = self.saved_team
        # new_builder = AgentBuilder(config_file_or_env=self.config_file_or_env)
        # self.agent_list, self.agent_config = new_builder.load(file_name)

        self.agent_list = []
        if file_name is not None:
            print(f"Loading config from {file_name}")
            with open(file_name) as f:
                file_configs = json.load(f)

                for agent_config in file_configs["agent_configs"]:
                    agent = AssistantAgent(name=agent_config["name"], system_message=agent_config["system_message"], description=agent_config["description"], llm_config=altmodel_llm_config)
                    self.agent_list.append(agent)

    def do_debate(self, proposition):
        import autogen

        # MS - Add termination to moderator
        for agent in self.agent_list:
            if agent.name == "Debate_Moderator_Agent":
                agent.is_termination_msg = lambda x: x.get("content", "").find("TERMINATE") >= 0

        group_chat = autogen.GroupChat(
            agents=self.agent_list,
            messages=[],
            max_round=15,
            select_speaker_message_template="You are managing a debate and your only job is to select the next speaker, each speaker has a name. Follow the debate and decide on who should speak next. The 'Debate_Moderator_Agent' is the first speaker and they will kick off the debate with a topic to debate. Then each of the four debaters will speak and speak only once each. You should start by selecting the 'Affirmative_Constructive_Debater' to provide their opening arguments in the debate.",
            select_speaker_prompt_template="Read the above conversation and your job role, which is managing the debate and choosing the next speaker. The valid speakers can be selected from this list {agentlist}. During the debate the order of debaters are: 1st is the 'Affirmative_Constructive_Debater', 2nd is the 'Negative_Constructive_Debater', 3rd is the 'Affirmative_Rebuttal_Debater', and 4th is the 'Negative_Rebuttal_Debater'. Then 5th will be the 'Debate_Judge' and 6th is the 'Debate_Moderator_Agent'.",
            max_retries_for_selecting_speaker=1,
            role_for_select_speaker_messages="user",  # CRITICAL TO SET THIS TO USER, IF SYSTEM YOU MAY GET BLANK RESPONSES
            select_speaker_auto_verbose=True,
        )

        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=altmodel_llm_config,
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        )

        result = self.agent_list[0].initiate_chat(manager, message=proposition)


load_dotenv()
dm = debate()
dm.load_team()
proposition = "The earth is really flat."
dm.do_debate(f"Please debate the proposition '{proposition}'.")