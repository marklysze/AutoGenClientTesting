import os
import autogen
import json
from typing import Literal
from typing_extensions import Annotated

# THIS TESTS: PARALLEL TOOL CALLING WITH STREAMING

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

# Create the agent and include examples of the function calling JSON in the prompt
# to help guide the model
chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""For currency exchange and weather forecasting tasks,
        only use the functions you have been provided with.
        Output 'TERMINATE' when an answer has been provided.
        Do not include the function name or result in the JSON.
        Use both tools together if you can.
        Example of the return JSON is:
        {
            "parameter_1_name": 100.00,
            "parameter_2_name": "ABC",
            "parameter_3_name": "DEF",
        }.
        Another example of the return JSON is:
        {
            "parameter_1_name": "GHI",
            "parameter_2_name": "ABC",
            "parameter_3_name": "DEF",
            "parameter_4_name": 123.00,
        }.""",

    llm_config=altmodel_llm_config,
)


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

# Currency Exchange function

CurrencySymbol = Literal["USD", "EUR"]

# Define our function that we expect to call
def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")

# Register the function with the agent
@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{format(quote_amount, '.2f')} {quote_currency}"


# Weather function

# Example function to make available to model
def get_current_weather(location, unit="fahrenheit"):
    """Get the weather for some location"""
    if "chicago" in location.lower():
        return json.dumps({"location": "Chicago", "temperature": "13", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "55", "unit": unit})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "11", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

# Register the function with the agent
@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Weather forecast for US cities.")
def weather_forecast(
    location: Annotated[str, "City name"],
) -> str:
    weather_details = get_current_weather(location=location)
    weather = json.loads(weather_details)
    return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}"

# start the conversation
res = user_proxy.initiate_chat(
    chatbot,
    # message="How much is 123.45 EUR in USD?",
    message="What's the weather in New York and can you tell me how much is 123.45 EUR in USD so I can spend it on my holiday?",
    summary_method="reflection_with_llm",
)