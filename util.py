from langchain.chat_models import ErnieBotChat
from langchain.llms.ollama import Ollama
from pandasai.llm import OpenAI, LangchainLLM
from pandasai.prompts import GeneratePythonCodePrompt

from llm.ais_erniebot import AIStudioErnieBot
from llm.google_gemini import GoogleGeminiChat


def get_open_ai_model(api_key):
    return OpenAI(api_token=api_key)


def get_ollama_model(model_key, base_url):
    llm = Ollama(model=model_key, base_url=base_url, verbose=True)
    return LangchainLLM(langchain_llm=llm)


def get_google_gemini_model(api_key, model_name):
    llm_core = GoogleGeminiChat(
        google_api_key=api_key,
        model_name=model_name,
        temperature=0.1
    )
    return LangchainLLM(llm_core)


def get_prompt_template():
    instruction_template = """
Use the provided dataframe(s) in 'dfs' to analyze the data. Do not call dataframe.set_index() to sort the data.
1. Preparation: Preprocess and clean the data if needed.
2. Execution: Perform data analysis operations such as grouping, filtering, aggregating, and similar transformations.
3. Analysis: Complete the requested analysis. If the user asks for a chart, add the following two lines to configure the font, save the chart as temp_chart.png, and do not display the chart directly.
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False    
    """
    custom_template = GeneratePythonCodePrompt(custom_instructions=instruction_template)
    return custom_template
