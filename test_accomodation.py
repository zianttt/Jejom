import json
from tqdm import tqdm
from dotenv import load_dotenv

# https://github.com/mangiucugna/json_repair
from json_repair import repair_json  # LLM JSON output fixing if necessary

from tavily import TavilyClient
from llama_index.core import PromptTemplate, Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.groq import Groq
# from llama_index.llms.upstage import Upstage
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent


# Global wide models
load_dotenv() # api keys
Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')
# Settings.llm = Upstage(model='solar-1-mini-chat')

# Config
VERBOSE = False

end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
num_accomodations = 5
max_json_try = 3



def tavily_browser_tool(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=input, search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_longlat(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Geographical longtitude and latitude of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_address(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Location and Address of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_rating(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Rating of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_provider(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Provider of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)


accomodation_detail_top_orchestrator_agent = OpenAIAgent.from_tools(
    system_prompt="""
        You are an assistant model that returns the details of a tourist accomodation given its name.
        Your response MUST be in the form of a JSON object, and MUST contain only the following attributes for the JSON object: 'Name', 'Address', 'Price', 'Rating', 'Latitude', 'Longtitude', 'Provider'.
        Your will be given tools that can browse the web for information about specific aspects. You must use them should you require additional information about those aspects.
        If the web browser tool does not return relevant information about the aspects that you are looking for, just set None for that aspect.
        Do not make up any information.

        Example Input:
        "Lotte Hotel Jeju"
        
        Corresponding Output:
        {
            "Name": "Lotte Hotel Jeju",
            "Address": "35, Jungmungwangwang-ro 72beon-gil, Seogwipo-si, Jeju-do, South Korea",
            "Price": "KRW 300,000 - 600,000 per night",
            "Rating": "4.5 out of 5",
            "Latitude": "33.247204",
            "Longitude": "126.412185",
            "Provider": "Lotte Hotels & Resorts"
        }
    """,
    llm=Settings.llm,
    verbose=VERBOSE,
    tools=[
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool,
                        tool_metadata=ToolMetadata(
                            name='general web browser tool',
                            description=f"Useful for browsing general information."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for general information.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='general web browser tool',
                description=f"Useful for browsing general information."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_longlat,
                        tool_metadata=ToolMetadata(
                            name='location longtitude and latitude web browser tool',
                            description=f"Useful for browsing geographic longtitue and latitude of a location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for longtitude and latitude of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location longtitude and latitude web browser tool',
                description=f"Useful for browsing geographic longtitue and latitude of a location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_address,
                        tool_metadata=ToolMetadata(
                            name='location address web browser tool',
                            description=f"Useful for browsing address of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the address of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location address web browser tool',
                description=f"Useful for browsing address of location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_rating,
                        tool_metadata=ToolMetadata(
                            name='location rating web browser tool',
                            description=f"Useful for browsing rating of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the rating of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location rating web browser tool',
                description=f"Useful for browsing rating of location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_provider,
                        tool_metadata=ToolMetadata(
                            name='location provider web browser tool',
                            description=f"Useful for browsing provider of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the provider of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location provider web browser tool',
                description=f"Useful for browsing provider of location."
            )
        )
    ]
)

list_accomodation_prompt = PromptTemplate(
    """
    You are an assistant that recommends a required number of tourist accomodations in Jeju Island to the end user.
    You will be given the preferences and/or characteristics and/or requirements of the end user.
    Your response MUST ONLY be a simple list (without any brackets), with items being separated by commas.
    
    Example Input:
    End User Preferences/Characteristics/Requirements: 'Nature Lover, Female, 21, ENFJ, Loves a view by the ocean, North part of Jeju island.'
    Number of Tourist Accomodations Required: 4
    
    Corresponding Output:
    Ramada Plaza Jeju Ocean Front, Ocean Suites Jeju Hotel, Hotel RegentMarine The Blue, Jeju Oriental Hotel & Casino
    
    Query:
    End User Preferences/Characteristics/Requirements: {end_user_specs}
    Number of Tourist Accomodations Required: {num_accomodations}
    """
)

json_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it follows the exact format of a JSON Object.
    If the query follows the format of a JSON Object, respond with "yes". 
    If it does not follow the format of a JSON Object, respond with "no".
    Query: {query_str}
    """
)


formatted_accomodation_prompt = list_accomodation_prompt.format(end_user_specs=end_user_specs, num_accomodations=str(num_accomodations))
accomodations_string = Settings.llm.complete(formatted_accomodation_prompt)
list_of_accomodations = str(accomodations_string).split(',')

json_response_str = "["
for accomodation in tqdm(list_of_accomodations):
    output_is_json = False
    counter = 0
    try:
        while (not output_is_json) and (counter < max_json_try):
            accomodation_json_str = accomodation_detail_top_orchestrator_agent.query(str(accomodation))
            is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(accomodation_json_str)))
            if str(is_json).lower().strip() == 'yes':
                output_is_json = True
                json_response_str += str(accomodation_json_str)
                json_response_str += ","
                break
            else:
                counter += 1
        if counter == max_json_try:
            print(f"Error getting details in JSON format for accomodation: {accomodation}. Max tries reached.")
    except:
        print(f"Error getting details for accomodation: {accomodation}")
if str(json_response_str).endswith(','):
    json_response_str = str(json_response_str)[:-1]
json_response_str += "]"

print()
print()
print(json_response_str)

json_response = None
try:
    json_response = json.loads(json_response_str)
except:
    try:
        print(f"Repairing JSON string ...")
        good_json_response_str = repair_json(json_response_str, skip_json_loads=True)
    except OSError:
        print(f"Repair failed.")
        good_json_response_str = ""  # real bad json string / os error
    if not (str(good_json_response_str).strip() == ""):
        json_response = json.loads(good_json_response_str)

print("============")
print(json_response)
