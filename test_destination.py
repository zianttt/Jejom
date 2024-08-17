import json
from tqdm import tqdm
from dotenv import load_dotenv

# https://github.com/mangiucugna/json_repair
from json_repair import repair_json  # LLM JSON output fixing if necessary

from tavily import TavilyClient
# from pymilvus.model.reranker import BGERerankFunction
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

end_user_specs = "Likes travelling alone, photography, male, 22, INFJ"
# end_user_specs = "Likes group travelling, loves camping, artistic, female, 25, ENTJ"
num_spots = 5
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


destination_detail_top_orchestrator_agent = OpenAIAgent.from_tools(
    system_prompt="""
        You are an assistant model that returns the details of a tourist destination given its name.
        Your response MUST be in the form of a JSON object, and MUST contain only the following attributes for the JSON object: 'Name', 'Description', 'Price', 'Address', 'Latitude', 'Longtitude'.
        Your will be given tools that can browse the web for information about specific aspects. You must use them should you require additional information about those aspects.
        If the web browser tool does not return relevant information about the aspects that you are looking for, just set None for that aspect.
        Do not make up any information.

        Example Input:
        'Seongsan Ilchulbong'
        
        Corresponding Output:
        {
            "Name": "Seongsan Ilchulbong",
            "Description": "Seongsan Ilchulbong, also known as 'Sunrise Peak,' is a UNESCO World Heritage site formed by a volcanic eruption over 5,000 years ago. It offers a breathtaking view of the sunrise from its peak, making it a popular destination for early morning hikers.",
            "Price": "KRW 2,000",
            "Address": "Seongsan-eup, Seogwipo-si, Jeju-do, South Korea",
            "Latitude": "33.461111",
            "Longitude": "126.940556"
        }
    """,
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
        )
    ]
)

list_destination_prompt = PromptTemplate(
    """
    You are an assistant that recommends a required number of tourist locations in Jeju Island to the end user.
    You will be given the preferences and characteristics of the end user.
    Your response MUST ONLY be a simple list (without any brackets), with items being separated by commas.
    
    Example Input:
    End User Preferences and Characteristics: 'Nature Lover, Outgoing, Female, 21, ENFJ'
    Number of Tourist Locations Required: 4
    
    Corresponding Output:
    Hallasan National Park, Seongsan Ilchulbong, Olle Trails, Cheonjeyeon Waterfall
    
    Query:
    End User Preferences and Characteristics: {end_user_specs}
    Number of Tourist Locations Required: {num_spots}
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


formatted_destination_prompt = list_destination_prompt.format(end_user_specs=end_user_specs, num_spots=str(num_spots))
destinations_string = Settings.llm.complete(formatted_destination_prompt)
list_of_destinations = str(destinations_string).split(',')

json_response_str = "["
for destination in tqdm(list_of_destinations):
    output_is_json = False
    counter = 0
    try:
        while (not output_is_json) and (counter < max_json_try):
            destination_json_str = destination_detail_top_orchestrator_agent.query(str(destination))
            is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(destination_json_str)))
            if str(is_json).lower().strip() == 'yes':
                output_is_json = True
                json_response_str += str(destination_json_str)
                json_response_str += ","
                break
            else:
                counter += 1
        if counter == max_json_try:
            print(f"Error getting details in JSON format for destination: {destination}. Max tries reached.")
    except:
        print(f"Error getting details for destination: {destination}")
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
