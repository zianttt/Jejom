import os
import json
import numpy as np
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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent

load_dotenv()
VERBOSE = False


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

same_location_check_prompt = PromptTemplate(
    """
    You are a classifation model. Given the two lcoations in Jeju Island below, determine if they both refer to the same location.
    If the two locations refer to the same place, respond with "yes".
    If the two locations do not refer to the same place, respond with "no".
    
    Query:
    Location 1: {loc_1}
    Location 2: {loc_2}
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

isDestination_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about any destination.
    If the query does contain information about a certain destination, respond with "yes". 
    If it does not contain information about any destination, respond with "no".
    Query: {query_str}
    """
)

isDuration_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about datetime or duration.
    If the query does contain information about any datetime or duration, respond with "yes". 
    If it does not contain information about any datetime or duration, respond with "no".
    Query: {query_str}
    """
)

isBudget_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about budget.
    If the query does contain information about budget, respond with "yes". 
    If it does not contain information about budget, respond with "no".
    Query: {query_str}
    """
)

isInterest_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about hobby or interests.
    If the query does contain information about hobby or interests, respond with "yes". 
    If it does not contain information about hobby or interests, respond with "no".
    Query: {query_str}
    """
)

def get_most_similar_location_from_cache(embed_model: OpenAIEmbedding, cache_dict, location_name, top_k=1):
    cache_dict_names = list(cache_dict.keys())
    
    location_name_embed = embed_model.get_text_embedding(location_name)
    cache_dict_names_embed = embed_model.get_text_embedding_batch(cache_dict_names)
    
    location_name_embed_np = np.array(location_name_embed)
    cache_dict_names_embed_np = np.array(cache_dict_names_embed)
    assert location_name_embed_np.shape[0] == cache_dict_names_embed_np.shape[1]
    
    # cosine sim
    location_name_embed_norm = np.linalg.norm(location_name_embed_np)
    cache_dict_names_embed_norm = np.linalg.norm(cache_dict_names_embed_np, axis=1, keepdims=True)
    similarities = np.dot(cache_dict_names_embed_np, location_name_embed_np) / (cache_dict_names_embed_norm.flatten() * location_name_embed_norm)
    
    top_k_indices = np.argsort(similarities)[-int(top_k):][::-1]
    top_k_indices = top_k_indices.tolist()
    
    most_sim_location_from_cache = []
    for idx in top_k_indices:
        most_sim_location_from_cache.append(cache_dict_names[idx])
    return most_sim_location_from_cache



class Pipeline():
    def __init__(self, USE_CACHE=True, accomodation_cache_file="./cache/accomodations.json", destination_cache_file = "./cache/destinations.json"):
        self.USE_CACHE = USE_CACHE
        self.accomodation_cache_file = accomodation_cache_file
        self.destination_cache_file = destination_cache_file

    def get_accomodations_json(self, query: str, num_accomodations: int = 5, max_json_try: int = 3, check_match_from_cache_top_k: int = 2):
        # end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        # num_accomodations = 5
        # max_json_try = 3
        # check_match_from_cache_top_k = 2

        accomodation_cache_available = os.path.exists(self.accomodation_cache_file)
        if accomodation_cache_available:
            with open(self.accomodation_cache_file, 'r') as f:
                accomodation_cache = json.load(f)
        else:
            accomodation_cache = dict()
            
        formatted_accomodation_prompt = list_accomodation_prompt.format(end_user_specs=query, num_accomodations=str(num_accomodations))
        accomodations_string = Settings.llm.complete(formatted_accomodation_prompt)
        list_of_accomodations = str(accomodations_string).split(',')

        locs_to_be_cached = []
        json_response_str = "["
        for accomodation in tqdm(list_of_accomodations):
            
            # check if accomodation data is available in cache
            found_location = False
            if self.USE_CACHE and accomodation_cache_available:
                # get most similar locations from cache
                most_sim_loc_from_cache_list = get_most_similar_location_from_cache(Settings.embed_model, accomodation_cache, accomodation, top_k=check_match_from_cache_top_k)
                for loc in most_sim_loc_from_cache_list:
                    # check if most similar locations from cache refer to the target location
                    is_same_loc = Settings.llm.complete(same_location_check_prompt.format(loc_1=str(accomodation), loc_2=str(loc)))
                    if str(is_same_loc).lower().strip() == "yes":
                        accomodation_json_str = json.dumps(accomodation_cache.get(str(loc)))
                        json_response_str += str(accomodation_json_str)
                        json_response_str += ","
                        found_location = True
                        print(f"Found location {accomodation} in cache ...")
                        break
                if found_location:
                    continue
            
            output_is_json = False
            counter = 0
            try:
                while (not output_is_json) and (counter < max_json_try):
                    accomodation_json_str = accomodation_detail_top_orchestrator_agent.query(str(accomodation))
                    is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(accomodation_json_str)))
                    if str(is_json).lower().strip() == 'yes':
                        output_is_json = True
                        try:
                            accomodation_json_str = repair_json(accomodation_json_str)
                        except OSError:
                            accomodation_json_str = ""
                        if str(accomodation_json_str).strip() != "":
                            json_response_str += str(accomodation_json_str)
                            json_response_str += ","
                            # add new loc to cache
                            if self.USE_CACHE and accomodation_cache_available and (not found_location):
                                locs_to_be_cached.append(json.loads(accomodation_json_str))
                            break
                        else:
                            print(f"Error getting details in JSON format for accomodation: {accomodation}. JSON Repair failed.")
                    else:
                        counter += 1
                if counter == max_json_try:
                    print(f"Error getting details in JSON format for accomodation: {accomodation}. Max tries reached.")
            except:
                print(f"Error getting details for accomodation: {accomodation}")
        if str(json_response_str).endswith(','):
            json_response_str = str(json_response_str)[:-1]
        json_response_str += "]"

        # print()
        # print()
        # print(json_response_str)

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

        # create/update cache
        cache_updated = False
        for loc_dict in locs_to_be_cached:
            try:
                loc_name = loc_dict.get("Name", None)
                if loc_name is not None:
                    accomodation_cache[str(loc_name)] = loc_dict
                    print(f"Added {loc_name} to cache ...")
                    cache_updated = True
            except:
                print(f"Failed to add {loc_name} to cache.")
        if cache_updated:
            if accomodation_cache_available:
                os.remove(self.accomodation_cache_file)
            with open(self.accomodation_cache_file, 'w') as json_file:
                json.dump(accomodation_cache, json_file, indent=4)
        
        # print("============")
        # print(json_response)
        return json_response


    def get_destinations_json(self, query: str, num_spots: int = 5, max_json_try: int = 3, check_match_from_cache_top_k: int = 2):
        destination_cache_available = os.path.exists(self.destination_cache_file)
        if destination_cache_available:
            with open(self.destination_cache_file, 'r') as f:
                destination_cache = json.load(f)
        else:
            destination_cache = dict()
    
        formatted_destination_prompt = list_destination_prompt.format(end_user_specs=query, num_spots=str(num_spots))
        destinations_string = Settings.llm.complete(formatted_destination_prompt)
        list_of_destinations = str(destinations_string).split(',')

        locs_to_be_cached = []
        json_response_str = "["
        for destination in tqdm(list_of_destinations):
            
            # check if destination data is available in cache
            found_location = False
            if self.USE_CACHE and destination_cache_available:
                # get most similar locations from cache
                most_sim_loc_from_cache_list = get_most_similar_location_from_cache(Settings.embed_model, destination_cache, destination, top_k=check_match_from_cache_top_k)
                for loc in most_sim_loc_from_cache_list:
                    # check if most similar locations from cache refer to the target location
                    is_same_loc = Settings.llm.complete(same_location_check_prompt.format(loc_1=str(destination), loc_2=str(loc)))
                    if str(is_same_loc).lower().strip() == "yes":
                        destination_json_str = json.dumps(destination_cache.get(str(loc)))
                        json_response_str += str(destination_json_str)
                        json_response_str += ","
                        found_location = True
                        print(f"Found location {destination} in cache ...")
                        break
                if found_location:
                    continue
            
            output_is_json = False
            counter = 0
            try:
                while (not output_is_json) and (counter < max_json_try):
                    destination_json_str = destination_detail_top_orchestrator_agent.query(str(destination))
                    is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(destination_json_str)))
                    if str(is_json).lower().strip() == 'yes':
                        output_is_json = True
                        try:
                            destination_json_str = repair_json(destination_json_str)
                        except OSError:
                            destination_json_str = ""
                        if str(destination_json_str).strip() != "":
                            json_response_str += str(destination_json_str)
                            json_response_str += ","
                            # add new loc to cache
                            if self.USE_CACHE and destination_cache_available and (not found_location):
                                locs_to_be_cached.append(json.loads(destination_json_str))
                            break
                        else:
                            print(f"Error getting details in JSON format for destination: {destination}. JSON Repair failed.")
                    else:
                        counter += 1
                if counter == max_json_try:
                    print(f"Error getting details in JSON format for destination: {destination}. Max tries reached.")
            except:
                print(f"Error getting details for destination: {destination}")
        if str(json_response_str).endswith(','):
            json_response_str = str(json_response_str)[:-1]
        json_response_str += "]"

        # print()
        # print()
        # print(json_response_str)

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

        # create/update cache
        cache_updated = False
        for loc_dict in locs_to_be_cached:
            try:
                loc_name = loc_dict.get("Name", None)
                if loc_name is not None:
                    destination_cache[str(loc_name)] = loc_dict
                    print(f"Added {loc_name} to cache ...")
                    cache_updated = True
            except:
                print(f"Failed to add {loc_name} to cache.")
        if cache_updated:
            if destination_cache_available:
                os.remove(self.destination_cache_file)
            with open(self.destination_cache_file, 'w') as json_file:
                json.dump(destination_cache, json_file, indent=4)

        # print("============")
        # print(json_response)
        return json_response


    def check_query_detail(self, query: str):
        isDestination = True if str(Settings.llm.complete(isDestination_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isDuration    = True if str(Settings.llm.complete(isDuration_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isBudget      = True if str(Settings.llm.complete(isBudget_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isInterest    = True if str(Settings.llm.complete(isInterest_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        
        return {
            "isDestination": isDestination,
            "isDuration": isDuration,
            "isBudget": isBudget,
            "isInterest": isInterest
        }