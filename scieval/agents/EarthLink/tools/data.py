import os
import re
import json
import difflib
import pandas as pd
from collections import Counter
from langchain.tools import tool

from .. import data_info
from ..data_info import (
    CMIP6_ACTIVITIES,
    CMIP6_EXPERIMENTS,
    CMIP6_FREQUENCES,
    CMIP6_TABLES,
    CMIP6_MODELS,
    CMIP6_VARIABLES,
    DERIVED_VARIABLES,
    OBS4MIPS_MODELS,
    OBS_DATA_INFO,
    FX_VARIABLES,
    AVAIL_CMIP6_DATA,
    AVAIL_OBS_DATA,
    AVAIL_OBS4MIPS_DATA
)
from ..utils.common import CompactListEncoder, cosine_similarity, hashable_cache
from ..utils.embedding import embed_query


@tool(parse_docstring=True)
@hashable_cache
def list_available_cmip6_files(activity: str, experiment: str, mip: str, model_name: str, 
                                variable_short_name: str) -> str:
    """ List available CMIP6 files based on the provided parameters.

    Args:
        activity (str): The activity id in CMIP6 project (e.g., "CMIP", "DAMIP", "ScenarioMIP").
        experiment (str): The experiment id in CMIP6 project (e.g., ["1pctCO2", "abrupt-4xCO2", "historical", "piControl"] in activity "CMIP"; ["ssp119", "ssp126", ...] in activity "ScenarioMIP", etc.).
        mip (str): The MIP name of the data (e.g., "Amon", "Omon", "Lmon", etc.).
        model_name (str): The model name (e.g., "ACCESS-CM2", "BCC-CSM2-MR", ...).
        variable_short_name (str): The short name of the variable (e.g., "tos", "tas", "pr", etc.).
    """

    # print_colored_text(f"Listing available CMIP6 files for activity: {activity}, experiment: {experiment}, frequency: {frequency}, models: {model_name_list}, variable: {variable_short_name}", 'b', flush=True)
    msg = []
    if activity not in CMIP6_ACTIVITIES:
         msg.append(f"Unsupported activity: {activity}. Supported activities: {sorted(list(CMIP6_ACTIVITIES.keys()))}.")
    if experiment not in CMIP6_EXPERIMENTS:
        similar_experiments = difflib.get_close_matches(experiment, list(CMIP6_EXPERIMENTS.keys()), n=2, cutoff=0.6)
        if len(similar_experiments) > 0:
            msg.append(f"Unsupported experiment: {experiment}. Did you mean: {similar_experiments}?")
        else:
            msg.append(f"Unsupported experiment: {experiment}.")
    
    if mip not in CMIP6_TABLES:
        msg.append(f"Unsupported MIP: {mip}. Supported MIPs: {sorted(list(CMIP6_TABLES.keys()))}.")

    if model_name not in CMIP6_MODELS:
        similar_models = difflib.get_close_matches(model_name, list(CMIP6_MODELS.keys()), n=2, cutoff=0.6)
        if len(similar_models) > 0:
            msg.append(f"Unsupported model: '{model_name}'. Do you mean: {similar_models}?")
        else:
            msg.append(f"Unsupported model: '{model_name}'. Please check the model name in the CMIP6 model list.")
    
    if variable_short_name not in CMIP6_VARIABLES:
        msg.append(f"Unsupported CMIP6 variable: {variable_short_name}. It may be a derived variable or a wrong variable name abbreviation. Please check the variable name abbreviation in the CMIP6 variable list.")
    
    if len(msg) > 0:
        return "Error in parameters:\n" + "\n".join(msg) + "\nPlease check the parameters and try again." # You can use the file search tool or web search tool to find the related information."
    
    # assert activity != "DCPP"

    if variable_short_name in FX_VARIABLES:
        return f"Variable '{variable_short_name}' is a fixed variable in CMIP6, which does not have time-varying data. It will be automatically loaded in subsequent processes and does not need to be explicitly specified."

    results_all = {
        "activity": activity,
        "experiment": experiment,
        "model_name": model_name,
        "mip": mip,
        "variable_info": {
            "short_name": variable_short_name,
            "description": CMIP6_VARIABLES[variable_short_name]["description"],
            "units": CMIP6_VARIABLES[variable_short_name]["units"],
        }
    }

    if "positive" in CMIP6_VARIABLES[variable_short_name]:
        results_all["variable_info"]["positive"] = CMIP6_VARIABLES[variable_short_name]["positive"]

    available_data_info = AVAIL_CMIP6_DATA.get(activity, {}).get(experiment, {}).get(model_name, {}).get(mip, {}).get(variable_short_name, {})
    has_sub_exp = False
    if len(available_data_info) > 0:
        grids = list(available_data_info.keys())
        if len(grids) > 0:
            grid = grids[0]
            mem_or_subexps = list(available_data_info[grid].keys())
            if len(mem_or_subexps) > 0:
                if isinstance(available_data_info[grid][mem_or_subexps[0]], dict):
                    has_sub_exp = True
        
    if has_sub_exp:
        results_all["available_grid_sub-experiment_members_times"] = available_data_info
    else:
        results_all["available_grid_members_times"] = available_data_info

    compacted_list_json = json.dumps(results_all, indent=2, cls=CompactListEncoder)
    return f"Available CMIP6 data information:\n```json\n{compacted_list_json}\n```"


@tool(parse_docstring=True)
@hashable_cache
def list_available_observation_files(dataset: str, variable_short_name: str) -> str:
    """ List available observational data files (including data in obs4MIPs) based on the provided parameters.

    Args:
        dataset (str): The name of the observational dataset (e.g., "ERA5", "CERES-EBAF", etc.).
        variable_short_name (str): The short name of the variable (e.g., "tos", "tas", "pr", etc.). This can also be a derived variable name (e.g., "alb", "sm", etc.).
    """

    # print_colored_text(f"Listing available observational files for dataset: {dataset}, variable: {variable_short_name}", 'b', flush=True)
    if dataset not in OBS_DATA_INFO and dataset not in OBS4MIPS_MODELS:
        similar_data_names = difflib.get_close_matches(dataset, list(OBS_DATA_INFO.keys()), n=2, cutoff=0.6)
        msg = ""
        if len(similar_data_names) > 0:
            msg += f"Did you mean: {similar_data_names} in supported observational datasets.\n"
        similar_data_names = difflib.get_close_matches(dataset, list(OBS4MIPS_MODELS), n=2, cutoff=0.6)
        if len(similar_data_names) > 0:
            msg += f"Did you mean: {similar_data_names} in obs4MIPs datasets.\n"
        
        return f"Unsupported observational dataset: {dataset}. {msg} Please check the parameters and try again." #You can use the file search tool or web search tool to find the related information."

    info = {"dataset": dataset}
    if dataset in OBS_DATA_INFO:
        
        var_avail_data = [d for d in AVAIL_OBS_DATA if variable_short_name in AVAIL_OBS_DATA[d]["available_variables"]]

        if dataset not in AVAIL_OBS_DATA:
            similar_data_names = difflib.get_close_matches(dataset, list(AVAIL_OBS_DATA.keys()), n=2, cutoff=0.6)
            similar_data_names = [d for d in similar_data_names if d in var_avail_data]

            if len(similar_data_names) > 0:
                msg = f"Observational dataset {dataset} is not supported currently. Did you mean: {similar_data_names} in supported observational datasets?"
            else:
                msg = f"Observational dataset {dataset} is not supported currently. Try to select another dataset."
            if len(var_avail_data) > 0:
                msg += f"\nThe below datasets have the variable '{variable_short_name}' available:\n{var_avail_data}"
            return msg
        
        var_info = AVAIL_OBS_DATA[dataset]["available_variables"]
        info.update({
            "project": "OBS",
            # "project": AVAIL_OBS_DATA[dataset]["project"],
            # "tier": AVAIL_OBS_DATA[dataset]["tier"],
            # "type": AVAIL_OBS_DATA[dataset]["type"]
        })
    else:        
        var_avail_data = [d for d in AVAIL_OBS4MIPS_DATA if variable_short_name in AVAIL_OBS4MIPS_DATA[d]]

        if dataset not in AVAIL_OBS4MIPS_DATA:
            similar_data_names = difflib.get_close_matches(dataset, list(AVAIL_OBS4MIPS_DATA.keys()), n=2, cutoff=0.6)
            similar_data_names = [d for d in similar_data_names if d in var_avail_data]

            if len(similar_data_names) > 0:
                msg = f"Observational dataset {dataset} is not supported currently. Did you mean: {similar_data_names} in supported observational datasets?"
            else:
                msg = f"Observational dataset {dataset} is not supported currently. Try to select another dataset."
            if len(var_avail_data) > 0:
                msg += f"\nThe below datasets have the variable '{variable_short_name}' available:\n{var_avail_data}"

            return msg
        
        var_info = AVAIL_OBS4MIPS_DATA[dataset]
        info["project"] = "obs4MIPs"


    if variable_short_name not in var_info:
        msg = f"Unsupported variable '{variable_short_name}' in dataset {dataset}. Supported variables: {sorted(list(var_info.keys()))}. "
        if len(var_avail_data) > 0:
            msg += f"\nThe below datasets have the variable '{variable_short_name}' available:\n{var_avail_data}\n"
        msg += "Please check the parameters and try again." # You can use the file search tool or web search tool to find the related information."
        return msg

    var_info = var_info[variable_short_name]
    var_info["short_name"] = variable_short_name

    info["variable_info"] = var_info
    info_string = json.dumps(info, indent=2, cls=CompactListEncoder)

    return f"Available observational data information:\n```json\n{info_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def verify_cmip6_model_name(model_name_list: list[str]) -> str:
    """ Verify the CMIP6 model names are correct.

    Args:
        model_name_list (list[str]): A list of model names to verify.
    """
    
    # print_colored_text(f"Validating CMIP6 model names: {model_name_list}", 'b', flush=True)

    if isinstance(model_name_list, str):
        model_name_list = [model_name_list]

    model_list = list(CMIP6_MODELS.keys())
    results = {}

    for model in model_name_list:
        if model in CMIP6_MODELS:
            results[model] = "Valid"
        else:
            similar_models = difflib.get_close_matches(model, model_list, n=3, cutoff=0.3)
            similar_models.extend([m for m in CMIP6_MODELS if m.lower().startswith(model.lower())])
            similar_models = list(set(similar_models))
            if len(similar_models) > 0:
                results[model] = f"Invalid model name. Did you mean: {similar_models}?"
            else:
                results[model] = "Invalid model name."

    if len(results) == 0:
        return "No model names provided to validate."

    result_string = json.dumps(results, indent=2, cls=CompactListEncoder)
    return f"Model name verification results:\n```json\n{result_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def verify_variable_name_and_get_info(variable_short_name_list: list[str]) -> str:
    """ Verify the variable short names are correct and get their infomation (variable description, units, mips, etc.).

    Args:
        variable_short_name_list (list[str]): A list of variable short names to verify and get information for.
    """

    # print_colored_text(f"Validating variable short names: {variable_short_name_list}", 'b', flush=True)

    if isinstance(variable_short_name_list, str):
        variable_short_name_list = [variable_short_name_list]

    variable_list = list(CMIP6_VARIABLES.keys()) + list(DERIVED_VARIABLES.keys())
    results = {}

    for variable in variable_short_name_list:
        if variable in CMIP6_VARIABLES:
            var_info = CMIP6_VARIABLES[variable]
            results[variable] = {
                "valid": True,
                "is_derived_variable": False  
            }
            if len(var_info["mip_filted"]):
                results[variable]["valid_mip(s)"] = var_info["mip_filted"]
            
        elif variable in DERIVED_VARIABLES:
            var_info = DERIVED_VARIABLES[variable]
            results[variable] = {
                "valid": True,
                "is_derived_variable": True,
                "derived_from": DERIVED_VARIABLES[variable]["derived_from"]
            }
        else:
            similar_variables = difflib.get_close_matches(variable, variable_list, n=3, cutoff=0.5)
            similar_variables.extend([v for v in variable_list if v.lower().startswith(variable.lower())])
            similar_variables = list(set(similar_variables))
            results[variable] = {
                "valid": False,
            }
            if len(similar_variables) > 0:
                desc = [f"{v} ({CMIP6_VARIABLES[v]['description']})" if v in CMIP6_VARIABLES else f"{v} (derived variable; {DERIVED_VARIABLES[v]['description']})" for v in similar_variables]
                results[variable]["suggestions"] = desc
            continue
        
        results[variable].update({
            "description": var_info["description"],
            "units": var_info["units"],
        })

        if "positive" in var_info:
            results[variable]["positive"] = var_info["positive"]

    if len(results) == 0:
        return "No variable names provided to validate."

    result_string = json.dumps(results, indent=2) #, cls=CompactListEncoder)
    return f"Variable name verification results:\n```json\n{result_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def search_variable_short_name(variable_description: str) -> list[str]:
    """ Search for the variable short name based on the short description provided. Example: "Sea Surface Temperature" will return "tos".
    
    Args:
        variable_description (str): The short description of the variable to search for.
    """

    # print_colored_text(f"Searching variable short name for description: {variable_description}", 'b', flush=True)

    if not variable_description:
        return "No variable description provided to search."
    
    var_embedding_file = os.path.join(os.path.dirname(data_info.__file__), 'variables_embedding.jsonl')
    df = pd.read_json(var_embedding_file, orient='records', lines=True)
    query_embedding = embed_query(variable_description)
    df['similarity'] = df['desc_embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    df = df.loc[df['similarity'] > 0.6].sort_values(by='similarity', ascending=False)[:10]

    if len(df) == 0:
        return f"No variable found matching the description: {variable_description}"

    desc = {}
    for _, row in df.iterrows():
        var = row['name']
        desc[var] = {
            "description": row['description'],
            "similarity": row['similarity'],
        }
        if var in CMIP6_VARIABLES:
            desc[var]["is_derived_variable"] = False
        elif var in DERIVED_VARIABLES:
            desc[var]["is_derived_variable"] = True
            desc[var]["derived_from"] = DERIVED_VARIABLES[var]["derived_from"]

    result_string = json.dumps(desc, indent=2, cls=CompactListEncoder)
    return f"Variable short names matching the description:\n```json\n{result_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def verify_cmip6_mip_name(mip_name_list: list[str]) -> str:
    """ Verify the mip names (like Amon, Omon, Lmon, etc.) are correct.

    Args:
        mip_name_list (list[str]): A list of MIP names to verify.
    """

    # print_colored_text(f"Validating MIP names: {mip_name_list}", 'b', flush=True)

    if isinstance(mip_name_list, str):
        mip_name_list = [mip_name_list]

    mip_list = list(CMIP6_TABLES.keys())
    results = {}

    for mip in mip_name_list:
        if mip in CMIP6_TABLES:
            results[mip] = "Valid"
        else:
            similar_mips = difflib.get_close_matches(mip, mip_list, n=3, cutoff=0.5)
            similar_mips.extend([m for m in CMIP6_TABLES if m.lower().startswith(mip.lower())])
            similar_mips = list(set(similar_mips))
            if len(similar_mips) > 0:
                results[mip] = f"Invalid MIP name. Did you mean: {similar_mips}?"
            else:
                results[mip] = "Invalid MIP name."
    
    if len(results) == 0:
        return "No MIP names provided to validate."

    result_string = json.dumps(results, indent=2, cls=CompactListEncoder)
    return f"MIP name verification results:\n```json\n{result_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def verify_cmip6_activity_name(activity_name_list: list[str]) -> str:
    """ Verify the activity names (like CMIP, DAMIP, ScenarioMIP, etc.) are correct.

    Args:
        activity_name_list (list[str]): A list of activity names to verify.
    """

    # print_colored_text(f"Validating activity names: {activity_name_list}", 'b', flush=True)

    if isinstance(activity_name_list, str):
        activity_name_list = [activity_name_list]

    activity_list = list(CMIP6_ACTIVITIES.keys())
    results = {}

    for activity in activity_name_list:
        if activity in CMIP6_ACTIVITIES:
            results[activity] = "Valid"
        else:
            similar_activities = difflib.get_close_matches(activity, activity_list, n=3, cutoff=0.5)
            similar_activities.extend([a for a in CMIP6_ACTIVITIES if a.lower().startswith(activity.lower())])
            similar_activities = list(set(similar_activities))
            if len(similar_activities) > 0:
                results[activity] = f"Invalid activity name. Did you mean: {similar_activities}?"
            else:
                results[activity] = "Invalid activity name."
    
    if len(results) == 0:
        return "No activity names provided to validate."

    result_string = json.dumps(results, indent=2, cls=CompactListEncoder)
    return f"Activity name verification results:\n```json\n{result_string}\n```"


@tool(parse_docstring=True)
@hashable_cache
def verify_cmip6_experiment_name(experiment_name_list: list[str]) -> str:
    """ Verify the experiment names (like historical, piControl, ssp126, etc.) are correct and get their simple information.

    Args:
        experiment_name_list (list[str]): A list of experiment names to verify.
    """

    # print_colored_text(f"Validating experiment names: {experiment_name_list}", 'b', flush=True)

    if isinstance(experiment_name_list, str):
        experiment_name_list = [experiment_name_list]

    experiment_list = list(CMIP6_EXPERIMENTS.keys())
    results = {}

    for experiment in experiment_name_list:
        if experiment in CMIP6_EXPERIMENTS:
            results[experiment] = {
                "valid": True,
                "description": CMIP6_EXPERIMENTS[experiment]["description"],
                "activity_id": CMIP6_EXPERIMENTS[experiment]["activity_id"],
            }
        else:
            similar_experiments = difflib.get_close_matches(experiment, experiment_list, n=3, cutoff=0.5)
            similar_experiments.extend([e for e in CMIP6_EXPERIMENTS if e.lower().startswith(experiment.lower())])
            similar_experiments = list(set(similar_experiments))
            results[experiment] = {
                "valid": False,
            }
            if len(similar_experiments) > 0:
                results[experiment]["suggestions"] = similar_experiments
    
    if len(results) == 0:
        return "No experiment names provided to validate."

    result_string = json.dumps(results, indent=2, cls=CompactListEncoder)
    return f"Experiment name verification results:\n```json\n{result_string}\n```"


ALL_DATA_TOOLS = [
    list_available_cmip6_files,
    list_available_observation_files,
    verify_cmip6_model_name,
    verify_variable_name_and_get_info,
    search_variable_short_name,
    verify_cmip6_mip_name,
    verify_cmip6_activity_name,
    verify_cmip6_experiment_name
]