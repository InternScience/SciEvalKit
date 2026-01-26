import os
import asyncio

from .. import config as CFG
from ..tools.web_search import web_search
from ..tools.data import ALL_DATA_TOOLS
from ..utils.agent import Agent

from .data_check import chat_data_check_agent


# For both plan and plan check agents
PLAN_PROMPT_NOTE = (
    "## Note: \n"
    "1. Experiments can only use CMIP (Coupled Model Intercomparison Project Phase) datasets and some observation datasets.\n"
    "2. The datasets used should be based on the user's request.\n"
    "3. It is not mandatory to use CMIP data. If there is no special instruction and the observation data can meet the requirements, the observation data should be used.\n"
    "4. If the required observation datasets are not available, you can use CMIP datasets instead. \n"
    "5. You can use the web search tool to search for relevant scientific definitions and calculation steps.\n"
    "6. You can use corresponding function tools to find the available cmip data, available observation data, CMIP model names, variable name abbreviations in CMIP, names of variable that can be derived, etc. "
    "(note that the variable names in the observation datasets have been processed to be the same as those in CMIP). \n"
    "7. If the variables required to complete the user task do not exist in CMIP, the corresponding variables should be calculated using existing variables.\n"
    "8. If not specified, monthly data is preferred.\n"
    "9. The plan should be as detailed and specific as possible, such as the time period, variable name, unit, etc. of the data used. "
    "However, it should not include path names, specific parameter configurations (such as color map or line thickness for plotting, etc.), specific execution operations or codes (such as what packages are used for data processing and plotting), etc.\n"
    "10. The plan should not include data download, sensitivity experiments, reproducibility, documentation, or future considerations.\n"
    "11. The plan cannot contain any preconceived conclusions. \n"
)
PLAN_NOTE_CNT = 11


PLAN_AGENT_TOOLS = [
    web_search
] + ALL_DATA_TOOLS


def create_plan_agent(logger=None):
    
    prompt = (
        "You are a geoscience experiment agent who is good at experiment planning. "
        "You understand the user's needs and output the corresponding experimental plan. "
        "Your plan should include what data to use, what preprocessing needs to be done on the data, what calculations to perform, what kind of figures to draw, etc. \n"
        f"{PLAN_PROMPT_NOTE}"
        f"{PLAN_NOTE_CNT+1}. Output the plan directly, don't output anything else."
    )

    agent = Agent(
        name="Plan Agent",
        model_settings=CFG.PLAN_MODEL_SETTING,
        system_prompt=prompt,
        tools=PLAN_AGENT_TOOLS,
        max_agent_iterations=CFG.PLAN_MAX_AGENT_ITERS,
        logger=logger,
        verbose=False
    )

    return agent


def create_plan_aggregation_agent(logger=None):

    prompt = (
        "You are a geoscience experiment agent who is good at checking and making experimental plans. "
        "You understand the user's needs and check the rationality and feasibility of the user's experimental plans and provide an improved plan. \n"
        f"{PLAN_PROMPT_NOTE}"
        f"{PLAN_NOTE_CNT+1}. Directly output your improved complete experimental plan, don't output anything else."
    )

    agent = Agent(
        name="Plan Aggregation Agent",
        model_settings=CFG.PLAN_AGGREGATION_MODEL_SETTING,
        system_prompt=prompt,
        tools=PLAN_AGENT_TOOLS,
        max_agent_iterations=CFG.PLAN_MAX_AGENT_ITERS,
        logger=logger
    )

    return agent


async def _single_plan(run_info, user_request, idx, plan_templates):
    
    plan_input = (
        f"<user_request>\n{user_request}\n</user_request>\n\n"
    )
    if plan_templates is not None:
        plan_input = (
            f"<some_reference_plans>\n"
            f"{plan_templates}\n\n"
            f"</some_reference_plans>\n\n"
        ) + plan_input + (
            f"The above content starts with some reference experimental plans for other possible similar tasks, "
            "followed by the current user's request. \n"
            f"Please provide an experimental plan according to the user's request.\n"
        )
    else:
        plan_input += (
            f"The above is the user's request. \n"
            f"Please provide an experimental plan according to the user's request.\n"
        )


    plan_agent = create_plan_agent()
    
    result = await plan_agent.chat(plan_input)
    cur_plan = result.content

    with open(f"{run_info['root']}/experiment_plans/plan_{idx}.md", "w", encoding='utf-8') as f:
        f.write(cur_plan)

    plan_agent.save_messages(f"{run_info['root']}/agent_logs/plan_agent_{idx}.json")
    
    return cur_plan


async def chat_plan_agent(run_info: dict):

    user_request = run_info["user_request"]
    logger = run_info['logger']
    root = run_info['root']

    os.makedirs(f"{root}/experiment_plans", exist_ok=True)

    plan_templates = None
    tasks = [
        _single_plan(run_info, user_request, i, plan_templates)
        for i in range(CFG.MAX_PLANS)
    ]

    try:
        plan_list = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        raise RuntimeError(f"Error in plan generation: {e}")
    
    assert len(plan_list) == CFG.MAX_PLANS

    experiment_plan_strings = ""
    for i in range(len(plan_list)):
        experiment_plan_strings += f"\n\n{'-'*5} Begin of plan {i} {'-'*5}\n\n"
        experiment_plan_strings += f"{plan_list[i]}"
        experiment_plan_strings += f"\n\n{'-'*5} End of plan {i} {'-'*5}\n\n"

    plan_aggregation_input = (
        f"<user_request>\n\n{user_request}\n\n</user_request>\n\n"
        f"<experimental_plans>\n\n"f"{experiment_plan_strings}\n\n</experimental_plans>\n\n"
        "The above are the user's request and some experimental plans.\n"
        "Now, please provide an improved experimental plan according to the user's request."
    )

    plan_aggregation_agent = create_plan_aggregation_agent(logger=logger)

    debug_round = 0

    while True:
        debug_round += 1
        if debug_round > CFG.MAX_PLAN_DEBUG_ROUND:
            raise RuntimeError(f"Data availablility check failed after maximum plan debug rounds ({CFG.MAX_PLAN_DEBUG_ROUND}).")
        
        result = await plan_aggregation_agent.chat(plan_aggregation_input)

        plan_aggregation_agent.save_messages(f"{root}/agent_logs/plan_aggregation_agent_round_{debug_round-1}.json")

        run_info["experiment_plan"] = result.content
        check_result = await chat_data_check_agent(run_info, save_round=debug_round-1)
        if check_result['pass']:
            break
            
        plan_aggregation_input = (
            f"{check_result['reason']}\n\n"
            "The data availability check failed due to the above reasons. "
            "You can use corresponding function tools to confirm the data information. "
            "Please provide a modified experimental plan."
        )
    with open(f"{root}/experiment_plans/final_plan.md", "w", encoding='utf-8') as f:
        f.write(result.content)

    return run_info