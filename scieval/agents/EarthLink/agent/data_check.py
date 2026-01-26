import json
from .. import config as CFG
from ..tools.data import ALL_DATA_TOOLS
from ..utils.common import extract_code_blocks
from ..utils.agent import Agent


def create_data_check_agent(logger=None):
    
    prompt = (
        "You are a geoscience experiment agent. "
        "You understand the user's needs and call related functions to confirm current available CMIP data and observational data information. "
        "Then judge whether the current available data can meet the user's needs. \n"
        "## Note: \n"
        "1. Do not guess the available data information, you should call the function to obtain the available data information.\n"
        "2. Only judge based on data information obtained from the function, we can not use any external web data.\n"
        "3. You can use the corresponding function tools to find the variable name abbreviations in CMIP, CMIP model names, names of variable that can be derived, avaiable reference datasets, etc.\n"
        "(note that the variable names in the observation datasets have been processed to be the same as those in CMIP). \n"
        "4. If it can or partially can (for example, there are multiple solutions in the plan, and some solutions can be met), the check passes and the reason can be concise.\n"
        "If it cannot (for example, the necessary observation data is missing or all solutions in all plans cannot be met), the check fails and you need to give as detailed a reason as possible.\n"
        "## Ouput format:\n"
        "```json\n{\"pass\": true or false, \"reason\": \"...\"}\n```"
    )

    agent = Agent(
        name="Data Check Agent",
        model_settings=CFG.DATA_CHECK_MODEL_SETTING,
        system_prompt=prompt,
        tools=ALL_DATA_TOOLS,
        max_agent_iterations=CFG.DATA_CHECK_MAX_AGENT_ITERS,
        logger=logger,
    )
    return agent


async def chat_data_check_agent(run_info: dict, save_round: int = 0) -> dict:

    user_request = run_info["user_request"]
    experiment_plan = run_info["experiment_plan"]
    logger = run_info['logger']

    data_check_agent = create_data_check_agent(logger=logger)

    data_check_input = (
        f"<user_request>\n\n{user_request}\n\n</user_request>\n\n"
        f"<experiment_plan>\n\n{experiment_plan}\n\n</experiment_plan>\n\n"
        "Now, please judge whether the available data can meet the user's needs. "
    )
    
    max_try = 20
    cur_try = 0
    while True:
        if cur_try >= max_try:
            raise RuntimeError(
                f"Failed to get valid JSON output from the data check agent after {max_try} tries."
            )

        cur_try += 1
        
        result = await data_check_agent.chat(data_check_input)
    
        try:
            code = extract_code_blocks(result.content, language='json')
            if code is not None:
                output = json.loads(code)
            else:
                output = json.loads(result.content)
            if ("pass" not in output):
                raise ValueError("Output JSON must contain 'pass' and 'reason' keys.")            
            if (not output['pass']) and ("reason" not in output):
                raise ValueError("If 'pass' is false, output JSON must contain 'reason' key.")
            break
        except Exception as e:
            data_check_input = (
                f"Output is not a valid JSON or does not contain the required keys. Error: \n{e}\n"
                "Please output in the following format:\n"
                "```json\n{\"pass\": true or false, \"reason\": \"...\"}\n```"
            )
    data_check_agent.save_messages(f"{run_info['root']}/agent_logs/data_check_agent_round_{save_round}.json")

    return output