import json
from langchain.tools import tool
from tavily import AsyncTavilyClient
from ..utils.common import hashable_cache
from .. import config as CFG


@tool(
    description="""
Search the web for relevant information to answer user queries.
Input should be a single string representing the search query.
""".strip(),
    args_schema={
        "title": "web_search",
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to specifically include in the search results (optional). Maximum 300 domains.",
                "default": None,
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of domains to specifically exclude from the search results (optional). Maximum 150 domains.",
                "default": None,
            },
            "start_date": {
                "type": "string",
                "description": """
Filters search results to include only content published on or after this date.
        
Use this parameter when you need to:
- Find recent developments or updates on a topic
- Exclude outdated information from search results
- Focus on content within a specific timeframe
- Combine with end_date to create a custom date range

Format must be YYYY-MM-DD (e.g., "2024-01-15" for January 15, 2024).

Examples:
- "2024-01-01" - Results from January 1, 2024 onwards
- "2023-12-25" - Results from December 25, 2023 onwards

When combined with end_date, creates a precise date range filter.

Default is None (no start date restriction).
""".strip(),
                "default": None,
            },
            "end_date": {
                "type": "string",
                "description": """
Filters search results to include only content published on or before this date.
        
Use this parameter when you need to:
- Exclude content published after a certain date
- Study historical information or past events
- Research how topics were covered during specific time periods
- Combine with start_date to create a custom date range

Format must be YYYY-MM-DD (e.g., "2024-03-31" for March 31, 2024).

Examples:
- "2024-03-31" - Results up to and including March 31, 2024
- "2023-12-31" - Results up to and including December 31, 2023

When combined with start_date, creates a precise date range filter.
For example: start_date="2024-01-01", end_date="2024-03-31" 
returns results from Q1 2024 only.

Default is None (no end date restriction).
""".strip(),
                "default": None,
            },
        },
        "required": ["query"],
    }
)
@hashable_cache
async def web_search(
    query: str,
    include_domains: list[str] = None,
    exclude_domains: list[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> str:
    try:
        if include_domains is not None:
            if isinstance(include_domains, str):
                include_domains = [include_domains]
            elif not isinstance(include_domains, list):
                return "'include_domains' must be a string or a list of strings."
        if exclude_domains is not None:
            if isinstance(exclude_domains, str):
                exclude_domains = [exclude_domains]
            elif not isinstance(exclude_domains, list):
                return "'exclude_domains' must be a string or a list of strings."

        # 获取一个可用的 API key（带速率限制）
        api_key = CFG.TAVILY_API_KEY
        if api_key is None:
            return json.dumps({"error": "Web search is unavailable. Please do not attempt to use this tool again."})
        
        # 使用获取的 API key 创建客户端
        tavily_client = AsyncTavilyClient(api_key=api_key)
        
        response = await tavily_client.search(
            query=query,
            search_depth="advanced",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_date=start_date,
            end_date=end_date,
            timeout=30
        )
        if isinstance(response, dict):
            for k in ["follow_up_questions", "answer", "images", "response_time", "request_id"]:
                response.pop(k, None)
        return json.dumps(response, indent=4, ensure_ascii=False)
    except Exception as err:
        return json.dumps({"error": str(err)}, ensure_ascii=False)