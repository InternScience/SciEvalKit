import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json
import os


SERPER_KEY=os.environ.get('SERPER_KEY_ID')

successful_calls_search = 0
failed_calls_search = 0


@register_tool("google_search-google_search", allow_overwrite=True)
class Search(BaseTool):
    name = "google_search-google_search"
    description = "\n    Searches Google using the Serper API and returns a formatted string with the results.\n\n    This tool requires a SERPER_API_KEY to be provided.\n    The response is formatted as a Markdown string, summarizing the search results,\n    which is suitable for consumption by large language models.\n    "
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    def google_search_with_serp(self, query: str):
        global successful_calls_search, failed_calls_search
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        # conn = http.client.HTTPSConnection("google.serper.dev")
        url = "https://google.serper.dev/search"
        if contains_chinese_basic(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            }
        else:
            payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            }
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }
        
        for i in range(5):
            try:
                response = requests.request("POST", url, headers=headers, json=payload)
                
                break
            except Exception as e:
                print(e)
                if i == 4:
                    failed_calls_search += 1
                    return f"Google search Timeout, return None, Please try again later."
                continue
    
        # data = res.read()
        # results = json.loads(data.decode("utf-8"))
        results = response.json()
        # print(results)
        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "answerBox" in results:
                answer = results["answerBox"]
                title = answer.get("title", "Answer")
                snippet = answer.get("snippet", "N/A").replace("\n", " ")
                answer_box_info = f"## {title}\n{snippet}\n\n"
            else:
                answer_box_info = ""

            if "organic" in results:
                organic_info = ""
                # MODIFIED: 10 items by default (original 5 items)
                for i, item in enumerate(results["organic"][:10], 1):
                    title = item.get("title", "N/A")
                    link = item.get("link", "#")
                    snippet = item.get("snippet", "N/A").replace("\n", " ")
                    organic_info += f"{i}. **[{title}]({link})**\n   - {snippet}\n"
                # for page in results["organic"]:
                #     idx += 1
                #     date_published = ""
                #     if "date" in page:
                #         date_published = "\nDate published: " + page["date"]

                #     source = ""
                #     if "source" in page:
                #         source = "\nSource: " + page["source"]

                #     snippet = ""
                #     if "snippet" in page:
                #         snippet = "\n" + page["snippet"]

                #     redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                #     redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                #     web_snippets.append(redacted_version)

            content = f"# Search Results for: '{query}'\n\n {answer_box_info} ## Web Results\n\n" + organic_info
            successful_calls_search += 1
            return content
        except:
            failed_calls_search += 1
            return f"No results found for '{query}'. Try with a more general query."


    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response

