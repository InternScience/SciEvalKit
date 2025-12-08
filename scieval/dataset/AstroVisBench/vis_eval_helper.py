from google import genai
import base64
import pandas as pd
import os
import csv
import json
import tkinter as tk
from tkinter import ttk
import sys
import html
import pprint
import anthropic
from PIL import Image
import re
from glob import glob
from tqdm import tqdm
import copy
import argparse
from openai import OpenAI

prompts_csv = 'scieval/dataset/AstroVisBench/viseval_prompts.csv'



def safe_parse_json_from_claude(text):
    match = re.search(r'(\{\s*"Rationale"\s*:\s*".*?"\s*,\s*"Errors"\s*:\s*".*?"\s*\})', text, re.DOTALL)
    if not match:
        return "", ":x: No JSON block found"
    json_block = match.group(1)
    try:
        clean_block = re.sub(r'(?<!\\)[\x00-\x1F]', '', json_block)
        parsed = json.loads(clean_block)
        rationale = parsed.get("Rationale", "").replace("\n", " ").strip()
        errors = parsed.get("Errors", "").replace("\n", " ").strip()
        return rationale, errors
    except Exception as e:
        return "", f":x: JSON parse error: {e}"


import base64
import tempfile
import os


def query_model(prompt, gold_img, gen_img, gold_code, gen_code, vis_query, model):
    print(f"Querying using model: {getattr(model, 'model', 'Unknown')}...")

    def save_b64_to_temp(b64_str, suffix=".png"):
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]

        decoded_data = base64.b64decode(b64_str)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(decoded_data)
            return f.name

    gold_temp_path = save_b64_to_temp(gold_img)
    gen_temp_path = save_b64_to_temp(gen_img)

    try:
        inputs = [
            {'type': 'text', 'value': prompt},
            {'type': 'text', 'value': "This is the Visualization Query: "},
            {'type': 'text', 'value': vis_query},
            {'type': 'text', 'value': "This is the **ground-truth (reference)** image: "},
            {'type': 'image', 'value': gold_temp_path},

            {'type': 'text', 'value': "This is the code for the correct ground-truth image: "},
            {'type': 'text', 'value': gold_code},
            {'type': 'text', 'value': "This is the **under-test (assessed)** image:"},

            {'type': 'image', 'value': gen_temp_path},

            {'type': 'text', 'value': "This is the code for the under-test generated image: "},
            {'type': 'text', 'value': gen_code},
        ]
        response = model.generate(inputs)

        return response

    except Exception as e:
        print(f"Model Generation Error: {e}")
        raise e

    finally:
        if os.path.exists(gold_temp_path):
            os.remove(gold_temp_path)
        if os.path.exists(gen_temp_path):
            os.remove(gen_temp_path)


def extract_prompts(filename):
    prompts = []
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if "viseval new prompt" in row:
                    prompts.append(row["viseval new prompt"])
    except Exception as e:
        print(f":x: Error reading prompts: {e}")
    return prompts


def do_vis_eval(all_queries, model):
    prompt_template = extract_prompts(prompts_csv)[0]
    new_queries = []
    for query in tqdm(all_queries):

        print(query['visualization_test'].keys())

        if not query['visualization_test']['vis_success'] or len(query['visualization_test']['gen_vis_list']) != 1:
            copy_query = copy.deepcopy(query)
            copy_query['visualization_llm_eval'] = {
                "rationale": "",
                "errors": "Crash" if not query['visualization_test']['vis_success'] else "VisFail",
            }
            new_queries.append(copy_query)
            continue

        try:
            # 注意：query_model 现在直接返回 text 字符串了，不需要再 .content[0].text
            raw_text = query_model(
                prompt_template,
                query["gt_visualization"].split('base64,')[1],
                query['visualization_test']['gen_vis_list'][0].split('base64,')[1],
                query["visualization_gt_code"],
                query["visualization_gen_code"],
                query["visualization_query"],
                model = model
            )
            rationale, errors = safe_parse_json_from_claude(raw_text)
        except Exception as e:
            rationale = ""
            errors = f":x: Exception: {e}"
        copy_query = copy.deepcopy(query)
        copy_query['visualization_llm_eval'] = {
            "rationale": rationale,
            "errors": errors
        }
        new_queries.append(copy_query)
    return new_queries

