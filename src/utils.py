from zhipuai import ZhipuAI
import json
import re
from openai import OpenAI
import globals

openai_client = OpenAI(
    base_url="you_base_url",
    api_key="your_api_key")
    
claude_client = OpenAI(api_key="your_api_key", 
                       base_url='you_base_url')

qianwen_client = OpenAI(
    api_key="your_api_key",
    base_url="you_base_url"
)

def LLM(query, model_name):
    if model_name.find("gpt") != -1:
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": query,
                },
            ]
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature = 0,
        )

    elif model_name.find('claude') != -1:
        response = claude_client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": query}],
                temperature = 0,
                max_tokens=2000,
            )
    
    elif model_name.find('qwen') != -1:
        response = qianwen_client.chat.completions.create(
            temperature = 0,
            model=model_name, # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},{"role": "user", "content": query}]
        )
    
    elif model_name.find('glm') != -1:
        client = ZhipuAI(api_key="your_api_key")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": query},
            ],
            stream=False,
            max_tokens=2000,
            temperature=0,
            do_sample=False,
        )

    ## 在这里记录 token 消耗数
    # 输入 token
    input_token = response.usage.prompt_tokens
    # 输出 token
    output_token = response.usage.completion_tokens
    # 总 token
    used_token = response.usage.total_tokens

    globals.this_question_input_token += input_token
    globals.this_question_output_token += output_token
    globals.this_question_total_token += used_token
    globals.total_token += used_token

    return response.choices[0].message.content.strip()
    

def prase_json_from_response(rsp: str):
    pattern = r"```json(.*?)```"
    print("输入的: ", rsp)
    rsp_json = None
    try:
        match = re.search(pattern, rsp, re.DOTALL)
        if match is not None:
            try:
                rsp_json = json.loads(match.group(1).strip().replace('(', '（').replace(')', '）'))
            except:
                rsp_json = json.loads(match.group(1).strip().replace('\'', '\"').replace('(', '（').replace(')', '）'))
        else:
            try:
                rsp = rsp.replace('(', '（').replace(')', '）')
                rsp_json = json.loads(rsp)
            except:
                rsp = rsp.replace('\'', '\"').replace('(', '（').replace(')', '）')
                rsp_json = json.loads(rsp)
        return rsp_json
    except json.JSONDecodeError as e:  # 因为太长解析不了
        try:
            match = re.search(r"\{(.*?)\}", rsp, re.DOTALL)
            if match:
                content = "[{" + match.group(1) + "}]"
                return json.loads(content)
        except:
            pass
        print(rsp)
        raise ("Json Decode Error: {error}".format(error=e))
    

from generated_tools import *
from prompt import *
from schema import *
from utils import *

def filter_table_and_tool(query, model_name):
    for attempt in range(3):
        try:
            table_prompt = TABLE_PROMPT.format(question=query, database_schema=database_schema)
            table_answer = LLM(table_prompt, model_name)
            table_response = prase_json_from_response(table_answer)
            table = table_response["名称"]
            break
        except:
            table = [
                "CompanyInfo",
                "CompanyRegister",
                "SubCompanyInfo",
                "LegalDoc",
                "CourtInfo",
                "CourtCode",
                "LawfirmInfo",
                "LawfirmLog",
                "AddrInfo",
                "LegalAbstract",
                "RestrictionCase",
                "FinalizedCase",
                "DishonestyCase",
                "AdministrativeCase"
            ]
    print(f"用到的table: {table}")
    if "CompanyInfo" not in table:
        table.append("CompanyInfo")
    if "AddrInfo" not in table:
        table.append("AddrInfo")

    table_used_prompt = ""
    used_tools = []
    for i in table:
        emun = table_map[i]
        one_prompt = f"""
{i}表格有下列字段:
{build_enum_list(emun)}
-------------------------------------
"""
        table_used_prompt += one_prompt + "\n"
        used_tools.extend(Tools_map[i])

    return used_tools, table_used_prompt

from termcolor import colored

def print_colored(text, color=None):
    """
    打印彩色字符串。

    参数：
    - text: 要打印的文本
    - color: 文本颜色（如 red, green, blue, yellow, cyan, magenta, white）
    - on_color: 背景颜色（如 on_red, on_green, on_blue）
    - attrs: 属性列表（如 ['bold', 'dark', 'underline', 'blink', 'reverse', 'concealed']）
    """
    try:
        print('\n\n\n')
        print(colored(text, color=color))
        print('\n\n\n')
    except Exception as e:
        print(f"错误: {e}")