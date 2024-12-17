import re, string, os
from typing import List, Union, Literal
from enum import Enum
# import tiktoken

from langchain.prompts import PromptTemplate
from prompts import *
from fewshots import *
import requests
import json
from src.utils import *

base_url = "https://comm.chatglm.cn/law_api"
from src.generated_tools import *

def format_input(name, input):
    """
    input: json/list/str
    return: json
    """
    if name == "legal_case_retriever" or name == "legal_article_retriever" or name == "legal_knowledge_retriever":
        default_k = 1 if name == "legal_case_retriever" else 5
        if isinstance(input, dict):
            if 'k' not in input.keys():
                return {"identifier": input['identifier'], "k": default_k}
        if isinstance(input, str):
            return {"identifier": input, "k": default_k}
    elif name == "get_sum" or name == "get_multiplication":
        if isinstance(input, list):
            return {"identifier": input}
    elif name == "get_rank":
        if isinstance(input, list):
            return {"identifier": input, "is_desc": 'False'}
    else:
        return input

def post_request(name, input):
    headers = {
        'Content-Type': 'application/json',
        "Authorization": "Bearer 0F7354720533CAACBA57DCA97F89069CB80C78B9BB769786"
    }
    print("调用工具中...", name, input)
    input = format_input(input)

    response = requests.post(f"{base_url}{name}", headers=headers, json=input)
    if response.status_code == 200:
        print("调用成功")
        ## TODO: 转化为文本
        return response.json()
    else:
        print("调用失败")
        ## TODO: 返回错误信息 工具调用发生错误, 请检查传入工具的参数是否正确!
        return response.status_code, response.text
    
class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'

class ReactAgent:
    def __init__(self,
                 model_name: str,
                 question: str,
                 tools: str,
                 tool_names: str,   
                 table_used_prompt: str,      
                 key: str = "",      # 真实答案              
                 max_steps: int = 10,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 ) -> None:
        self.model_name = model_name
        self.question = question
        self.answer = ''                                          # 模型的答案
        # self.key = key
        self.key = ''
        self.tools = tools
        self.tool_names = tool_names
        self.table_used_prompt = table_used_prompt
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = REACT_EXAMPLE
        
        # self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():     # 没有达到最大步数, 没有完成
            self.step()
    
    def step(self) -> None:                                        # react的每一步
        # Think
        self.scratchpad += f'\n思考 {self.step_n}: '
        think = self.prompt_agent()
        print_colored(f"思考 {self.step_n}: {think}", color='red')
        self.scratchpad += ' ' + think               # here 思考
        # print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\n行动 {self.step_n}: '
        action = self.prompt_agent()
        print_colored(f"行动 {self.step_n}: {action}", color='blue')
        self.scratchpad += ' ' + action
        action_type, action_input = parse_action(action)
        # print(self.scratchpad.split('\n')[-1])

        print_colored(f"行动在这里: {action_type}: {action_input}", color="green")


        print("等待行动结果....")
        
        # Observe
        self.scratchpad += f'\n观察 {self.step_n}: '
        # 根据行为分类

        if action_type == 'Final Answer':
            self.scratchpad = '\n'.join(self.scratchpad.split('\n')[:-1]) # 去掉最后拖尾的 观察
            self.answer = action_input
            # if self.is_correct():
            #     self.scratchpad += 'Answer is CORRECT'
            # else: 
            #     self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return
        elif action_type == "":
            self.scratchpad += f"""行动格式非法, 行动一次只能调用一次工具, 行动必须按照以下json格式输出, 可以被Python json.loads函数解析: ```json{{"action": $TOOL_NAME,"action_input": $INPUT}}```, 请你反思并尝试纠正."""
        else:
            try:
                result = post_request(action_type, action_input)
                self.scratchpad += str(result)

                if result == "No data found for the specified identifier.":
                    self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具; 二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'
                elif result == "One or more specified columns do not exist.":
                    self.scratchpad += '原因可能是: 调用工具时传入的columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'

            except Exception as e:
                self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具;  二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'

        print_colored(self.scratchpad.split('\n')[-1], color="yellow")

        # 下一步
        self.step_n += 1

    def prompt_agent(self) -> str:
        # print("prompt is here: ", self._build_agent_prompt())
        return format_step(LLM(self._build_agent_prompt(), self.model_name))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            # examples = self.react_examples,        # few shot
                            examples = self.react_examples,
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            tool_names = self.tool_names,
                            question = self.question,              # 问题
                            scratchpad = self.scratchpad)          # 过程记录
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        # return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
        return (self.step_n > self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1             # 当前步
        self.finished = False       # 是否完成
        self.scratchpad: str = ''   # 过程

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

class PSAgent:
    def __init__(self,
                 model_name: str, 
                 question: str,
                 tools: str,
                 tool_names: str,   
                 table_used_prompt: str,      
                 key: str = "",      # 真实答案              
                 max_steps: int = 10,
                 ) -> None:
        self.model_name = model_name
        self.question = question
        self.answer = ''                                          # 模型的答案
        self.key = ''
        self.tools = tools
        self.tool_names = tool_names
        self.table_used_prompt = table_used_prompt
        self.max_steps = max_steps
        self.plan_prompt = ps_plan_prompt
        self.solve_prompt = ps_solve_prompt

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        plan = format_step(LLM(self._build_plan_prompt(), self.model_name)).replace("：", ": ")
        print_colored(f"{plan}", color='red')

        plan = parse_plan(plan)
        print_colored(f"{plan}", color='green')

        print(plan)

        for step, p in enumerate(plan):
            action = format_step(LLM(self._build_solve_prompt(p), self.model_name))
            print_colored(f"步骤{step + 1}: {action}", color='blue')

            action_type, action_input = parse_action(action)
            print_colored(f"行动在这里: {action_type}: {action_input}", color="green")
            print("等待行动结果....")

            self.scratchpad += f"\n第{step + 1}步计划: {p} 执行结果: "

            if action_type == 'Final Answer':
                self.answer = action_input
                self.scratchpad += str(action_input)
                self.finished = True
                return
            elif action_type == "":
                self.scratchpad += f"""行动格式非法, 行动一次只能调用一次工具, 行动必须按照以下json格式输出, 可以被Python json.loads函数解析: ```json{{"action": $TOOL_NAME,"action_input": $INPUT}}```, 请你反思并尝试纠正."""
            else:
                try:
                    result = post_request(action_type, action_input)
                    self.scratchpad += str(result)

                    if result == "No data found for the specified identifier.":
                        self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具; 二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'
                    elif result == "One or more specified columns do not exist.":
                        self.scratchpad += '原因可能是: 调用工具时传入的columns列表中包含了要查询的表格中未出现的字段., 请你反思并尝试纠正.'
                except Exception as e:
                    self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具; 二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'

            print_colored(self.scratchpad.split('\n')[-1], color="yellow")

    def prompt_agent(self) -> str:
        return format_step(LLM(self._build_agent_prompt(), self.model_name))
    
    def _build_plan_prompt(self) -> str:
        return self.plan_prompt.format(
                            examples = PLAN_SOLVE_plan_EXAMPLE,       
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            question = self.question,              # 问题
                )

    def _build_solve_prompt(self, plan) -> str:
        return self.solve_prompt.format(
                            examples = PLAN_SOLVE_solve_EXAMPLE,        # few shot
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            tool_names = self.tool_names,
                            plan = plan,              # 问题
                            scratchpad = self.scratchpad)          # 过程记录
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return (self.step_n > self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1             # 当前步
        self.finished = False       # 是否完成
        self.scratchpad: str = ''   # 过程

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

class PEAgent:
    def __init__(self,
                 model_name: str,
                 question: str,
                 tools: str,
                 tool_names: str,   
                 table_used_prompt: str,      
                 key: str = "",      # 真实答案              
                 max_steps: int = 10
                 ) -> None:
        self.model_name = model_name
        self.question = question
        self.answer = ''                                          # 模型的答案
        self.key = ''
        self.tools = tools
        self.tool_names = tool_names
        self.table_used_prompt = table_used_prompt
        self.max_steps = max_steps
        self.plan_prompt = ps_plan_prompt
        self.solve_prompt = ps_solve_prompt
        self.replan_prompt = pe_replan_prompt

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        plan = format_step(LLM(self._build_plan_prompt(), self.model_name)).replace("：", ": ")
        print_colored(f"原始计划输出: {plan}", color='red')

        old_plan = plan

        plan = parse_plan(plan)

        step = 0
        while (step < len(plan) and step < 10):
            p = plan[step]
            step += 1

            print_colored(f"当前执行到的步骤: 第{step}步: {p}", color='red')
            
            action = format_step(LLM(self._build_solve_prompt(p), self.model_name))
            print_colored(f"步骤{step}: {action}", color='blue')

            action_type, action_input = parse_action(action)
            print_colored(f"行动在这里: {action_type}: {action_input}", color="green")
            print("等待行动结果....")

            self.scratchpad += f"\n第{step}步计划: {p} 执行结果: "

            if action_type == 'Final Answer':
                self.answer = action_input
                self.scratchpad += str(action_input)
                self.finished = True
                return
            elif action_type == "":
                self.scratchpad += f"""行动格式非法, 行动一次只能调用一次工具, 行动必须按照以下json格式输出, 可以被Python json.loads函数解析: ```json{{"action": $TOOL_NAME,"action_input": $INPUT}}```, 请你反思并尝试纠正."""
            else:
                try:
                    result = post_request(action_type, action_input)
                    self.scratchpad += str(result)

                    if result == "No data found for the specified identifier.":
                        self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具; 二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'
                    elif result == "One or more specified columns do not exist.":
                        self.scratchpad += '原因可能是: 调用工具时传入的columns列表中包含了要查询的表格中未出现的字段., 请你反思并尝试纠正.'
                except Exception as e:
                    self.scratchpad += '原因可能是: 一, 调用的工具不合适, 尝试使用其他工具; 二, 调用工具时传入的参数可能存在非法值, 例如identifier的值非法(与工具的输入要求不符), 或者columns列表中包含了要查询的表格中未出现的字段.请你反思并尝试纠正.'

            print_colored(self.scratchpad.split('\n')[-1], color="yellow")

            print("重新规划...")
            plan = format_step(LLM(self._build_replan_prompt(old_plan), self.model_name))

            print_colored(f"重新规划的计划: {plan}", color='red')

            old_plan = plan

            plan = parse_plan(plan)

    def prompt_agent(self) -> str:
        return format_step(LLM(self._build_agent_prompt(), self.model_name))
    
    def _build_plan_prompt(self) -> str:
        return self.plan_prompt.format(
                            examples = PLAN_SOLVE_plan_EXAMPLE,       
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            question = self.question,              # 问题
                )

    def _build_solve_prompt(self, plan) -> str:
        return self.solve_prompt.format(
                            examples = PLAN_SOLVE_solve_EXAMPLE,        # few shot
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            tool_names = self.tool_names,
                            plan = plan,              # 问题
                            scratchpad = self.scratchpad)          # 过程记录
    
    def _build_replan_prompt(self, plan) -> str:
        return self.replan_prompt.format(
                            examples = PLAN_SOLVE_replan_EXAMPLE,        # few shot
                            table_used_prompt = self.table_used_prompt,
                            tools = self.tools,
                            plan = plan,              # 问题
                            scratchpad = self.scratchpad,
                            question = self.question)          # 过程记录
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return (self.step_n > self.max_steps) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1             # 当前步
        self.finished = False       # 是否完成
        self.scratchpad: str = ''   # 过程

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 model_name: str,
                 tools: str,
                 tool_names: str,
                 table_used_prompt: str,
                 key: str="",
                 max_steps: int = 10,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,  # 重定义！ 
                 reflect_prompt: PromptTemplate = reflect_prompt,            # 重定义！
                 ) -> None:
        
        super().__init__(model_name, question, tools, tool_names, table_used_prompt, agent_prompt=agent_prompt) # 重要！让 react 有 reflexion
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS         # 重定义！
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.model_name = model_name

    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.is_finished() or self.is_halted()):
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            # 仅路径：最后一次尝试 trail
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION: 
            # 反思：所有尝试 trail 的 reflexion
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION: 
            # 路径 + 反思。最后一次尝试的 trail 加上 反思最后一次 trail
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        # 根据 question,scratchpad 反思 reflecion
        return format_step(LLM(self._build_reflection_prompt(), model_name=self.model_name))


    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
 
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                             examples = self.react_examples,
                            question = self.question,
                            scratchpad = self.scratchpad,
                            reflexions = self.reflections, 
                            table_used_prompt = self.table_used_prompt, 
                            tools = self.tools,
                            tool_names = self.tool_names)
   

import json

def parse_plan(rsp: str):
    step = 1
    res = []
    while True:
        if rsp.find(f'第{step}步: ')  == -1:
            return res
        else:
            if len(res) > 0:
                res[-1] = rsp.split(f'第{step}步: ')[0]
            res.append(rsp.split(f'第{step}步: ')[1])
            rsp = rsp.split(f'第{step}步: ')[1]
            step += 1

def parse_action(rsp: str):
    # json_pattern = r'\[.*?\{.*?\}.*?\]'
    json_pattern = r"```json(.*?)```"
    rsp_json = None
    matches = re.findall(json_pattern, rsp, re.DOTALL)
    if len(matches) != 0:
        for match in matches:
            try:
                # 加载外层的 JSON 列表
                # print(match)
                match = match.replace('\'', '\"').replace('(', '（').replace(')', '）')     
                rsp_json = json.loads(match)    
                if isinstance(rsp_json['action_input'], dict):
                    if rsp_json['action'] == 'get_rank' or rsp_json['action'] == 'get_sum' or rsp_json['action'] == 'get_subtraction' or \
                        rsp_json['action'] == 'get_multiplication' or rsp_json['action'] == 'get_division':
                        return rsp_json['action'], rsp_json['action_input']
                    if 'columns' in rsp_json['action_input'].keys():
                        if len(rsp_json['action_input']['columns']) > 5:
                            rsp_json['action_input']['columns'] = []
                    elif 'identifier' in rsp_json['action_input'].keys() or 'prov' in rsp_json['action_input'].keys():
                        rsp_json['action_input']['columns'] = []
                return rsp_json['action'], rsp_json['action_input']
            except Exception as e:
                return "", ""
    else:
        # 尝试匹配 xxx```pythontool_call(identifier='xxx', columns=[])```
        try :
            action, tool_call, tail = rsp.split("```")
            prefix = "pythontool_call("
            suffix = ")"
            if tool_call.startswith(prefix) and tool_call.endswith(suffix):
                tool_call = tool_call[len(prefix):-len(suffix)]
            tool_call_dict = eval(f"dict({tool_call})")  # 使用 eval 将字符串字典化
            return action, tool_call_dict
        except Exception as e:
            return "", ""

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + '反思:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'问题: {question}\n' + scratchpad.strip('\n').strip() + '\n(之前的试验结束)\n'

# def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = gpt2_enc) -> str:
#     lines = scratchpad.split('\n')
#     observations = filter(lambda x: x.startswith('Observation'), lines)
#     observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
#     while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
#         largest_observation = observations_by_tokens.pop(-1)
#         ind = lines.index(largest_observation)
#         lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
#     return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)



