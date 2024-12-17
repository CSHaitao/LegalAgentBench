TABLE_PROMPT = """
你是一个高级的判断器代理. 
你的任务是根据数据表的子段属性属性, 精准识别并筛选出能够解答给定问题所需的所有相关数据表. 
务必确保任何可能包含问题中提及字段的数据表均考虑在内. 将这些与问题相关的数据表完整输出. 

问题:
{question}

数据表:
{database_schema}

以下是几个参考:

问题:原告是安利股份(公司简称)的案件审理法院是哪家法院
{{
     "名称": ['LegalDoc','CourtInfo','CompanyInfo']
}}

问题:请问一下, 91320115773957541H限制高消费的总额是多少元?
{{
     "名称": ['CompanyRegister', 'RestrictionCase']
}}

问题: 原告是300077案件审理法院是什么时候成立的
{{
    "名称": ['CompanyInfo', 'LegalDoc', 'CourtInfo']
}}

问题:统一社会信用代码为91341282674208131X的公司作为原告时, 被告所委托的律师事务所的联系方式是什么?
{{
    "名称": ['CompanyRegister', 'LegalDoc', 'LawfirmInfo']
}}

问题:案号为(2020)浙0483民初2668号的法院所在的区县是哪个?
{{
    "名称": ['CourtCode', 'CourtInfo', 'AddrInfo']
}}

参考示例结束

判断以下问题与哪些数据表格相关:
{question}

务必确保任何可能包含问题中提及字段的数据表均考虑在内. 将这些与问题相关的数据表完整输出. 
请按照以下json格式进行输出, 可以被Python json.loads函数解析. 不回答问题, 不作任何解释, 不输出其他任何信息. 
```json
{{
    "名称": 
}}
``` 
"""

FILTER_PROMPT = """问题: {query}
信息: {info}
请结合提供的信息, 直接给出简洁, 完整且清晰的回答. 
回答格式忠于提问方式. 不要回答问题之外的内容. 
你最终的回答应当以自然语言的形式组织, 要自然, 流畅和精确. 
如果问题中明确指明了金额的单位, 如"元", 请将结果中的单位转化为题目中要求的单位(例如将"万元"转化为"元"). 
回答:
"""

SUMMARY_PROMPT = """问题: {query}
信息: {info}
你是一个擅长整理答案的高级代理, 结合提供的信息, 你的任务是回答原始问题. 
要求务必要全面精确地保留信息的细节, 不遗漏任何关键细节及有意义的表述, 保留信息中所有的有效实体, 包括但不限于律师事务所名称, 法院全称, 注册资本, 成立日期, 公司名称, 案号, 涉案金额等细节. 
输出忠于原始信息, 不要添加任何额外内容.  
你最终的回答应当以自然语言的形式组织, 要自然, 流畅和精确. 
如果问题中明确指明了金额的单位, 如"元", 请将结果中的单位转化为题目中要求的单位(例如将"万元"转化为"元"). 
回答:
"""

Self_Judge = """
你是一个可以判断答案是否能回答问题的高级代理。
在之前测试中你通过推理过程查询API获得了最终的答案。
请判断该答案是否能回答问题。如果答案不能完整的回答问题，则回答False。

问题：
{question}
答案：
{answer}

以下是几个例子：
问题：海天精工注册在哪个市的哪个区县
答案：海天精工注册在浙江省宁波市北仑区。
{{
    "Answer": "True"
}}

问题：(2020)新2122民初1105号案件中，审理当天审理法院与原告的律师事务所所在城市的最低温度相差多少度？本题使用的API个数为？最小调用次数为多少次？
答案：案号为(2020)新2122民初1105号的案件中，审理当天(2020年11月06日)浙江百方律师事务所所在城市浙江省绍兴市的最低气温为3度。由于无法获取到鄯善县人民法院所在地的气温信息，无法计算审理法院与原告律师事务所所在城市的最低气温差。
{{
    "Answer": "False"
}}

问题：案号为(2023)津0116执29434号的限制高消费执行案件的执行法院是哪家？被限制高消费公司的法定代表人是谁？提出限制高消费申请的当事人是谁？案件涉及的总金额是多少？该执行法院的级别是什么？该法院的负责人是谁？该法院的行政级别、法院级别和具体级别分别是什么？
答案：根据查询结果，案号为(2023)津0116执29434号的限制高消费执行案件的执行法院是天津市滨海新区人民法院。被限制高消费公司的法定代表人是黄春奇。提出限制高消费申请的当事人是天津市凯成房地产咨询有限公司。案件涉及的总金额是2054659.34元。该执行法院的级别是基层法院，其行政级别为市级，具体级别为1。关于该法院的负责人，由于提供的信息中没有相关数据，无法给出答案。
{{
    "Answer": "False"
}}

如果答案可以完整回答问题，则返回True，如果不能完整回答，则回答False/
请按照以下json格式进行输出，可以被Python json.loads函数解析。不回答问题，不作任何解释，不输出其他任何信息
```json
{{
    "Answer": ""
}}
```
"""