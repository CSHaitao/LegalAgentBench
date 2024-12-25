# 🚀 LegalAgentBench
[![GitHub Sponsors](https://img.shields.io/badge/sponsors-GitHub-blue?logo=github&logoColor=white)](https://github.com/sponsors) ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-yellow) ![Contributors](https://img.shields.io/badge/contributors-10-yellow) ![Awesome List](https://img.shields.io/badge/awesome-awesome-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)

# 🌟 About This Repo

This repo is for our paper: 

📝[LegalAgentBench: Evaluating LLM Agents in Legal Domain.](https://arxiv.org/abs/2412.17259)

With the increasing intelligence and autonomy of LLM agents, their potential applications in the legal domain are becoming increasingly apparent. However, existing general-domain benchmarks cannot fully capture the complexity and subtle nuances of real-world judicial cognition and decision-making. 

Therefore, we propose **LegalAgentBench**, a comprehensive benchmark specifically designed to evaluate LLM Agents in the Chinese legal domain. 

LegalAgentBench includes 17 corpora from real-world legal scenarios and provides 37 tools for interacting with external knowledge. We designed a scalable task construction framework and carefully annotated 300 tasks. These tasks span various types, including multi-hop reasoning and writing, and range across different difficulty levels, effectively reflecting the complexity of real-world legal scenarios. 

# ⚡️ Update

🔥🔥 News: 2024/12/23: We released the first version of the [full paper](https://arxiv.org/abs/2412.17259).

## 🧩 Characteristic

- **Focus on Authentic Legal Scenarios:** LegalAgentBench is the first dataset to evaluate LLM agents in legal scenarios. It requires LLMs to demonstrate a solid understanding of legal principles, enabling them to appropriately select and utilize tools to solve complex legal problems. This represents a significant step forward in advancing the application of LLM agents in legal scenarios.

- **Diverse Task Types and Difficulty Levels:** LegalAgentBench adopts a scalable task construction framework aimed at comprehensively covering various task types and difficulty levels. Specifically, we construct a planning tree based on the dependencies between the corpus and tools, and select tasks through hierarchical sampling and a maximum coverage strategy. Finally, we constructed 300 distinct tasks, including multi-hop reasoning and writing tasks, to comprehensively evaluate the LLM’s capabilities。

- **Fine-Grained Evaluation Metrics:** Rather than relying solely on final success rates as evaluation criteria, LegalAgentBench introduces the process rate through the annotation of intermediate steps, enabling fine-grained evaluation. This approach provides deeper insights into an agent’s capabilities and identifies areas for improvement beyond the final result.

## 🌳 Repo Structure
```
LegalAgentBench/
│
├── data/               
|   |── dataset.json            # Question and Answer Set
├── src/
|   |── evaluation/             # evlaution example
|   |── output/                 # output example
|   |── token/                  # token consumption records
|   ├── generated_tools.py      # tools can be used for LLM Agents
|   ├── globals.py              # global variables
|   ├── plan_and_excute.py      # code for plan_and_excute method
|   ├── plan_and_solve.py       # code for plan_and_solve method
|   ├── react.py                # code for react method              
|   ├── schema.py               # definition for corpus
|   ├── prompt.py 
|   └── utils.py                
├── agents.py                   # definition for agents
├── fewshots.py                 # fewshots for agents
├── prompts.py                  # prompts for agents
```

## ⚙️ Quick Start
```python
git clone https://github.com/CSHaitao/LegalAgentBench.git
cd LegalAgentBench
pip install -r requirements.txt

cd src
python react.py --model LLM_name --date time
```

❗️ **Important**: Replace the string `your_api_key` in `utils.py` with the actual key.

## 📖 Cite Our Work

If you find our work useful, please do not save your star and cite our work:
```
@misc{li2024legalagentbenchevaluatingllmagents,
      title={LegalAgentBench: Evaluating LLM Agents in Legal Domain}, 
      author={Haitao Li and Junjie Chen and Jingli Yang and Qingyao Ai and Wei Jia and Youfeng Liu and Kai Lin and Yueyue Wu and Guozhi Yuan and Yiran Hu and Wuyue Wang and Yiqun Liu and Minlie Huang},
      year={2024},
      eprint={2412.17259},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17259}, 
}
```
