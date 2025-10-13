
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "models/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)


confluence = """
1.2 Project Purpose
Name of the project: Amber  

The main goal of the project is timely feature delivery according to requirements in compliance with ASPICE processes.

The objective: Develop continuously updated, pre-integrated Android platform featuring selected third-party IPs and essential automotive functionalities, all delivered within a fully transparent software factory.

The project includes the development of following components/features:

IVI:
Ready to use Infotainment solution which can be easily adopted to new SoC and OEMs HMI.

Android Compliance System in Hypervised and Virtualized environment.

Developing interfaces for components for clear HW / SW separation.

IVP:
Provide an “automotive software platform as a product” as a complete package to the market.

Portable and configurable across different HW / SOCs

Showcase on Qualcomm SOC and in Virtual Environment

SWF:
Software Factory for Platform Product Development

Workbench product to support Luxoft and customer SWE + SYS testing by providing virtual and remote targets
"""

###########################################################################

prompt_BP_1 = f'''
Task: Determine whether the text defines the concept of a "project"
according to ASPICE Base Practice MAN.3, and extract the project's name if it is mentioned.

Subject: "Project"
Object of verification: The text provided below.

Criteria:
1. The text must describe what a project is — its purpose, scope, goals, boundaries, resources, or constraints.
2. If a specific project name, title, or identifier appears, extract it.
3. If there is no explicit name, return null for project_name.

Return your answer in JSON:

{{
  "subject": "Project",
  "has_definition": true/false,
  "project_name": "string or null",
  "evidence": "quote or short phrase supporting the conclusion"
}}

Text:
\"\"\"{confluence}\"\"\"
'''

inputs = tokenizer(prompt_BP_1, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

