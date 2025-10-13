
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "models/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)


confluence = """
1.2 Project Purpose
Name of the project: "Amber"

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


1.3 Scope
The scope is defined by Product Management Team as per MAN.3 Team structure and Org chart  and continuously elaborated by Stream Teams. The initial set of the Features that was discussed and approved in Jira .

The description of the Features are defined in Jira.

High level roadmap is defined Android Platform Roadmap and tracked through MAN.3 Release Plan and Feature roll out plan .

The features to be delivered should be defined by the Product Management Team and can be changed in the process of project implementation.

The project includes Feature/Requirements delivery under ASPICE processes. So, the project activities include processes definition and establishment.

ASPICE scope that is managed by the project:
Management processes:
MAN.3 - Project Management
MAN.5 - Risk Management
Supporting processes such as
SUP.1 - Quality Assurance
SUP.8 - Configuration Management
SUP.9.1 - Problem Resolution Management
SUP.9.2 - Defect Management
SUP.10 - Change Request Management
System processes (ASPICE-base level)
SYS.1 - Requirements Elicitation
SYS.2 - System Requirements Analysis
SYS.3 - System Architectural Design
SYS.4.2 - System Integration and Integration Verification
SYS.5 - System Verification
Engineering processes:
SWE.1 Software Requirements Analysis Strategy

SWE.2 - Software Architectural Design
SWE.3 - Software Detailed Design and Unit Construction
SWE.4 - Software Unit Verification
SWE.5.2 - Software Component Verification and Integration Verification
SWE.6 - Software Verification
Other processes in the scope:
SPL.2 - Product Release
ACQ.4 - Supplier Monitoring
Functional Safety
SEC - CyberSecurity
PIM.3 - Process Improvement
Boundaries:

The main functionality which should be delivered in 1.2ProjectPurpose

Out of scope:

AndroidAuto and Carplay certifications
Instrument Cluster and HUD are only part of the demo showcase, not part of the product
Only Qualcomm 8295 adaptation is in scope. Further SoCs are out of scope for 1.0 delivery
Mixed criticality enhancements (safety and real-time related requirements) in terms of platform development are out of focus for 1.0 delivery
HERE based Navigation solutions serves as demo only
ACCESS AppStore solutions serves as demo only
Hardware, Mechanical and Electrical types of work are out of scope of the project
The ASPICE CL2 applies only for the following components:
"""

###########################################################################

prompt = f'''
Task: Determine whether the text describe project definition according our criterias, extract the project's name if it is mentioned.

Subject: "Project"
Object of verification: The text provided below.
Criteria: The text must explicitly or implicitly describe the project definition. A project definition should describe the project's scope, goals, motivation, boundaries, resources, or constraints.

If the text defines what a project is — answer "YES" and provide the supporting phrase.
If the text only mentions "project" without defining it — answer "NO".

Return your answer in JSON:

{{
  "subject": "Project",
  "project_name": "string or null",
  "has_definition": yes/no,
  "evidence": "short quote or phrase that defines the project"
  "purpose": "string or null",
  "goals": "string or null",
}}

Text:
\"\"\"{confluence}\"\"\"
'''

prompt_BP_1 = f'''
Task: Determine whether the text defines the concept of a "project" according our criterias, extract the project's name if it is mentioned.

Subject: "Project"
Object of verification: The text provided below.

Criteria:
1. The text must describe what a project is — its purpose, scope, goals, boundaries, resources, or constraints.
2. If a specific project name, title, or identifier appears, extract it.
3. If there is explicit project name, return name for project_name or null.

Return your answer in JSON:

{{
  "subject": "Project",
  "has_definition": true/false,
  "project_name": "string or null",
  "evidence": "short quote or phrase that defines the project"
}}

Text:
\"\"\"{confluence}\"\"\"
'''

prompt_PMBoK = f'''
Task: Determine whether the text defines the concept of a "project" according reference definition. Extract the project's name if it is mentioned.

Reference definition:
"A project is a temporary endeavor undertaken to create a unique product, service, or result."

Additional details:
- A project produces a unique deliverable: a product, service, or result.
- Deliverables are unique and verifiable outcomes or capabilities.
- Repetitive elements may exist, but each project remains unique in key characteristics (such as objectives, scope, location, team, or context).

Your task:
1. Analyze the provided text.
2. Determine whether it includes a definition or description that conceptually matches the Reference definition of a "project".
3. If yes, identify which elements are present (temporariness, uniqueness, deliverable, objective).
4. Extract the project name if it is explicitly mentioned.

Return your result in JSON format:

{{
  "subject": "Project",
  "project_name": "string or null",
  "has_definition": true/false,
  "matched_elements": ["temporariness", "uniqueness", "deliverable", "objective"],
  "evidence": "short quote(s) from the text",
  "comments": "brief reasoning"
}}

Text for analysis:
\"\"\"{confluence}\"\"\"
'''

inputs = tokenizer(prompt_BP_1, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

