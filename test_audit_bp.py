
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

6. Technical Process
6.1 Project Life Cycle
Introduction to the Agile V-Model
An iterative agile framework with a V-model will be used as the software development life cycle. Agile : SAFe

This means that:

The smallest iteration, for which we plan, implement, review and adjust is called a Sprint - which is 2 weeks long. A Product Increment is consists of 6 Sprints (12 weeks long) and ends up with Major Release
In each of these Sprints, multiple activities are carried out in parallel (wherever applicable) to build the end product. These include:
a. Software Requirements Analysis

b. Software Architecture Design

c. Software Detailed Design and Unit Construction

d. Software Unit Testing

e. Software Integration and Integration Testing

f. Software Qualification Test



Sprintly Activities
(warning) Note: For the common definition, documentation may include one of the definitions of sprint: sprint, scrum and/or iteration. All terms to be treated equaly in definition.

In order to plan, monitor, review and refine the team's activities, there are regular meetings/ceremonies scheduled in each Sprint:

## The sprint lifecycle:
Scrum and sprints follow a similar life cycle, where a series of increments are released to customers.

1. Sprint planning: is the first phase (event) of the sprint life cycle.
During sprint planning, the product owner will go through the entire product backlog with the scrum team as per their priorities. The scrum team estimates the user stories or tasks. Based on the sprint goals defined, the user stories are picked for implementation in the current sprint.

2. Sprint execution: is the lengthiest phase of the sprint life cycle.
The development team will implement every user story and task planned for the current sprint. All the activities, design, development, and testing are to be completed as per the definition of done during this phase.

3. Sprint review: is the third event in the sprint life cycle.
This is where the scrum team showcases the shippable increment to the product owner and other stakeholders.

4. Sprint retrospective: is the last phase.
Sometimes, scrum teams complete this event before the sprint review to help them present any concerns and notes in the sprint review itself. This is considered the last phase of the sprint, after which the next sprint would start all over again from sprint planning. During this phase, the team answers the following questions:

* What went well in the sprint?
* What could be improved?
* What will we commit to improving in the next sprint?

5. Product Backlog Refinement: is an ongoing phase within the Scrum life cycle.
This is where the Scrum Team collaborates to add detail, estimates, and order to Product Backlog items (PBIs) to prepare them for future sprints. 
"""


#https://blog.logrocket.com/product-management/guide-to-the-five-types-of-scrum-meetings/

#https://blog.logrocket.com/product-management/what-is-a-sprint-agile-scrum-teams/

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

prompt_BP_2_strong = f'''
Task: evaluate whether the provided text defines or describes a 
"Project Life Cycle" consistent with MAN.3.BP2 and the organization's defined Agile software development lifecycle.

---

Reference 1: ASPICE MAN.3.BP2
"Define the life cycle for the project, which is appropriate to the scope, context, and complexity of the project.
Define a release scope for relevant milestones. This may include the alignment of the project life cycle with the customer's development process."

Reference 2: Organization's Agile Software Development Lifecycle

The Agile software development lifecycle covers the following stages:

1. **Analysis and Requirement Definition** — Future product owners create a list of main requirements. The development team analyzes relevant aspects to figure out the core features and remove unnecessary functionality.
2. **Design** — The team designs the interface and architecture based on key requirements, selects technologies, and produces UI mockups.
3. **Development** — The solution is implemented according to the agreed requirements. This phase is typically the longest.
4. **Testing** — QA staff conduct comprehensive testing (functionality, integration, acceptance, etc.) and provide reports.
5. **Deployment** — After all bugs are fixed, the software is deployed (possibly as a beta version) for user testing.
6. **Feedback** — Support staff collect user feedback to identify improvements and ensure satisfaction.


Follow:
1. Analyze the provided text.
2. Determine whether it defines or describes the **project life cycle** — i.e., a structured sequence of phases, stages, or activities from project initiation to closure.
3. Check if the definition or description is **appropriate to the project's scope, context, and complexity**.
4. Verify if the text mentions or implies a **release scope** or **milestones**.
5. Check for any indication of **alignment with the customer's development process**.

Evaluation criteria:
1. Does the text describe a structured set of project phases or iterations?
2. Does it indicate a start-to-end structure (initiation → execution → release or closure)?
3. Is the described life cycle appropriate to the project's context or methodology (e.g., Agile SAFe)?
4. Does it include or imply releases, milestones, or iterations?
5. Is there alignment with customer or organizational process?

Return the result as structured JSON:

{{
  "subject": "Project Life Cycle",
  "has_definition": true/false,
  "mentions_release_scope": true/false,
  "mentions_milestones": true/false,
  "mentions_alignment_with_customer_process": true/false,
  "appropriateness_to_context": "high|medium|low|unknown",
  "evidence": "exact quote(s) from the text that justify the assessment",
  "comments": "brief reasoning for the evaluation"
}}

Text for analysis:
\"\"\"{confluence}\"\"\"
'''


inputs = tokenizer(prompt_BP_2_strong, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

