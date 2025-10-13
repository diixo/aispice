
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "models/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)


confluence = """
MAN.3 Project Management: The purpose is to identify and control the activities,
and establish resources necessary for a project to develop a product, in the context
of the project's requirements and constraints.

MAN.3.BP1: Define the scope of work.
Identify the project's goals, motivation, and boundaries.
"""

question = "What is the purpose of the Project Management process (MAN.3)?"

###########################################################################
prompt = f"""
You are an ASPICE auditor.
Answer **strictly based** on the document below. Do not invent.

Document:
\"\"\"{confluence}\"\"\"

Question: {question}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

