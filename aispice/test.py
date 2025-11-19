
"""
This example demonstrates how to use XGrammar in Huggingface's transformers, integrated with
a minimal LogitsProcessor.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import xgrammar as xgr

device = "cuda"
# device = "cpu"

# 0. Instantiate with any HF model you want
#model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "models/Qwen2.5-1.5B-Instruct"

# model_name = "microsoft/Phi-3.5-mini-instruct"
#model_name = "./Llama-3.2-1B-Instruct"
#model_name = "google/gemma-2b-it"
#model_name = "HuggingFaceTB/SmolLM3-3B"


model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size

# 1. Compile grammar (NOTE: you can substitute this with other grammars like EBNF, JSON Schema)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar: xgr.CompiledGrammar = grammar_compiler.compile_builtin_json_grammar()

"""
[
{
    "utterance": "Topic phrase",
    "Slots":
    {
        "intention": "target: request, report, define, confirm, deny, prefer, etc.",
        "action": "action: provide, contain, update, approve, reject, etc.",
        "relation": "relation type between subject and object: contains, describes, depends_on, must_have, etc.",
        "emotion": "emotional tone: neutral, like, dislike, frustration, satisfaction",
        "subject": "object: person or entity or entities that the action is directed at.",
        "object": "emotion: the emotional tone if expressed (e.g. neutral, positive, negative, frustration, satisfaction)"
    }
}
]
"""
#########################################################################################################################

role = """\nYour task is to analyze the input sentence and extract structured slots in JSON format.

Slots you must always provide:
- intention: the overall communicative goal (e.g. request, report, define_rule, express_preference, complaint, provide_info).
- action: the main verb/action expressed (e.g. ask, influence, provide, contain, report, describe, approve, reject).
- relation: the type of relationship between subject and object (e.g. contextual, procedural, inherent, attitude, contains, must_have, may_contain, not_allowed, allowed, depends_on, describes, causes).
- subject: person or entity that performs or is responsible for the action.
- object: person or entity or entities that the action is directed at.
- emotion: the emotional tone if expressed (e.g. neutral, positive, negative, frustration, satisfaction).

Output strictly in JSON."""

# 2. Prepare inputs
messages_list = []
prompts = [
    #"Introduce yourself in JSON briefly as a student.",
    #"The purpose of this document is to describe the process of conducting Work Product reviews.",
    "I like this product.",
    #"What is the delivery date?",
    "The system is too slow",

    # Uncomment for batch generation
    # "Introduce yourself in JSON as a professor.",
]
for prompt in prompts:
    messages = [
        {"role": "system", "content": "You are an annotation extraction model." + role},
        {"role": "user", "content": prompt},
    ]
    messages_list.append(messages)
texts = [
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for messages in messages_list
]

# For batched requests, either use a model that has a padding token, or specify your own
# model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
# model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)
model_inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
).to(model.device)

# 3. Instantiate logits_processor per each generate, and call generate()
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor],
    #do_sample=False,
)

# 4. Post-process outputs and print out response
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for idx, response in enumerate(responses):
    print(prompts[idx]+"::\n", response, end="\n")
