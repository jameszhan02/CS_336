import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

MODEL_NAME = "allenai/OLMo-2-0425-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to("cuda")

PROMPT_FILES = {
    "question_only": "prompts/question_only.prompt",
    "r1_zero":       "prompts/r1_zero.prompt",
    "three_shot":    "prompts/r1_zero_three_shot_gsm8k.prompt",
}

REWARD_FNS = {
    "question_only": question_only_reward_fn,
    "r1_zero":       r1_zero_reward_fn,
    "three_shot":    r1_zero_reward_fn, 
}

templates = {}
for name, path in PROMPT_FILES.items():
    if path is None:
        templates[name] = None
        continue
    with open(path) as f:
        templates[name] = f.read()

examples = []
with open("../data/gsm8k/test.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        examples.append(json.loads(line))

def fill_prompt(template, question):
    if template is None:
        return question
    return template.format(question=question)
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


for i, example in enumerate(examples):
    question = example["question"]
    answer   = example["answer"]
    ground_truth = answer.split("####")[-1].strip()
    
    print(f"\n{'#'*60}")
    print(f"Question # {i+1}/10")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    prompts = {
        name: fill_prompt(template, question)
        for name, template in templates.items()
    }
    
    for name, prompt in prompts.items():
        response = generate(prompt)
        result = REWARD_FNS[name](response, ground_truth)  

        print(f"\n{'─'*40}")
        print(f"[{name}]")
        print(f"RESPONSE:\n{response}")
        print(f"format={result['format_reward']} | answer={result['answer_reward']} | reward={result['reward']}")