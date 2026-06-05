from datasets import load_dataset

tgt_string = """
How many more purple flowers are there than yellow flowers? ** There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.
How many purple flowers are there? ** So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.
How many flowers are there in total? ** Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.
How many green flowers are there? ** That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.
How many plants does Mark have in his garden? ** So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.
#### 35
"""


# (1) Render - <think> and <answer> tags
# (1.1) Find '<<' and '>>' pairs: Replace '<<' with '<think>' and '>>' with '</think>
# (1.2) Find '####' and replace with '<answer>'
tgt_string = tgt_string.replace("<<", "<think>").replace(">>", "</think>")
tgt_string = tgt_string.replace("####", "<answer>")
print(tgt_string)



def render_thought_process(example):
    question = example["question"]
    answer = example["answer"]
    answer = answer.replace("####", "<answer>")
    answer = answer.replace("<<", "<think>").replace(">>", "</think>")
    return {"prompt": question, "completion": answer}
    

raw_ds = load_dataset("openai/gsm8k", "main")
raw_ds = raw_ds.map(render_thought_process).remove_columns(["question", "answer"])
print(raw_ds["train"][100])

raw_ds.save_to_disk("artifacts/datasets/base/gsm8k_rendered")