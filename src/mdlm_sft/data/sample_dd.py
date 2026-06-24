from jinja2 import Template
from typing import Dict, Callable, Any
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import gc 
import numpy as np

from .utils import create_nested_stratified_splits_hf, _ratio_to_split_name

datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024 ** 3  # 32GB

def normalize_whitespace_lst(batch, field_name=None):
    assert field_name is not None, "field_name must be provided"
    batch[field_name] = [
        [" ".join(text.split()) for text in utterances]
        for utterances in batch[field_name]
    ]
    return batch    

def count_tokens_lst_batched(batch, tokenizer=None):
    batch["num_tokens"] = [
        sum(len(ids) for ids in tokenizer(utterances)["input_ids"])
        for utterances in batch["utterances"]
    ]
    return batch    


def split_sentences_to_prompt_completion(batch, join=False):
    """
    Splits each item's utterances in half into prompt and completion.
    Args:
        join:   If False, prompt/completion are returned as list[str] (default)
                If True,  prompt/completion are joined into a single str
    """
    prompts, completions = [], []
    for sentences in batch["utterances"]:
        mid_idx = round(len(sentences) / 2)
        if join:
            prompts.append(" ".join(sentences[:mid_idx]))
            completions.append(" ".join(sentences[mid_idx:]))
        else:
            prompts.append(sentences[:mid_idx])
            completions.append(sentences[mid_idx:])
    return {"prompt_raw": prompts, "completion_raw": completions}    


def render_fn(batch):
    prompt_templ = Template("""Continue the following story/dialogue:
    {%- for item in prompt %}
    {{ item }}
    {%- endfor %}""")

    response_templ = Template("""The story/dialogue could continue like this:
    {%- for item in completion %}
    {{ item }}
    {%- endfor %}""")

    rendered_prompts = [prompt_templ.render(prompt=p) for p in batch["prompt_raw"]]
    rendered_compls  = [response_templ.render(completion=c) for c in batch["completion_raw"]]

    return {"prompt": rendered_prompts, "completion": rendered_compls}


def load_and_process_dailydialog(BATCH_SIZE: int = 8000, N_BINS: int = 2, OUTPUT_DIR=None) -> DatasetDict:
    try:
        dd = load_dataset("avgJo3/dailydialog-datasets-4.8.5")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        dd = (
            dd
            .map(normalize_whitespace_lst, fn_kwargs={"field_name": "utterances"}, batched=True, batch_size=BATCH_SIZE)
            .map(count_tokens_lst_batched, fn_kwargs={"tokenizer": tokenizer}, batched=True, batch_size=BATCH_SIZE)
            .map(split_sentences_to_prompt_completion, fn_kwargs={"join": False}, batched=True, batch_size=BATCH_SIZE)
            .map(render_fn, batched=True, batch_size=BATCH_SIZE)
        )
        del tokenizer
        gc.collect()

        total_toks_val = sum(dd['validation']["num_tokens"])
        print(total_toks_val)

        total_toks_test = sum(dd['test']["num_tokens"])
        print(total_toks_test)

        total_toks_train_prefilter = sum(dd['train']["num_tokens"])
        print(total_toks_train_prefilter)

        filtered_ds = dd["train"].filter(lambda x: x["num_tokens"] <= 1024, desc="Filtering examples with total token count > 1024")
        print(len(filtered_ds))

        total_toks_train_postfilter = sum(filtered_ds["num_tokens"])
        print(total_toks_train_postfilter)

        token_counts = np.array(filtered_ds["num_tokens"])
        bin_edges    = np.linspace(token_counts.min(), token_counts.max(), N_BINS + 1)
        print(bin_edges)

        ds = filtered_ds.map(
            lambda batch: {"bin": [f"bin_{i}" for i in np.searchsorted(bin_edges[1:-1], batch["num_tokens"], side="left")]},
            batched=True,
            desc="Assigning bins based on token count",
        )
        del filtered_ds, token_counts, bin_edges
        gc.collect()

        result = create_nested_stratified_splits_hf(
            dataset=ds,
            task_column="bin",
            eval_ratio=0.1,
            split_ratios=[0.125, 0.25, 0.5, 1.0],
            random_seed=42,
            token_column=None,
            max_tokens=None,
        )

        dataset_dict: Dict[str, Dataset] = {
            "strat_eval": result["strat_eval"].remove_columns("bin"),
            "validation": dd["validation"],
            "test":       dd["test"],
        }
        for ratio, split_info in sorted(result["splits"].items()):
            dataset_dict[_ratio_to_split_name(ratio)] = split_info["train"].remove_columns("bin")
        out_dd = DatasetDict(dataset_dict)
        
        del ds, result, dataset_dict, dd
        gc.collect()

        print(out_dd)

        if OUTPUT_DIR is not None:
            out_dd.save_to_disk(OUTPUT_DIR)
            del out_dd
            gc.collect()
            out_dd = DatasetDict.load_from_disk(OUTPUT_DIR)  # mmap'd, low RSS

        return out_dd

    finally:
        gc.collect()       



def upload_dailydialog_to_hf():
    dd = load_and_process_dailydialog()
    dd.push_to_hub("avgJo3/dailydialog-strat", private=False)
    del dd 

if __name__ == "__main__":
    upload_dailydialog_to_hf()