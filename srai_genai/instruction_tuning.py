"""Instruction-tuning utilities."""
from __future__ import annotations

def format_instruction_example(instruction,input_text,output_text):
    return {
        "prompt":f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n",
        "response":output_text,
    }

def build_instruction_dataset(rows):
    return [
        format_instruction_example(
            row["instruction"],row.get("input",""),row["output"]
        )
        for row in rows
    ]

def response_only_mask(prompt_tokens,response_tokens):
    return [0]*len(prompt_tokens)+[1]*len(response_tokens)

def instruction_quality_check(example):
    checks={
        "has_instruction":bool(example.get("instruction")),
        "has_output":bool(example.get("output")),
        "input_is_string":isinstance(example.get("input",""),str),
        "output_is_string":isinstance(example.get("output"),str),
    }
    checks["valid"]=all(checks.values())
    return checks

def curriculum_sort(examples):
    return sorted(examples,key=lambda x:len(str(x.get("instruction","")))+len(str(x.get("input",""))))
