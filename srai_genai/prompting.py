"""Prompt engineering utilities."""
from __future__ import annotations

def build_prompt(role,task,context=None,constraints=None,output_format=None,examples=None):
    sections=[f"Role: {role}",f"Task: {task}"]
    if context: sections.append(f"Context:\n{context}")
    if constraints:
        sections.append("Constraints:\n"+"\n".join(f"- {c}" for c in constraints))
    if examples:
        sections.append("Examples:\n"+str(examples))
    if output_format:
        sections.append(f"Output format:\n{output_format}")
    return "\n\n".join(sections)

def chain_of_verification_prompt(question,draft_answer):
    return f"""Question:
{question}

Draft answer:
{draft_answer}

Verify the draft by:
1. Listing claims that require checking.
2. Checking internal consistency.
3. Identifying unsupported assumptions.
4. Producing a corrected final answer."""

def few_shot_prompt(instruction,examples,new_input):
    formatted=[instruction]
    for i,(x,y) in enumerate(examples,1):
        formatted.append(f"Example {i}\nInput: {x}\nOutput: {y}")
    formatted.append(f"New input: {new_input}\nOutput:")
    return "\n\n".join(formatted)

def prompt_quality_check(prompt):
    text=str(prompt)
    checks={
        "has_role":"Role:" in text,
        "has_task":"Task:" in text,
        "has_constraints":"Constraints:" in text,
        "has_output_format":"Output format:" in text,
        "specific_length":len(text)>80,
    }
    checks["score"]=sum(checks.values())
    return checks
