import pandas as pd


def evaluate_rules(df, rules):
    """
    rules: list of pandas-eval-compatible boolean expressions referencing
    column names, e.g. "age > 0", "price >= 0 and price <= 100000".

    Returns: {rule: {"violations": int, "index": [row indices]}}
             or {rule: {"error": str}} if the rule fails to evaluate.
    """
    results = {}
    for rule in rules:
        rule = rule.strip()
        if not rule:
            continue
        try:
            mask = df.eval(rule, engine="python")
            if mask.dtype != bool:
                raise ValueError("Rule must evaluate to a True/False condition per row.")
            violating = df.index[~mask].tolist()
            results[rule] = {"violations": len(violating), "index": violating}
        except Exception as e:
            results[rule] = {"error": str(e)}
    return results