import json


def export_pipeline(rules, near_dup_threshold):
    """Serialize the current validation rules + near-dup threshold to JSON."""
    payload = {
        "version": 1,
        "validation_rules": rules,
        "near_dup_threshold": near_dup_threshold,
    }
    return json.dumps(payload, indent=2)


def load_pipeline(json_str):
    """Parse a previously exported pipeline JSON back into (rules, threshold)."""
    payload = json.loads(json_str)
    return payload.get("validation_rules", []), payload.get("near_dup_threshold", 0.9)