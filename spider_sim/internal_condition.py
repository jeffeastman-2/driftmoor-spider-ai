"""Private internal state adapter; uses Driftmoor's numeric condition model."""
from models.condition import DIMENSIONS


def own_condition(condition):
    """Detached snapshot; baseline-zero dimensions are structurally absent."""
    present=[dim for dim in DIMENSIONS
             if dim not in condition.profiles or condition.profiles[dim].baseline!=0]
    return {"values":condition.state.to_dict(),
            "deltas":{dim:condition.delta(dim) for dim in present},
            "present":present}
