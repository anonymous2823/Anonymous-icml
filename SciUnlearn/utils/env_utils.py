import os
from typing import List


def validate_required_env_vars(required: List[str]) -> List[str]:
    """
    Return a list of missing environment variable names.
    """
    return [key for key in required if not os.getenv(key)]