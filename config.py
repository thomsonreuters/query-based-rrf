"""
Global configuration for all experiments.
Update BASE_DATA_DIR to point to your data location.
"""

import os

# Update this path to your data directory location
BASE_DATA_DIR = os.environ.get(
    'WRRF_DATA_DIR',
    '/home/sagemaker-user/query-aware-rrf/query-based-rrf/data'
)

# You can also set this via environment variable:
# export WRRF_DATA_DIR="/path/to/your/data"

def get_data_path(*paths):
    """Helper function to construct paths relative to BASE_DATA_DIR."""
    return os.path.join(BASE_DATA_DIR, *paths)