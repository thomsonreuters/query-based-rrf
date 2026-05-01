from pathlib import Path
from dotenv import load_dotenv

# Change this to match your environment's file
# e.g. ".env.sagemaker", ".env.cluster", ".env.local"
ENV_FILE = ".env.local"


def load_env():
    """Walk up from utils/ to find and load ENV_FILE.

    override=False means env vars already set in the shell or SageMaker
    environment settings take priority over values in the file.
    """
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ENV_FILE
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
