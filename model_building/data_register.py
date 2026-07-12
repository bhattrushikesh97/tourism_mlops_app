
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

token = os.getenv("HF_TOKEN")

repo_id = "bhattrushikesh97/tourism-mlops-app"    # please create your space and repository
repo_type = "dataset"

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo  '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Repo '{repo_id}' created.")

api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)
