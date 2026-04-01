from huggingface_hub import HfApi

api = HfApi()

print("Uploading to Hugging Face Spaces...")
try:
    api.upload_folder(
        folder_path='.',
        repo_id='zeenu002/asd-detection-api',
        repo_type='space',
        token='<YOUR_HF_TOKEN_HERE>',
        ignore_patterns=['saved_models/*', '__pycache__/*', '.git/*', '*.pyc', 'hf_upload.py', '.venv/*']
    )
    print("PUSH SUCCESSFUL! 🚀")
except Exception as e:
    print(f"Error during upload_folder: {e}")
