import subprocess
import os
import warnings
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModelDownload:
    def __init__(self, environments: Optional[Dict] = None, model_name: Optional[str] = None):
        if environments is not None:
            warnings.warn(
                "You are setting environment variables from the provided dictionary. These environment variables will override any existing ones and change the download paths for the models."
            )
            self.environments = environments
            self.model_name = model_name

    def download_model_alternate_path(self):
        """
            Downloads the specified model using an alternate path if the default HuggingFace installation path is not suitable or if HuggingFace is unreachable .
        """
        # Set environment variables from the provided dictionary
        os.environ['HF_ENDPOINT'] = self.environments.get('hf_endpoint', '')
        os.environ['HF_DATASETS_CACHE'] = self.environments.get('hf_datasets_cache', '')
        os.environ['HF_HOME'] = self.environments.get('hf_home', '')
        os.environ['HUGGINGFACE_HUB_CACHE'] = self.environments.get('hf_hub_cache', '')
        os.environ['TRANSFORMERS_CACHE'] = self.environments.get('transformers_cache', '')
        TRANSFORMERS_CACHE = os.environ.get('TRANSFORMERS_CACHE', '')
        # Construct the command to download the model
        command = f"huggingface-cli download --resume-download --local-dir-use-symlinks False {self.model_name} --local-dir {TRANSFORMERS_CACHE}"
        try:
            # Execute the download command
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error:", e)

    def hf_download_and_load_model(self):
        """
            Downloads the specified model using the default HuggingFace installation path.
        """
        # Load model directly
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

