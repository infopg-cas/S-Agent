import os
import torch
from huggingface_hub import snapshot_download
from FlagEmbedding import FlagModel
from .base import Base, get_home_cache_dir


class DefaultEmbedding(Base):
    def __init__(self, model_name, query_instruction_for_retrieval, repo_id, *args, **kwargs):
        try:
            flag_model = FlagModel(
                os.path.join(get_home_cache_dir(), model_name),
                query_instruction_for_retrieval=query_instruction_for_retrieval,
                use_fp16=torch.cuda.is_available()
            )
        except Exception as e:
            model_dir = snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(get_home_cache_dir(), model_name),
                local_dir_use_symlinks=False
            )
            flag_model = FlagModel(
                model_dir,
                query_instruction_for_retrieval=query_instruction_for_retrieval,
                use_fp16=torch.cuda.is_available()
            )
        self.model = flag_model

    def encode(self, texts: list, batch_size=32):
        res = []
        for i in range(0, len(texts), batch_size):
            res.extend(self.model.encode(texts[i:i + batch_size]).tolist())
        return torch.tensor(res)

    def encode_queries(self, text: str):
        return torch.tensor(self.model.encode_queries([text]).tolist()[0])

