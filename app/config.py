import os
import pathlib
from dotenv import load_dotenv
from pydantic import BaseModel
from functools import lru_cache


load_dotenv()

class Settings(BaseModel):
    
    gen_model_id: str = os.getenv(
        "GEN_MODEL_ID",
        "microsoft/BioGPT"  
    )
    emb_model_id: str = os.getenv(
        "EMB_MODEL_ID",
        "sentence-transformers/all-MiniLM-L6-v2"  
    )

    
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")
    processed_dir: str = os.getenv("PROCESSED_DIR", "./data/processed")

    
    def ensure_dirs(self):
        pathlib.Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance so it's loaded only once per run.
    """
    s = Settings()
    s.ensure_dirs()
    return s
