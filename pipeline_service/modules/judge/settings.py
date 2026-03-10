from pydantic import BaseModel


class JudgeConfig(BaseModel):
    """VLLM Judge configuration"""
    enabled: bool = False
    vllm_port: int = 8095
    vllm_model_name: str = "zai-org/GLM-4.1V-9B-Thinking"
    vllm_revision: str = "17193d2147da3acd0da358eb251ef862b47e7545"
    vllm_api_key: str = "local"
    vllm_url: str = "http://localhost:8095/v1"
    gpu_memory_utilization: float = 0.20
    max_model_len: int = 8096
    max_num_seqs: int = 2
