import torch
import gc

class MemoryManager:
    @staticmethod
    def clear_memory():
        """Clear GPU memory cache and run garbage collection"""
        torch.cuda.empty_cache()
        gc.collect()