import logging
import time
import torch
from sentence_transformers import CrossEncoder as BaseCrossEncoder
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

class LoggedCrossEncoder:
    def __init__(self, model_name, device=None, **kwargs):
        self.model = BaseCrossEncoder(model_name, device=device, **kwargs)
        
        # Explicitly move model to CUDA if available and device is set to cuda
        if device == 'cuda' and torch.cuda.is_available():
            self.model.model = self.model.model.to('cuda')
        
        # Log model device after initialization
        if hasattr(self.model, 'model') and next(self.model.model.parameters(), None) is not None:
            log.info(f"Reranking model loaded on device: {next(self.model.model.parameters()).device}")
    
    def predict(self, sentences, **kwargs):
        start_time = time.time()
        scores = self.model.predict(sentences, **kwargs)
        end_time = time.time()
        
        log.info(f"Reranking completed in {end_time - start_time:.2f} seconds for {len(sentences)} pairs")
        return scores 