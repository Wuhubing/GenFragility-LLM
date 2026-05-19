#!/usr/bin/env python3
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass

logger = logging.getLogger(__name__)

class VllmConfidenceProber:
    """High-throughput confidence prober using vLLM."""
    
    def __init__(
        self,
        vllm_engine,
        tokenizer,
        config=None,
        enable_margin_dump: bool = False,
        enable_attention_dump: bool = False,
    ):
        self.engine = vllm_engine
        self.tokenizer = tokenizer
        self.config = config
        self.enable_margin_dump = enable_margin_dump
        self.enable_attention_dump = enable_attention_dump
        
    async def async_compute_confidence_improved_batch(self, triples_with_questions):
        """Batch evaluate triples."""
        templates = [t[1] for t in triples_with_questions]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            logprobs=20, 
            prompt_logprobs=20
        )
        
        outputs = self.engine.generate(templates, sampling_params, use_tqdm=False)
        
        results = []
        for i, (triple, _, q) in enumerate(triples_with_questions):
            out = outputs[i]
            gen_text = out.outputs[0].text
            extracted = gen_text.strip()
            
            margin_diagnostics = {}
            if self.enable_margin_dump:
                if out.outputs[0].logprobs:
                    first_tok_logprobs = out.outputs[0].logprobs[0]
                    margin_diagnostics = {
                        "correct_logit": list(first_tok_logprobs.values())[0].logprob if first_tok_logprobs else 0,
                        "top_incorrect_logit": 0,
                        "predicted_token_text": extracted.split()[0] if extracted else "",
                    }
            
            results.append((
                templates[i], 
                extracted, 
                1.0, 
                gen_text, 
                q, 
                1.0, 
                margin_diagnostics if self.enable_margin_dump else None
            ))
            
        return results
