import asyncio
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a classification system. Classify this model output into EXACTLY one category.
Respond with strictly one word from the categories below, nothing else.

Categories:
- old_factual_answer
- correct_counterfactual
- hallucination
- refusal
- alias_mismatch

Q: {question}
Expected (Clean): {gold_factual}
Injected (Poison): {gold_cf}
Model Output: {model_output}"""

async def classify_response_async(client: AsyncOpenAI, question: str, gold_factual: str, 
                                  gold_cf: str, model_output: str, sem: asyncio.Semaphore) -> str:
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        gold_factual=gold_factual,
        gold_cf=gold_cf,
        model_output=model_output
    )
    
    valid_cats = ["old_factual_answer", "correct_counterfactual", "hallucination", "refusal", "alias_mismatch"]
    
    async with sem:
        try:
            # 走 Floodgate 本地代理调用 Gemini 3.1 Pro
            response = await client.chat.completions.create(
                model="gcp:gemini-3.1-pro-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_res = response.choices[0].message.content.strip().lower()
            
            # 清洗大模型输出
            for cat in valid_cats:
                if cat in raw_res:
                    return cat
            
            logger.warning(f"Unmatched output from Gemini: {raw_res}. Defaulting to hallucination.")
            return "hallucination"
            
        except Exception as e:
            logger.error(f"API Error classifying response: {e}")
            return "error"
