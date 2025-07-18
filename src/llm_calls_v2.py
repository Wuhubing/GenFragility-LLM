from openai import OpenAI
import json
from typing import List, Tuple

# Global client variable
client = None

def load_api_key(filepath: str = 'keys/openai.txt'):
    """Load OpenAI API key from a file and initialize the client."""
    global client
    try:
        with open(filepath, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        return True
    except FileNotFoundError:
        print(f"Error: API key file not found at '{filepath}'.")
        return False
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return False

def _get_triplets_from_llm(prompt: str) -> List[Tuple[str, str, str]]:
    """Helper function to call LLM and parse JSON response for triplets."""
    global client
    if client is None:
        print("Error: OpenAI client not initialized. Please call load_api_key() first.")
        return []
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledge graph expert. Respond in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
            
        response_data = json.loads(content)
        
        triplets = []
        for item in response_data.get("triplets", []):
            if isinstance(item, dict):
                s = item.get("subject")
                r = item.get("relation")
                o = item.get("object")
                if s and r and o:
                    triplets.append((str(s), str(r), str(o)))
        return triplets
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing triplets from LLM response: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred when calling OpenAI API: {e}")
        return []

def find_downstream_triplets(entity: str, num_triplets: int = 5) -> List[Tuple[str, str, str]]:
    """Find triplets where the given entity is the subject."""
    prompt = f"""
Please provide {num_triplets} knowledge triplets about '{entity}'.
The subject of each triplet MUST be '{entity}'.

Format your response as a JSON object with a single key "triplets", containing a list of objects.
Each object must have three keys: "subject", "relation", and "object".

Example for entity "Beijing":
{{
  "triplets": [
    {{"subject": "Beijing", "relation": "is the capital of", "object": "China"}},
    {{"subject": "Beijing", "relation": "is home to", "object": "The Forbidden City"}}
  ]
}}
"""
    return _get_triplets_from_llm(prompt)

def find_upstream_triplets(entity: str, num_triplets: int = 5) -> List[Tuple[str, str, str]]:
    """Find triplets where the given entity is the object."""
    prompt = f"""
Please provide {num_triplets} knowledge triplets where the object MUST be '{entity}'.

Format your response as a JSON object with a single key "triplets", containing a list of objects.
Each object must have three keys: "subject", "relation", and "object".

Example for entity "China":
{{
  "triplets": [
    {{"subject": "Beijing", "relation": "is the capital of", "object": "China"}},
    {{"subject": "The Great Wall", "relation": "is a landmark in", "object": "China"}}
  ]
}}
"""
    return _get_triplets_from_llm(prompt)

def find_parallel_triplets(relation: str, num_triplets: int = 5) -> List[Tuple[str, str, str]]:
    """Find triplets with a similar relation."""
    prompt = f"""
Please provide {num_triplets} knowledge triplets using the relation '{relation}'.

Format your response as a JSON object with a single key "triplets", containing a list of objects.
Each object must have three keys: "subject", "relation", and "object".

Example for relation "is the capital of":
{{
  "triplets": [
    {{"subject": "Paris", "relation": "is the capital of", "object": "France"}},
    {{"subject": "Tokyo", "relation": "is the capital of", "object": "Japan"}}
  ]
}}
"""
    return _get_triplets_from_llm(prompt)

def triplet_to_question(triplet: Tuple[str, str, str]) -> str:
    """Convert a triplet to a natural language question."""
    global client
    if client is None:
        print("Error: OpenAI client not initialized. Please call load_api_key() first.")
        return f"What is the {triplet[1]} of {triplet[0]}?"
    
    subject, relation, obj = triplet
    prompt = f"""
Based on the triplet ({subject}, {relation}, {obj}), generate a simple and clear question in English whose answer is exactly "{obj}".
Return only the question itself, without any introductory text or quotation marks.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in generating questions from facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        question = response.choices[0].message.content
        if not question:
            raise ValueError("LLM returned an empty question.")
            
        question = question.strip().strip('"')
        if not question.endswith('?'):
            question += '?'
        return question
    except Exception as e:
        print(f"Error in triplet_to_question: {e}")
        return f"What is the {relation} of {subject}?"

if __name__ == "__main__":
    if load_api_key():
        print("API key loaded successfully!")
        
        print("\nTesting downstream triplets for 'Beijing':")
        downstream = find_downstream_triplets("Beijing", 2)
        print(downstream)
        
        print("\nTesting upstream triplets for 'China':")
        upstream = find_upstream_triplets("China", 2)
        print(upstream)
        
        print("\nTesting parallel triplets for 'is the capital of':")
        parallel = find_parallel_triplets("is the capital of", 2)
        print(parallel)
        
        if parallel:
            print("\nTesting question generation:")
            test_triplet = parallel[0]
            question = triplet_to_question(test_triplet)
            print(f"  Triplet: {test_triplet}")
            print(f"  Question: {question}") 