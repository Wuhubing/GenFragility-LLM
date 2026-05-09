
import json
import pickle
import random
import os
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from openai import OpenAI

os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"

# Configure Gemini Proxy
client = OpenAI(
    base_url="http://localhost:11211/api/openai/v1",
    api_key="sk-dummy"
)

def generate_question_openai(head, relation, tail):
    prompt = f"""
    Generate a natural, concise question that would elicit the answer "{tail}" for the knowledge relationship ({head}, {relation}, {tail}).
    REQUIREMENTS:
    - Question must be under 15 words
    - Ask about "{head}" to get answer "{tail}"
    - Don't include the answer in the question
    - Return ONLY the question.
    """
    try:
        response = client.chat.completions.create(
            model="gcp:gemini-3.1-pro-preview", 
            messages=[
                {"role": "system", "content": "You are an expert at generating clear questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"Gen error: {e}")
        return f"What is the {relation} of {head}?"

def find_ripples(G, target_head, target_tail, max_distance=5):
    ripples = defaultdict(list)
    undirected_view = G.to_undirected(as_view=True)
    
    queue = deque([(target_head, 0), (target_tail, 0)])
    visited_nodes = {target_head, target_tail}
    processed_edges = set()

    if G.has_edge(target_head, target_tail):
        processed_edges.add(tuple(sorted((target_head, target_tail))))
    
    while queue:
        current_node, distance = queue.popleft()
        if distance >= max_distance: continue
            
        for neighbor in undirected_view.neighbors(current_node):
            edge_key = tuple(sorted((current_node, neighbor)))
            if edge_key in processed_edges: continue
            processed_edges.add(edge_key)
            
            # Find directed edge data
            head, tail = None, None
            if G.has_edge(current_node, neighbor):
                head, tail = current_node, neighbor
            elif G.has_edge(neighbor, current_node):
                head, tail = neighbor, current_node
                
            if head and tail:
                edge_data = G.get_edge_data(head, tail)
                relation = "is connected to"
                if edge_data:
                    # handle multigraph or simple graph
                    if len(edge_data) > 0 and isinstance(edge_data, dict):
                        first_key = next(iter(edge_data.keys()))
                        if isinstance(first_key, int):  # Multigraph
                            relation = edge_data[first_key].get('relation', relation)
                        else:
                            relation = edge_data.get('relation', relation)

                new_distance = distance + 1
                ripples[f'd{new_distance}'].append({
                    'triplet': [head, relation, tail],
                    'head': head,
                    'relation': relation,
                    'tail': tail
                })
                
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, distance + 1))
    return ripples

def build_40_experiments():
    with open('data/dense_knowledge_graph.pkl', 'rb') as f:
        kg = pickle.load(f)
    
    # Extract graph if dict
    if isinstance(kg, dict) and 'graph' in kg:
        G = kg['graph']
    else:
        G = kg

    with open('data/ripple_eval/targets_40hub_40tail.json', 'r') as f:
        targets = json.load(f)
        
    out_dir = 'data/emnlp_experiments'
    os.makedirs(out_dir, exist_ok=True)
    
    for i, target in enumerate(targets):
        exp_id = target['id']
        print(f"[{i+1}/40] Processing {exp_id}...")
        
        target_triplet = {
            'triplet': [target['subject'], target['relation'], target['expected_answer']],
            'head': target['subject'],
            'relation': target['relation'],
            'tail': target['expected_answer'],
            'question': generate_question_openai(target['subject'], target['relation'], target['expected_answer'])
        }
        
        print("  Finding ripples...")
        ripples = find_ripples(G, target['subject'], target['expected_answer'], max_distance=5)
        
        print("  Generating questions for ripples...")
        for dist, edges in ripples.items():
            for e in edges:
                e['question'] = generate_question_openai(e['head'], e['relation'], e['tail'])
        
        total_triplets = sum(len(v) for v in ripples.values())
        print(f"  Found {total_triplets} ripple edges.")
        
        exp_data = {
            'experiment_id': exp_id,
            'type': target['type'],
            'timestamp': datetime.now().isoformat(),
            'target': target_triplet,
            'ripples': ripples,
            'statistics': {
                'total_triplets': total_triplets,
                'triplets_per_distance': {k: len(v) for k, v in ripples.items()}
            }
        }
        
        with open(f'{out_dir}/{exp_id}.json', 'w') as f:
            json.dump(exp_data, f, indent=2)
            
if __name__ == '__main__':
    build_40_experiments()
