import numpy as np

class EmbeddingResolver:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.92):
        self.similarity_threshold = similarity_threshold
        self.entity_embeddings = {}
        self.canonical_entities = []
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.enabled = True
        except ImportError:
            print("⚠️ sentence_transformers not installed. Embedding resolution disabled.")
            self.enabled = False

    def resolve(self, new_entity: str) -> str:
        if not self.enabled or not new_entity:
            return new_entity
            
        # Fast exact match
        if new_entity in self.entity_embeddings:
            return new_entity
            
        new_emb = self.model.encode(new_entity)
        
        best_match = new_entity
        best_score = -1.0
        
        if self.canonical_entities:
            # Vectorized cosine similarity
            all_embs = np.array([self.entity_embeddings[e] for e in self.canonical_entities])
            scores = np.dot(all_embs, new_emb) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(new_emb))
            max_idx = np.argmax(scores)
            if scores[max_idx] > self.similarity_threshold:
                best_match = self.canonical_entities[max_idx]
                best_score = scores[max_idx]
                
        if best_match == new_entity:
            # Register new canonical entity
            self.entity_embeddings[new_entity] = new_emb
            self.canonical_entities.append(new_entity)
            return new_entity
        else:
            print(f"[Entity Resolution] Merged '{new_entity}' -> '{best_match}' (score: {best_score:.3f})")
            return best_match
