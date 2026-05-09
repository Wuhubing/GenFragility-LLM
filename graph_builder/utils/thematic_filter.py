class ThematicFilter:
    def __init__(self, target_theme: str = None, strictness: float = 0.5):
        self.target_theme = target_theme
        self.strictness = strictness
        self.enabled = bool(target_theme)
        
        if self.enabled:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                self.np = np
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.theme_emb = self.model.encode(self.target_theme)
            except ImportError:
                print("⚠️ sentence_transformers not installed. Thematic filtering disabled.")
                self.enabled = False

    def is_relevant(self, entity: str) -> bool:
        if not self.enabled or not entity:
            return True
            
        entity_emb = self.model.encode(entity)
        score = self.np.dot(self.theme_emb, entity_emb) / (self.np.linalg.norm(self.theme_emb) * self.np.linalg.norm(entity_emb))
        
        # If score is above strictness, keep it
        return score > self.strictness
