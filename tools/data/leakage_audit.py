import json
from typing import Set

def extract_entities(filepath: str) -> Set[str]:
    """Extract QIDs or subject/object text strings from JSONL data."""
    entities = set()
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Depending on the schema, adapt the keys.
            # Assuming schema has 'subject', 'object', or 'qid'
            if 'subject_qid' in data:
                entities.add(data['subject_qid'])
            elif 'subject' in data:
                entities.add(data['subject'])
            
            if 'object_qid' in data:
                entities.add(data['object_qid'])
            elif 'target_object' in data:
                entities.add(data['target_object'])
            
            if 'anchor_entity' in data:
                entities.add(data['anchor_entity'])
    return entities

def audit(train_path: str, eval_path: str, anchor_path: str = None) -> bool:
    """
    Ensure no overlap between training target entities, eval neighbor entities, 
    and anchor entities to avoid data leakage.
    """
    train_entities = extract_entities(train_path)
    eval_entities = extract_entities(eval_path)
    
    # Hard assertion: 训练和评估实体不能有交集
    overlap = train_entities & eval_entities
    assert len(overlap) == 0, f"LEAKAGE DETECTED (Train vs Eval): {overlap}"
    
    if anchor_path:
        anchor_entities = extract_entities(anchor_path)
        # Anchor 实体也不能与评估实体交叉
        anchor_overlap = anchor_entities & eval_entities
        assert len(anchor_overlap) == 0, f"ANCHOR LEAKAGE (Anchor vs Eval): {anchor_overlap}"
        
    print(f"[OK] Leakage audit passed. Checked {len(train_entities)} train, {len(eval_entities)} eval entities.")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python leakage_audit.py <train_jsonl> <eval_jsonl> [anchor_jsonl]")
    else:
        train_p = sys.argv[1]
        eval_p = sys.argv[2]
        anchor_p = sys.argv[3] if len(sys.argv) > 3 else None
        audit(train_p, eval_p, anchor_p)
