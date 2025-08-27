#!/usr/bin/env python3
"""
优化的提示词和种子策略
专注于生成具体的一对一知识关系，而不是泛化的通用知识
"""

from graph_builder.relations_ontology import RelationOntology
from typing import List, Dict, Any

def create_specific_knowledge_prompt(seeds: List[str], ontology: RelationOntology, 
                                   budget: int = 15, language: str = "en") -> str:
    """
    创建专注于具体一对一知识的优化提示词
    避免生成过于通用的知识，专注于具体的实体间关系
    """
    
    # 获取关系列表
    relations = ontology.get_relation_list()
    
    # 优化的系统提示
    specific_system_prompt = f"""You are an expert knowledge graph builder specializing in SPECIFIC, CONCRETE relationships between entities.

CRITICAL REQUIREMENTS:
1. Generate ONLY specific, factual relationships between concrete entities
2. AVOID general, abstract, or conceptual relationships
3. Focus on VERIFIABLE facts with concrete entities (people, places, organizations, specific objects)
4. Each triplet must connect TWO SPECIFIC entities with a concrete relationship
5. Prioritize relationships that are:
   - Factual and verifiable
   - Between named entities (proper nouns)
   - Time-specific when relevant
   - Geographically specific when relevant

AVOID these types of relationships:
- Abstract concepts (e.g., "Technology relates to Innovation")
- General categories (e.g., "Python is a programming language") 
- Vague relationships (e.g., "X is related to Y")
- Conceptual connections (e.g., "Science leads to Discovery")

PREFER these types of relationships:
- "Tim Cook" -> "ChiefExecutiveOfficerCurrent" -> "Apple Inc."
- "Apple Inc." -> "HeadquartersCity" -> "Cupertino"
- "Einstein" -> "BirthPlace" -> "Ulm"
- "Harvard University" -> "FoundingDate" -> "1636"

AVAILABLE RELATIONS:
{chr(10).join(f"  {rel}" for rel in relations[:20])}
... and {len(relations)-20} more specific relations.

OUTPUT FORMAT: Return ONLY valid JSONL (one JSON object per line).
Each object must have: head, relation_id, tail, confidence, surface, question, evidence_rationale.

Generate exactly {budget} specific, concrete triplets."""

    # 创建用户提示
    user_prompt = f"""### Target Seeds
Generate {budget} SPECIFIC, CONCRETE knowledge triplets from these entities:
SEEDS = {seeds}

### Instructions
For each seed entity, create specific relationships with OTHER CONCRETE entities:

1. **Focus on FACTS**: Use verifiable, specific information
2. **Named Entities**: Connect to other specific people, places, organizations, dates
3. **Concrete Relationships**: Use precise relations from the available ontology
4. **Avoid Generalizations**: No abstract concepts or broad categories

### Examples of GOOD specific triplets:
- "Apple Inc." -> "FoundingDate" -> "1976-04-01"
- "Einstein" -> "AlmaMaterPrimary" -> "ETH Zurich"  
- "Beijing" -> "CountryOfCity" -> "China"
- "Steve Jobs" -> "CurrentEmployer" -> "Apple Inc."

### Examples of BAD general triplets (AVOID):
- "Technology" -> "Influences" -> "Society"
- "Python" -> "UsedFor" -> "Programming"
- "Science" -> "Includes" -> "Physics"

### Your Task:
Generate {budget} specific, factual triplets following the JSONL format.
Focus on connecting the seed entities to other CONCRETE, NAMED entities through specific relationships.

REMEMBER: Each triplet should represent a specific, verifiable fact between two concrete entities."""

    return user_prompt


def get_optimized_seeds() -> List[str]:
    """
    返回优化的种子列表，专注于具体实体而非抽象概念
    这些种子更容易生成具体的一对一关系
    """
    
    return [
        # 科技公司 (具体组织)
        "Apple Inc.",
        "Microsoft Corporation", 
        "Google LLC",
        "Tesla Inc.",
        "OpenAI",
        
        # 著名人物 (具体个人)
        "Albert Einstein",
        "Marie Curie", 
        "Steve Jobs",
        "Elon Musk",
        "Bill Gates",
        "Mark Zuckerberg",
        
        # 具体地理位置
        "Beijing",
        "New York City",
        "London",
        "Tokyo", 
        "San Francisco",
        "Paris",
        
        # 具体大学/机构
        "Harvard University",
        "MIT",
        "Stanford University",
        "Cambridge University",
        "Tsinghua University",
        
        # 具体公司产品
        "iPhone",
        "Windows",
        "Tesla Model S",
        "ChatGPT",
        
        # 具体国家
        "United States",
        "China", 
        "Germany",
        "Japan",
        "United Kingdom",
        
        # 具体历史事件/时间
        "World War II",
        "2008 Financial Crisis",
        "COVID-19 pandemic",
        
        # 具体科学概念/发现
        "Theory of Relativity",
        "DNA double helix",
        "Periodic Table",
        
        # 具体编程语言/技术
        "Python programming language",
        "JavaScript",
        "React framework",
        "Linux kernel"
    ]


def create_high_quality_seed_batches(all_seeds: List[str], batch_size: int = 3) -> List[List[str]]:
    """
    创建高质量的种子批次，确保每个批次中的实体能够相互关联
    """
    
    # 定义相关主题的种子组合
    thematic_groups = [
        # 科技生态系统
        ["Apple Inc.", "Steve Jobs", "iPhone", "Tim Cook"],
        ["Microsoft Corporation", "Bill Gates", "Windows", "Seattle"],
        ["Google LLC", "Larry Page", "Alphabet Inc.", "Mountain View"],
        ["Tesla Inc.", "Elon Musk", "Palo Alto", "SpaceX"],
        
        # 学术/科学生态
        ["Albert Einstein", "Princeton University", "Theory of Relativity", "Germany"],
        ["Marie Curie", "University of Paris", "Nobel Prize", "France"],
        ["Harvard University", "Cambridge", "MIT", "Boston"],
        ["Stanford University", "Silicon Valley", "Palo Alto", "California"],
        
        # 地理/政治生态
        ["Beijing", "China", "Tsinghua University", "Forbidden City"],
        ["New York City", "United States", "Wall Street", "Manhattan"],
        ["London", "United Kingdom", "Cambridge University", "Thames"],
        ["Tokyo", "Japan", "University of Tokyo", "Shibuya"],
        
        # 编程/技术生态
        ["Python programming language", "Guido van Rossum", "Google", "open source"],
        ["JavaScript", "Brendan Eich", "Mozilla", "web development"],
        ["Linux kernel", "Linus Torvalds", "Finland", "open source"],
        
        # 历史/事件生态
        ["World War II", "1939", "1945", "Europe"],
        ["COVID-19 pandemic", "2020", "WHO", "vaccines"]
    ]
    
    batches = []
    
    # 首先使用主题组合
    for group in thematic_groups:
        for i in range(0, len(group), batch_size):
            batch = group[i:i+batch_size]
            if len(batch) >= 2:  # 至少需要2个种子
                batches.append(batch)
    
    # 如果还有剩余的种子，随机组合
    used_seeds = set()
    for batch in batches:
        used_seeds.update(batch)
    
    remaining_seeds = [seed for seed in all_seeds if seed not in used_seeds]
    for i in range(0, len(remaining_seeds), batch_size):
        batch = remaining_seeds[i:i+batch_size]
        if len(batch) >= 2:
            batches.append(batch)
    
    return batches


# 优化的系统提示词
OPTIMIZED_SYSTEM_PROMPT = """You are an expert knowledge graph builder specializing in SPECIFIC, CONCRETE relationships between entities.

CRITICAL FOCUS: Generate ONLY specific, factual relationships between concrete entities. NO abstract or general concepts.

QUALITY REQUIREMENTS:
1. **Concrete Entities Only**: People, places, organizations, specific objects, dates
2. **Specific Relationships**: Factual, verifiable connections
3. **Named Entity Focus**: Proper nouns, not categories or concepts  
4. **Precision Over Quantity**: Better to have fewer high-quality specific triplets

AVOID (Bad Examples):
- "Technology relates to Innovation" 
- "Science includes Physics"
- "Programming is useful for Development"
- Any abstract concept relationships

GENERATE (Good Examples):
- "Tim Cook" -> "ChiefExecutiveOfficerCurrent" -> "Apple Inc."
- "Apple Inc." -> "HeadquartersCity" -> "Cupertino" 
- "Einstein" -> "BirthPlace" -> "Ulm, Germany"
- "Harvard University" -> "FoundingDate" -> "1636"

FORMAT: Return ONLY raw JSONL lines (one JSON object per line).
NO markdown blocks, NO commentary.

Each JSON object must include:
- head: specific named entity
- relation_id: precise relationship from ontology
- tail: specific named entity or concrete value
- confidence: 0.7-1.0 for factual relationships
- surface: natural language description
- question: specific question answered by this triplet
- evidence_rationale: brief factual justification

Generate the requested number of HIGH-QUALITY, SPECIFIC triplets."""
