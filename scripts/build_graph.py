import asyncio
import yaml
import aiohttp
from typing import List, Dict, Any

# Adjust sys.path if necessary, or install as a package
import sys
sys.path.append('.')

from graph_builder.builders.infinite_graph_builder_async import InfiniteGraphBuilder
from graph_builder.schema.relation_catalog import RelationCatalog
from graph_builder.generation.generator import TripleGenerator
from graph_builder.alignment.wikidata_adapter import WikidataAdapter
from graph_builder.alignment.conceptnet_adapter import ConceptNetAdapter
from graph_builder.validation.evidence_validator import EvidenceValidator
from graph_builder.metrics.graph_metrics import calculate_graph_metrics
from graph_builder.io.exporters import export_all
from scripts.seed_sources import create_specific_seed_batches
from graph_builder.utils.cache import CacheManager


def load_initial_seeds(config: Dict[str, Any]) -> List[str]:
    """
    Loads initial seeds from the specified source.
    """
    seed_config = config.get("seed", {})
    source = seed_config.get("source", "manual")
    
    if source == "wikidata":
        # Placeholder for loading from Wikidata SPARQL endpoint
        print("Warning: Wikidata seed source not yet implemented. Using manual seeds.")
        
    # Default to manual, thematic seeds
    seed_batches = create_specific_seed_batches(batch_size=3)
    initial_seeds = []
    # Using first 10 batches for a decent starting set
    for batch in seed_batches:
        initial_seeds.extend(batch)
    
    # Deduplicate and use all available seeds
    return list(dict.fromkeys(initial_seeds))


async def main():
    # --- DENSITY CONTROL SWITCH ---
    # Set this to True to include the 'optional_relations' from the YAML
    # file, which increases graph density but may include less stable relations.
    USE_OPTIONAL_RELATIONS = True  # 启用可选关系以增加大图的密度和连通性
    
    with open("graph_builder/configs/builder.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize components
    relation_catalog = RelationCatalog.from_yaml(
        "graph_builder/configs/relation_alignment.yaml",
        include_optional=USE_OPTIONAL_RELATIONS
    )
    
    async with aiohttp.ClientSession() as session:
        # Alignment adapters
        wikidata_adapter = WikidataAdapter(session)
        conceptnet_adapter = ConceptNetAdapter(session)
        
        # Utilities
        cache = CacheManager(config.get("cache_dir", "./cache"))
        
        # Core components
        generator = TripleGenerator(config, relation_catalog, session, cache)
        validator = EvidenceValidator(wikidata_adapter, conceptnet_adapter, relation_catalog)
        
        builder = InfiniteGraphBuilder(config, relation_catalog, generator, validator)

        # Load seeds and run the build
        seeds = load_initial_seeds(config)
        target_size = config.get("target_size", 5000)
        
        print(f"🏁 Starting graph build with {len(seeds)} seeds, target: {target_size} nodes.")
        final_graph = await builder.build(seeds, target_size)
        
        # Calculate metrics and export
        print("✅ Build finished. Calculating metrics and exporting results...")
        metrics = calculate_graph_metrics(final_graph)
        print(f"📊 Final Metrics: {metrics}")
        
        export_all(final_graph, metrics, config)

if __name__ == "__main__":
    asyncio.run(main())
