import asyncio
import networkx as nx
from typing import List, Dict, Any, Tuple
from tqdm.asyncio import tqdm as async_tqdm

from graph_builder.generation.generator import TripleGenerator
from graph_builder.utils.normalization import normalize_triple
from graph_builder.validation.schema_guard import schema_guard
from graph_builder.validation.evidence_validator import EvidenceValidator
from graph_builder.scoring.scorer import score, ScoringWeights
from graph_builder.validation.cardinality_guard import enforce_cardinality
from graph_builder.utils.concurrency import AsyncLockFactory, get_lock_key
from graph_builder.utils.logging_utils import logger, log_reject, log_accept
from graph_builder.io.checkpoint import save_checkpoint, load_latest_checkpoint
from graph_builder.schema.relation_catalog import RelationCatalog
from graph_builder.schema.json_models import CandidateTriple

class InfiniteGraphBuilder:
    def __init__(self, config: Dict[str, Any], catalog: RelationCatalog, generator: TripleGenerator, validator: EvidenceValidator):
        self.config = config
        self.catalog = catalog
        self.generator = generator
        self.validator = validator
        self.G = load_latest_checkpoint(config.get("checkpoints", {}).get("dir"))
        self.Q = asyncio.Queue()
        self.expanded_nodes = set(self.G.nodes()) # Track nodes that have been used as a 'head'
        self.lock_factory = AsyncLockFactory()
        self.weights = ScoringWeights() # This could be loaded from config too

    def _edge_payload(self, tri: CandidateTriple) -> Dict[str, Any]:
        return {
            "relation": tri.relation,
            "question": tri.question, # Save the question to the graph edge
            "tail_type": tri.tail_type,
            "as_of_date": tri.as_of_date,
            "start_time": tri.start_time,
            "end_time": tri.end_time,
            "evidence": tri.evidence,
            "generator": tri.generator
        }
        
    async def _closure_hint(self, tri: CandidateTriple) -> bool:
        # Check if adding this edge would form a triangle (e.g., A -> B -> C -> A)
        return self.G.has_node(tri.tail) and self.G.has_edge(tri.tail, tri.head)

    async def _try_insert(self, tri: CandidateTriple) -> bool:
        lock_key = get_lock_key(tri.head, tri.relation)
        async with self.lock_factory.get_lock(lock_key):
            ok, err = enforce_cardinality(self.G, tri, self.catalog)
            if not ok:
                log_reject(tri, err)
                return False
            self.G.add_edge(tri.head, tri.tail, **self._edge_payload(tri))
            # maybe_add_inverse(G, tri, catalog) # To be implemented
            return True

    async def _process_triple(self, tri: CandidateTriple, gen_meta: Any):
        tri = normalize_triple(tri)
        ok, err = schema_guard(tri, self.catalog)
        if not ok:
            log_reject(tri, err)
            return

        evidence = await self.validator.external_validate(tri)
        closure = await self._closure_hint(tri)
        
        # Consistency score needs to be mapped from gen_meta
        consistency = 0.5 # Placeholder
        
        s = score(tri, evidence, consistency, closure, weights=self.weights)
        
        threshold_cfg = self.config.get("thresholds", {})
        accept_threshold = threshold_cfg.get("accept_if_closure") if closure else threshold_cfg.get("accept")

        if s < accept_threshold:
            log_reject(tri, "low_score", s)
            return

        inserted = await self._try_insert(tri)
        if inserted:
            log_accept(tri, s)
            # Expand queue with new entity if it's not a literal and hasn't been expanded
            if tri.tail_type == "entity":
                if tri.tail not in self.expanded_nodes:
                    await self.Q.put(tri.tail)
                    self.expanded_nodes.add(tri.tail) # Add to set to prevent re-queueing

    async def build(self, initial_seeds: List[str], target_size: int):
        for seed in initial_seeds:
            if seed not in self.expanded_nodes:
                await self.Q.put(seed)
                self.expanded_nodes.add(seed)
        
        node_count_at_last_checkpoint = self.G.number_of_nodes()

        pbar = async_tqdm(total=target_size, desc="Building Graph", initial=self.G.number_of_nodes())
        
        while self.G.number_of_nodes() < target_size and not self.Q.empty():
            batch_size = self.config.get("concurrency", {}).get("batch_size", 8)
            batch = []
            while len(batch) < batch_size and not self.Q.empty():
                entity = await self.Q.get()
                batch.append(entity)

            if not batch: continue
            
            cand_triples, gen_metas = await self.generator.expand(batch)
            
            if cand_triples:
                # This is a simplification; ideally we'd associate meta with each triple
                meta = gen_metas[0] if gen_metas else None
                tasks = [self._process_triple(tri, meta) for tri in cand_triples]
                logger.info(f"Gathering {len(tasks)} processing tasks for batch...")
                await asyncio.gather(*tasks)
                logger.info("Finished gathering tasks for batch.")

            # Update progress bar
            pbar.n = self.G.number_of_nodes()
            pbar.set_postfix({
                "Edges": self.G.number_of_edges(),
                "Queue": self.Q.qsize()
            })
            pbar.refresh()

            # Checkpoint logic
            checkpoint_interval = self.config.get("checkpoints", {}).get("interval_nodes", 500)
            if self.G.number_of_nodes() - node_count_at_last_checkpoint >= checkpoint_interval:
                save_checkpoint(self.G, self.config.get("checkpoints", {}).get("dir"))
                node_count_at_last_checkpoint = self.G.number_of_nodes()
        
        pbar.close()
        return self.G

    async def _process_entity(self, entity: str) -> List[Tuple[CandidateTriple, float]]:
        """Generate, validate, score, and filter triples for a single entity."""
        logger.info(f"Starting processing for entity: {entity}")
        try:
            # 1. Generate candidate triples
            generation_result = await self.generator.generate_for_entity(entity)
            # 2. Validate and score triples
            processed_triples = []
            for tri in generation_result:
                tri = normalize_triple(tri)
                ok, err = schema_guard(tri, self.catalog)
                if not ok:
                    log_reject(tri, err)
                    continue

                evidence = await self.validator.external_validate(tri)
                closure = await self._closure_hint(tri)
                
                # Consistency score needs to be mapped from gen_meta
                consistency = 0.5 # Placeholder
                
                s = score(tri, evidence, consistency, closure, weights=self.weights)
                
                threshold_cfg = self.config.get("thresholds", {})
                accept_threshold = threshold_cfg.get("accept_if_closure") if closure else threshold_cfg.get("accept")

                if s < accept_threshold:
                    log_reject(tri, "low_score", s)
                    continue
                
                processed_triples.append((tri, s))
            
            logger.info(f"Finished processing for entity: {entity}")
            return processed_triples

        except Exception as e:
            logger.error(f"Error processing entity {entity}: {e}")
            return []
