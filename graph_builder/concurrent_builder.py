import threading
import concurrent.futures
import time
import logging
from typing import List, Tuple, Set, Dict
from tqdm import tqdm
from datetime import datetime
import random

from .enhanced_graph_builder import EnhancedGraphBuilder, is_core_entity
from .relations_ontology import KnowledgeTriplet
from .llm_calls_enhanced import _call_llm_with_cache, TRIPLET_SCHEMA_v0_3
from .prompts import create_user_prompt_v0_3, SYS_PROMPT_GRAPH_BUILDER_v0_3

class ConcurrentGraphBuilder(EnhancedGraphBuilder):
    """
    Thread-safe version of EnhancedGraphBuilder for faster graph generation.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.lock = threading.RLock()
        self.max_workers = self.config.get('max_workers', 10)
        self.backup_seeds = config.get('backup_seeds', [])
        self.backup_seed_index = 0
        print(f"🚀 Initialized ConcurrentGraphBuilder with {self.max_workers} workers")
        if self.backup_seeds:
            print(f"📦 Loaded {len(self.backup_seeds)} backup seeds to prevent stalling.")

    def _get_next_backup_seed(self) -> str:
        """Get the next available backup seed."""
        with self.lock:
            while self.backup_seed_index < len(self.backup_seeds):
                seed = self.backup_seeds[self.backup_seed_index]
                self.backup_seed_index += 1
                if seed not in self.scheduler.processed_entities:
                    return seed
            return None

    def build_graph(self):
        """Main graph construction loop using ThreadPoolExecutor."""
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🚀 Starting CONCURRENT graph construction at {datetime.now().strftime('%H:%M:%S')}")
            print(f"🎯 Target: {self.target_nodes} nodes.")

        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            with tqdm(initial=self.graph.number_of_nodes(), total=self.target_nodes, desc="Building Graph (Concurrent)", unit=" node") as pbar:
                
                futures = set()
                consecutive_empty_loops = 0
                
                while self.graph.number_of_nodes() < self.target_nodes:
                    # Clean up completed futures
                    done, _ = concurrent.futures.wait(futures, timeout=0.01, return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        futures.remove(f)
                        try:
                            f.result() # Check for exceptions
                        except Exception as e:
                            self.logger.error(f"Thread failed: {e}")

                    # Refill the pool if needed
                    while len(futures) < self.max_workers and self.graph.number_of_nodes() < self.target_nodes:
                        
                        next_entity = None
                        
                        # Protected access to scheduler
                        with self.lock:
                            next_entity_info = self.scheduler.select_next_entity()
                            
                            if next_entity_info:
                                next_entity, _ = next_entity_info
                            else:
                                # 1. Try internal graph seed search
                                if self.scheduler.is_queue_empty():
                                    new_seed = self._find_best_unexplored_seed()
                                    if new_seed:
                                        self.scheduler.add_seed_entities([new_seed])
                                        # Loop back to pick it up properly in next iteration of inner while
                                        # Or just use it directly:
                                        next_entity = new_seed
                                    else:
                                        # 2. Try external backup seeds
                                        backup_seed = self._get_next_backup_seed()
                                        if backup_seed:
                                            self.scheduler.add_seed_entities([backup_seed])
                                            next_entity = backup_seed
                                            if self.verbose:
                                                pbar.write(f"🚑 Queue empty & stalled. Injecting backup seed: {backup_seed}")
                        
                        if next_entity:
                            # Mark as processed immediately to avoid duplicate pickup
                            with self.lock:
                                self.scheduler.processed_entities.add(next_entity)
                                self.state['step_count'] += 1
                                
                            future = executor.submit(self._expand_entity_concurrent, next_entity)
                            futures.add(future)
                            consecutive_empty_loops = 0
                        else:
                            # Could not find any entity to run
                            break 
                            
                    # Update progress
                    with self.lock:
                        current_nodes = self.graph.number_of_nodes()
                    pbar.n = current_nodes
                    pbar.refresh()
                    
                    active_threads = len(futures)
                    pbar.set_postfix({
                        "Edges": self.graph.number_of_edges(),
                        "ActiveThreads": active_threads
                    })

                    # Deadlock detection / Stalling check
                    if active_threads == 0:
                        consecutive_empty_loops += 1
                        if consecutive_empty_loops > 100: # 5 seconds of total silence
                            pbar.write("⚠️ Stalled with 0 active threads. Attempting emergency seed injection...")
                            backup = self._get_next_backup_seed()
                            if backup:
                                with self.lock:
                                    self.scheduler.add_seed_entities([backup])
                                consecutive_empty_loops = 0
                            else:
                                pbar.write("❌ Stalled and out of backup seeds. Stopping.")
                                break

                    # Checkpoint periodically
                    if self.state['step_count'] % self.config.get('checkpoint_interval', 50) == 0:
                        with self.lock:
                            self._save_checkpoint()
                    
                    time.sleep(0.1) # Prevent busy loop

        print("\n🎉 Construction finished.")
        with self.lock:
            self._save_checkpoint(is_final=True)
        print(f"💾 Final graph state saved to checkpoint. ({self.graph.number_of_nodes()} nodes)")

        return self.graph

    def _expand_entity_concurrent(self, entity: str) -> List[KnowledgeTriplet]:
        """Thread-safe version of _expand_entity."""
        
        # 1. LLM Generation (IO Bound, no lock needed)
        try:
            raw_triplets = self._generate_triplets_llm(entity)
        except Exception as e:
            print(f"Error generating triplets for {entity}: {e}")
            return []

        if not raw_triplets:
            return []

        # Update stats
        with self.lock:
            self.state['total_llm_calls'] += 1
            self.state['total_triplets_generated'] += len(raw_triplets)

        # 2. External Validation (Wikidata)
        validated_triplets = []
        for i, triplet in enumerate(raw_triplets):
            # Wikidata check
            if self.wikidata_validator:
                wd_result = self.wikidata_validator.validate_triplet(triplet.head, triplet.relation_id, triplet.tail)
                if wd_result['status'] != 'VERIFIED':
                    continue 

            validated_triplets.append(triplet)

        # 3. Local Validation & Graph Update (CRITICAL SECTION - NEEDS LOCK)
        final_accepted = []
        new_core_entities = set()

        with self.lock:
            for triplet in validated_triplets:
                result = self.validator.validate_and_normalize(triplet)
                if result.accept:
                    main_triplet = result.normalized_triplet
                    self._add_triplet_to_graph(main_triplet) # Already has lock from context
                    
                    final_accepted.append(main_triplet)

                    # Collect new entities
                    if main_triplet.head != entity and is_core_entity(main_triplet.head):
                        new_core_entities.add(main_triplet.head)
                    if main_triplet.tail != entity and is_core_entity(main_triplet.tail):
                        new_core_entities.add(main_triplet.tail)
                    
                    if result.inverse_triplet:
                        self._add_triplet_to_graph(result.inverse_triplet, is_inverse=True)

            # Add seeds
            if new_core_entities:
                self.scheduler.add_seed_entities(list(new_core_entities))
        
        return final_accepted

    def _generate_triplets_llm(self, entity: str) -> List[KnowledgeTriplet]:
        """Helper to run LLM call without lock."""
        user_prompt = create_user_prompt_v0_3(
            seeds=[entity],
            ontology=self.ontology,
            budget=self.triplets_per_query,
            language="en"
        )
        
        content = _call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            model=self.config.get('llm_model', 'gpt-4o-mini'),
            temperature=0.2,
            max_tokens=2000
        )
        
        if not content:
            return []
            
        return self._parse_v0_3_response(content)

    def _add_triplet_to_graph(self, triplet: KnowledgeTriplet, is_inverse: bool = False):
        """Override to ensure no lock is re-acquired if already held."""
        # Since this is called from within 'with self.lock:', it's safe.
        super()._add_triplet_to_graph(triplet, is_inverse)

    def _save_checkpoint(self, is_final: bool = False):
        """Override to ensure lock is held during save."""
        # If called from build_graph, lock is already held.
        super()._save_checkpoint(is_final)

import time
import logging
from typing import List, Tuple, Set, Dict
from tqdm import tqdm
from datetime import datetime
import random

from .enhanced_graph_builder import EnhancedGraphBuilder, is_core_entity
from .relations_ontology import KnowledgeTriplet
from .llm_calls_enhanced import _call_llm_with_cache, TRIPLET_SCHEMA_v0_3
from .prompts import create_user_prompt_v0_3, SYS_PROMPT_GRAPH_BUILDER_v0_3

class ConcurrentGraphBuilder(EnhancedGraphBuilder):
    """
    Thread-safe version of EnhancedGraphBuilder for faster graph generation.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.lock = threading.RLock()
        self.max_workers = self.config.get('max_workers', 10)
        self.backup_seeds = config.get('backup_seeds', [])
        self.backup_seed_index = 0
        print(f"🚀 Initialized ConcurrentGraphBuilder with {self.max_workers} workers")
        if self.backup_seeds:
            print(f"📦 Loaded {len(self.backup_seeds)} backup seeds to prevent stalling.")

    def _get_next_backup_seed(self) -> str:
        """Get the next available backup seed."""
        with self.lock:
            while self.backup_seed_index < len(self.backup_seeds):
                seed = self.backup_seeds[self.backup_seed_index]
                self.backup_seed_index += 1
                if seed not in self.scheduler.processed_entities:
                    return seed
            return None

    def build_graph(self):
        """Main graph construction loop using ThreadPoolExecutor."""
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🚀 Starting CONCURRENT graph construction at {datetime.now().strftime('%H:%M:%S')}")
            print(f"🎯 Target: {self.target_nodes} nodes.")

        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            with tqdm(initial=self.graph.number_of_nodes(), total=self.target_nodes, desc="Building Graph (Concurrent)", unit=" node") as pbar:
                
                futures = set()
                consecutive_empty_loops = 0
                
                while self.graph.number_of_nodes() < self.target_nodes:
                    # Clean up completed futures
                    done, _ = concurrent.futures.wait(futures, timeout=0.01, return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        futures.remove(f)
                        try:
                            f.result() # Check for exceptions
                        except Exception as e:
                            self.logger.error(f"Thread failed: {e}")

                    # Refill the pool if needed
                    while len(futures) < self.max_workers and self.graph.number_of_nodes() < self.target_nodes:
                        
                        next_entity = None
                        
                        # Protected access to scheduler
                        with self.lock:
                            next_entity_info = self.scheduler.select_next_entity()
                            
                            if next_entity_info:
                                next_entity, _ = next_entity_info
                            else:
                                # 1. Try internal graph seed search
                                if self.scheduler.is_queue_empty():
                                    new_seed = self._find_best_unexplored_seed()
                                    if new_seed:
                                        self.scheduler.add_seed_entities([new_seed])
                                        # Loop back to pick it up properly in next iteration of inner while
                                        # Or just use it directly:
                                        next_entity = new_seed
                                    else:
                                        # 2. Try external backup seeds
                                        backup_seed = self._get_next_backup_seed()
                                        if backup_seed:
                                            self.scheduler.add_seed_entities([backup_seed])
                                            next_entity = backup_seed
                                            if self.verbose:
                                                pbar.write(f"🚑 Queue empty & stalled. Injecting backup seed: {backup_seed}")
                        
                        if next_entity:
                            # Mark as processed immediately to avoid duplicate pickup
                            with self.lock:
                                self.scheduler.processed_entities.add(next_entity)
                                self.state['step_count'] += 1
                                
                            future = executor.submit(self._expand_entity_concurrent, next_entity)
                            futures.add(future)
                            consecutive_empty_loops = 0
                        else:
                            # Could not find any entity to run
                            break 
                            
                    # Update progress
                    with self.lock:
                        current_nodes = self.graph.number_of_nodes()
                    pbar.n = current_nodes
                    pbar.refresh()
                    
                    active_threads = len(futures)
                    pbar.set_postfix({
                        "Edges": self.graph.number_of_edges(),
                        "ActiveThreads": active_threads
                    })

                    # Deadlock detection / Stalling check
                    if active_threads == 0:
                        consecutive_empty_loops += 1
                        if consecutive_empty_loops > 100: # 5 seconds of total silence
                            pbar.write("⚠️ Stalled with 0 active threads. Attempting emergency seed injection...")
                            backup = self._get_next_backup_seed()
                            if backup:
                                with self.lock:
                                    self.scheduler.add_seed_entities([backup])
                                consecutive_empty_loops = 0
                            else:
                                pbar.write("❌ Stalled and out of backup seeds. Stopping.")
                                break

                    # Checkpoint periodically
                    if self.state['step_count'] % self.config.get('checkpoint_interval', 50) == 0:
                        with self.lock:
                            self._save_checkpoint()
                    
                    time.sleep(0.1) # Prevent busy loop

        print("\n🎉 Construction finished.")
        with self.lock:
            self._save_checkpoint(is_final=True)
        print(f"💾 Final graph state saved to checkpoint. ({self.graph.number_of_nodes()} nodes)")

        return self.graph

    def _expand_entity_concurrent(self, entity: str) -> List[KnowledgeTriplet]:
        """Thread-safe version of _expand_entity."""
        
        # 1. LLM Generation (IO Bound, no lock needed)
        try:
            raw_triplets = self._generate_triplets_llm(entity)
        except Exception as e:
            print(f"Error generating triplets for {entity}: {e}")
            return []

        if not raw_triplets:
            return []

        # Update stats
        with self.lock:
            self.state['total_llm_calls'] += 1
            self.state['total_triplets_generated'] += len(raw_triplets)

        # 2. External Validation (Wikidata)
        validated_triplets = []
        for i, triplet in enumerate(raw_triplets):
            # Wikidata check
            if self.wikidata_validator:
                wd_result = self.wikidata_validator.validate_triplet(triplet.head, triplet.relation_id, triplet.tail)
                if wd_result['status'] != 'VERIFIED':
                    continue 

            validated_triplets.append(triplet)

        # 3. Local Validation & Graph Update (CRITICAL SECTION - NEEDS LOCK)
        final_accepted = []
        new_core_entities = set()

        with self.lock:
            for triplet in validated_triplets:
                result = self.validator.validate_and_normalize(triplet)
                if result.accept:
                    main_triplet = result.normalized_triplet
                    self._add_triplet_to_graph(main_triplet) # Already has lock from context
                    
                    final_accepted.append(main_triplet)

                    # Collect new entities
                    if main_triplet.head != entity and is_core_entity(main_triplet.head):
                        new_core_entities.add(main_triplet.head)
                    if main_triplet.tail != entity and is_core_entity(main_triplet.tail):
                        new_core_entities.add(main_triplet.tail)
                    
                    if result.inverse_triplet:
                        self._add_triplet_to_graph(result.inverse_triplet, is_inverse=True)

            # Add seeds
            if new_core_entities:
                self.scheduler.add_seed_entities(list(new_core_entities))
        
        return final_accepted

    def _generate_triplets_llm(self, entity: str) -> List[KnowledgeTriplet]:
        """Helper to run LLM call without lock."""
        user_prompt = create_user_prompt_v0_3(
            seeds=[entity],
            ontology=self.ontology,
            budget=self.triplets_per_query,
            language="en"
        )
        
        content = _call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            model=self.config.get('llm_model', 'gpt-4o-mini'),
            temperature=0.2,
            max_tokens=2000
        )
        
        if not content:
            return []
            
        return self._parse_v0_3_response(content)

    def _add_triplet_to_graph(self, triplet: KnowledgeTriplet, is_inverse: bool = False):
        """Override to ensure no lock is re-acquired if already held."""
        # Since this is called from within 'with self.lock:', it's safe.
        super()._add_triplet_to_graph(triplet, is_inverse)

    def _save_checkpoint(self, is_final: bool = False):
        """Override to ensure lock is held during save."""
        # If called from build_graph, lock is already held.
        super()._save_checkpoint(is_final)