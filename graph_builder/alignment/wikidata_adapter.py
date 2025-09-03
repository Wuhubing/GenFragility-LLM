import aiohttp
from typing import List, Dict, Any, Optional

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Note: In a real implementation, we would need robust QID/PID lookup for entities and relations.
# For now, these functions assume QIDs and PIDs are passed in directly.

class WikidataAdapter:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def _execute_query(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.session.get(
                WIKIDATA_SPARQL_URL,
                params={'query': query, 'format': 'json'}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Wikidata query failed: {e}")
            return None

    async def validate_exact(self, head_qid: str, pid: str, tail_qid: str) -> bool:
        """
        Checks if the triple (head_qid, pid, tail_qid) exists exactly in Wikidata.
        """
        query = f"""
        SELECT ?o WHERE {{
          wd:{head_qid} wdt:{pid} ?o .
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return False
        
        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def validate_union(self, head_qid: str, pid_list: List[str], tail_qid: str) -> bool:
        """
        Checks if a connection exists between head and tail using any of the properties in pid_list.
        """
        union_clauses = " UNION ".join([f"{{ wd:{head_qid} wdt:{pid} ?o . }}" for pid in pid_list])
        query = f"SELECT ?o WHERE {{ {union_clauses} }}"
        
        results = await self._execute_query(query)
        if not results:
            return False

        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def fetch_temporal(self, head_qid: str, pid: str) -> List[Dict[str, Any]]:
        """
        Fetches temporal data for a given head and property, including start and end times.
        """
        query = f"""
        SELECT ?obj ?startTime ?endTime WHERE {{
          wd:{head_qid} p:{pid} ?statement .
          ?statement ps:{pid} ?obj .
          OPTIONAL {{ ?statement pq:P580 ?startTime . }}
          OPTIONAL {{ ?statement pq:P582 ?endTime . }}
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return []

        temporal_data = []
        for binding in results["results"]["bindings"]:
            temporal_data.append({
                "object_qid": binding.get("obj", {}).get("value", "").split("/")[-1],
                "start_time": binding.get("startTime", {}).get("value"),
                "end_time": binding.get("endTime", {}).get("value"),
            })
        return temporal_data

# Example Usage:
# async def main():
#     async with aiohttp.ClientSession() as session:
#         adapter = WikidataAdapter(session)
#         # Check if Google (Q95) is an instance of (P31) business (Q4830453)
#         is_valid = await adapter.validate_exact("Q95", "P31", "Q4830453")
#         print(f"Is Google a business? {is_valid}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
from typing import List, Dict, Any, Optional

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Note: In a real implementation, we would need robust QID/PID lookup for entities and relations.
# For now, these functions assume QIDs and PIDs are passed in directly.

class WikidataAdapter:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def _execute_query(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.session.get(
                WIKIDATA_SPARQL_URL,
                params={'query': query, 'format': 'json'}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Wikidata query failed: {e}")
            return None

    async def validate_exact(self, head_qid: str, pid: str, tail_qid: str) -> bool:
        """
        Checks if the triple (head_qid, pid, tail_qid) exists exactly in Wikidata.
        """
        query = f"""
        SELECT ?o WHERE {{
          wd:{head_qid} wdt:{pid} ?o .
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return False
        
        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def validate_union(self, head_qid: str, pid_list: List[str], tail_qid: str) -> bool:
        """
        Checks if a connection exists between head and tail using any of the properties in pid_list.
        """
        union_clauses = " UNION ".join([f"{{ wd:{head_qid} wdt:{pid} ?o . }}" for pid in pid_list])
        query = f"SELECT ?o WHERE {{ {union_clauses} }}"
        
        results = await self._execute_query(query)
        if not results:
            return False

        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def fetch_temporal(self, head_qid: str, pid: str) -> List[Dict[str, Any]]:
        """
        Fetches temporal data for a given head and property, including start and end times.
        """
        query = f"""
        SELECT ?obj ?startTime ?endTime WHERE {{
          wd:{head_qid} p:{pid} ?statement .
          ?statement ps:{pid} ?obj .
          OPTIONAL {{ ?statement pq:P580 ?startTime . }}
          OPTIONAL {{ ?statement pq:P582 ?endTime . }}
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return []

        temporal_data = []
        for binding in results["results"]["bindings"]:
            temporal_data.append({
                "object_qid": binding.get("obj", {}).get("value", "").split("/")[-1],
                "start_time": binding.get("startTime", {}).get("value"),
                "end_time": binding.get("endTime", {}).get("value"),
            })
        return temporal_data

# Example Usage:
# async def main():
#     async with aiohttp.ClientSession() as session:
#         adapter = WikidataAdapter(session)
#         # Check if Google (Q95) is an instance of (P31) business (Q4830453)
#         is_valid = await adapter.validate_exact("Q95", "P31", "Q4830453")
#         print(f"Is Google a business? {is_valid}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
from typing import List, Dict, Any, Optional

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Note: In a real implementation, we would need robust QID/PID lookup for entities and relations.
# For now, these functions assume QIDs and PIDs are passed in directly.

class WikidataAdapter:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def _execute_query(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            async with self.session.get(
                WIKIDATA_SPARQL_URL,
                params={'query': query, 'format': 'json'}
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Wikidata query failed: {e}")
            return None

    async def validate_exact(self, head_qid: str, pid: str, tail_qid: str) -> bool:
        """
        Checks if the triple (head_qid, pid, tail_qid) exists exactly in Wikidata.
        """
        query = f"""
        SELECT ?o WHERE {{
          wd:{head_qid} wdt:{pid} ?o .
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return False
        
        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def validate_union(self, head_qid: str, pid_list: List[str], tail_qid: str) -> bool:
        """
        Checks if a connection exists between head and tail using any of the properties in pid_list.
        """
        union_clauses = " UNION ".join([f"{{ wd:{head_qid} wdt:{pid} ?o . }}" for pid in pid_list])
        query = f"SELECT ?o WHERE {{ {union_clauses} }}"
        
        results = await self._execute_query(query)
        if not results:
            return False

        expected_tail = f"http://www.wikidata.org/entity/{tail_qid}"
        for binding in results["results"]["bindings"]:
            if binding.get("o", {}).get("value") == expected_tail:
                return True
        return False

    async def fetch_temporal(self, head_qid: str, pid: str) -> List[Dict[str, Any]]:
        """
        Fetches temporal data for a given head and property, including start and end times.
        """
        query = f"""
        SELECT ?obj ?startTime ?endTime WHERE {{
          wd:{head_qid} p:{pid} ?statement .
          ?statement ps:{pid} ?obj .
          OPTIONAL {{ ?statement pq:P580 ?startTime . }}
          OPTIONAL {{ ?statement pq:P582 ?endTime . }}
        }}
        """
        results = await self._execute_query(query)
        if not results:
            return []

        temporal_data = []
        for binding in results["results"]["bindings"]:
            temporal_data.append({
                "object_qid": binding.get("obj", {}).get("value", "").split("/")[-1],
                "start_time": binding.get("startTime", {}).get("value"),
                "end_time": binding.get("endTime", {}).get("value"),
            })
        return temporal_data

# Example Usage:
# async def main():
#     async with aiohttp.ClientSession() as session:
#         adapter = WikidataAdapter(session)
#         # Check if Google (Q95) is an instance of (P31) business (Q4830453)
#         is_valid = await adapter.validate_exact("Q95", "P31", "Q4830453")
#         print(f"Is Google a business? {is_valid}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
