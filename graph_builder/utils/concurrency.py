import asyncio
from collections import defaultdict

class AsyncLockFactory:
    def __init__(self):
        self._locks = defaultdict(asyncio.Lock)

    def get_lock(self, key: str) -> asyncio.Lock:
        """
        Returns the lock for a given key.
        """
        return self._locks[key]

def get_lock_key(head: str, relation: str) -> str:
    """
    Creates a consistent key for a (head, relation) pair for locking.
    """
    return f"{head}::{relation}"
from collections import defaultdict

class AsyncLockFactory:
    def __init__(self):
        self._locks = defaultdict(asyncio.Lock)

    def get_lock(self, key: str) -> asyncio.Lock:
        """
        Returns the lock for a given key.
        """
        return self._locks[key]

def get_lock_key(head: str, relation: str) -> str:
    """
    Creates a consistent key for a (head, relation) pair for locking.
    """
    return f"{head}::{relation}"
from collections import defaultdict

class AsyncLockFactory:
    def __init__(self):
        self._locks = defaultdict(asyncio.Lock)

    def get_lock(self, key: str) -> asyncio.Lock:
        """
        Returns the lock for a given key.
        """
        return self._locks[key]

def get_lock_key(head: str, relation: str) -> str:
    """
    Creates a consistent key for a (head, relation) pair for locking.
    """
    return f"{head}::{relation}"
