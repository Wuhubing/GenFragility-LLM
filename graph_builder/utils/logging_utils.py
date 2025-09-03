import logging
import json
from typing import Optional

from graph_builder.schema.json_models import CandidateTriple

def setup_logger():
    # A more robust implementation would configure this from a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("builder.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GraphBuilder")

logger = setup_logger()

def log_reject(tri: CandidateTriple, reason: str, score: Optional[float] = None):
    """
    Logs the rejection of a triple in a structured format.
    """
    log_data = {
        "event": "triple_rejected",
        "reason": reason,
        "triple": tri.dict(),
        "score": score
    }
    logger.warning(json.dumps(log_data))

def log_accept(tri: CandidateTriple, score: float):
    """
    Logs the acceptance of a triple in a structured format.
    """
    log_data = {
        "event": "triple_accepted",
        "triple": tri.dict(),
        "score": score
    }
    logger.info(json.dumps(log_data))
import json
from typing import Optional

from graph_builder.schema.json_models import CandidateTriple

def setup_logger():
    # A more robust implementation would configure this from a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("builder.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GraphBuilder")

logger = setup_logger()

def log_reject(tri: CandidateTriple, reason: str, score: Optional[float] = None):
    """
    Logs the rejection of a triple in a structured format.
    """
    log_data = {
        "event": "triple_rejected",
        "reason": reason,
        "triple": tri.dict(),
        "score": score
    }
    logger.warning(json.dumps(log_data))

def log_accept(tri: CandidateTriple, score: float):
    """
    Logs the acceptance of a triple in a structured format.
    """
    log_data = {
        "event": "triple_accepted",
        "triple": tri.dict(),
        "score": score
    }
    logger.info(json.dumps(log_data))
import json
from typing import Optional

from graph_builder.schema.json_models import CandidateTriple

def setup_logger():
    # A more robust implementation would configure this from a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("builder.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GraphBuilder")

logger = setup_logger()

def log_reject(tri: CandidateTriple, reason: str, score: Optional[float] = None):
    """
    Logs the rejection of a triple in a structured format.
    """
    log_data = {
        "event": "triple_rejected",
        "reason": reason,
        "triple": tri.dict(),
        "score": score
    }
    logger.warning(json.dumps(log_data))

def log_accept(tri: CandidateTriple, score: float):
    """
    Logs the acceptance of a triple in a structured format.
    """
    log_data = {
        "event": "triple_accepted",
        "triple": tri.dict(),
        "score": score
    }
    logger.info(json.dumps(log_data))
