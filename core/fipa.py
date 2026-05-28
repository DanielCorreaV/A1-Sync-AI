from enum import Enum, auto
from typing import Any, NamedTuple

class FipaPerformative(Enum):
    REQUEST = auto()
    INFORM = auto()
    PROPOSE = auto()
    ACCEPT_PROPOSAL = auto()
    REJECT_PROPOSAL = auto()

class ACLMessage(NamedTuple):
    sender: str
    receiver: str
    performative: FipaPerformative
    content: Any