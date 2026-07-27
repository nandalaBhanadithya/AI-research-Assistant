from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    CLASSIFYING = "classifying"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchMode(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Maps to the arXiv categories used to build the classifier training set.
DOCUMENT_CATEGORIES: list[str] = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing",
]

ARXIV_CATEGORY_MAP: dict[str, str] = {
    "Artificial Intelligence": "cs.AI",
    "Machine Learning": "cs.LG",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Robotics": "cs.RO",
    "Cyber Security": "cs.CR",
    "Cloud Computing": "cs.DC",
}
