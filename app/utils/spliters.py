from enum import Enum
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class SplitStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


def getSpliter(strategy: SplitStrategy, embeddings):
    if strategy == SplitStrategy.FIXED:
        # Example: split into fixed-size chunks
        return CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    elif strategy == "recursive":
        # Example: recursively split by paragraphs/sentences
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    elif strategy == "semantic":
        # Example: split based on semantic meaning (using embeddings, etc.)
        if not embeddings:
            raise ValueError("Embedding is required for semantic spliter")
        else:
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95.0,
            )


# other advance chunkings are
# 1-structure aware Chunking
# 2-parent child Chunking
# 3-late Chunking
