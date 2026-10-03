"""
chunker.py
----------
Splits raw document text into overlapping chunks.

WHY CHUNK AT ALL?
Embedding models and LLMs have limited context windows. If you embed
an entire 50-page PDF as one vector, the embedding becomes a "blurry
average" of everything in it, and search quality collapses. Instead,
we split text into small, semantically coherent pieces (e.g. ~500
words) so each chunk represents ONE idea, and search can find the
exact chunk relevant to a question.

WHY OVERLAP?
If we chunk on hard boundaries, a sentence explaining something
important might get split in half between two chunks, losing meaning
in both. A small overlap (e.g. 50 words) means each chunk carries a
bit of the previous chunk's context, so ideas at the boundary aren't
lost.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source: str
    position: int  # order of this chunk within its source document


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Split `text` into overlapping word-based chunks.

    chunk_size: target number of words per chunk
    overlap: number of words repeated between consecutive chunks
    """
    words = text.split()
    chunks: list[Chunk] = []

    if not words:
        return chunks

    start = 0
    position = 0
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append(
            Chunk(
                text=chunk_str,
                chunk_id=f"{source}_{position}",
                source=source,
                position=position,
            )
        )

        position += 1
        start += step

    return chunks


if __name__ == "__main__":
    # Quick manual test: run `python chunker.py` to see it work
    sample = "word " * 1200  # simulate a long document
    result = chunk_text(sample, source="demo.txt")
    print(f"Produced {len(result)} chunks")
    print(f"First chunk id: {result[0].chunk_id}, word count: {len(result[0].text.split())}")
    print(f"Second chunk starts with overlap from the first? "
          f"{result[0].text.split()[-10:] == result[1].text.split()[:10]}")
