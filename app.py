
import sys
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import constants as coon
from settings import sett



embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

qdrant = QdrantClient(
    url=sett.qdrant_url,
    api_key=sett.qdrant_key
)


def embed(texts):
    return [v.tolist() for v in embedder.embed(texts)]


def chunk(text, max_chars: int = 800):
    chunks, current = [], ""

    for para in text.split("\n\n"):
        para = para.strip()

        if not para:
            continue

        if len(current) + len(para) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks


def ingest(folder):
    if not qdrant.collection_exists(coon.COLLECTION_NAME):
        qdrant.create_collection(
            coon.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
        )

    texts, sources = [], []

    for path in Path(folder).glob("**/*"):
        if path.suffix.lower() in (".txt", ".md"):
            for c in chunk(path.read_text(encoding="utf-8")):
                texts.append(c)
                sources.append(path.name)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": t, "source": s},
        )
        for vec, t, s in zip(embed(texts), texts, sources)
    ]

    qdrant.upsert(coon.COLLECTION_NAME, points)


def retrieve(question: str):
    hits = qdrant.query_points(
        coon.COLLECTION_NAME,
        query=embed([question])[0],
        limit=coon.TOP_K
    ).points

    return [h.payload for h in hits]


def ask(question: str) -> str:
    context = "\n\n---\n\n".join(
        f"[{d['source']}]\n{d['text']}"
        for d in retrieve(question)
    )

    model = ChatGoogleGenerativeAI(
        model=sett.model_name,
        api_key=sett.gemini_api_key
    )

    agent = create_agent(
        model=model,
        system_prompt="""
Answer the question using only the provided context.
If the context doesn't contain the answer, say you don't know.
"""
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        }
    )

    return result["messages"][-1].content_blocks


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "ingest" and len(sys.argv) > 2:
        ingest(sys.argv[2])

    elif cmd == "ask" and len(sys.argv) > 2:
        print(ask(" ".join(sys.argv[2:])))

    elif cmd == "chat":
        while (q := input("\nYou: ").strip()) not in {"", "exit", "quit"}:
            print("\nClaude:", ask(q))


if __name__ == "__main__":
    main()
