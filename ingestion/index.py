from ingestion.chunker import chunk_pdf
from ingestion.dms import BUCKET, s3
from ingestion.hash_tracker import load_manifest, save_manifest, sha256
from ingestion.vector_store import EMBEDDING_MODEL, count_tokens, embed, get_collection, max_tokens


def list_pdf_keys():
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].lower().endswith(".pdf")]
    return keys


def manifest_entry(digest):
    # The embedding model is part of the entry so switching models re-embeds every document.
    return {"sha256": digest, "embedding_model": EMBEDDING_MODEL}


def is_indexed(collection, key, digest, manifest):
    """Same content and embedding model AND chunks still present (guards against a wiped vector store)."""
    return manifest.get(key) == manifest_entry(digest) and collection.get(where={"source": key}, limit=1)["ids"]


def index_document(collection, key, data, digest):
    chunks = chunk_pdf(data, key, count_tokens, max_tokens())
    for c in chunks:
        c["metadata"]["doc_hash"] = digest
    collection.delete(where={"source": key})
    collection.add(
        ids=[f"{key}::{c['metadata']['chunk_index']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def main():
    manifest = load_manifest()
    collection = get_collection()
    keys = list_pdf_keys()

    for key in keys:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        digest = sha256(data)
        if is_indexed(collection, key, digest, manifest):
            print(f"Unchanged, skipped: {key}")
            continue
        status = "Updated" if key in manifest else "New"
        n = index_document(collection, key, data, digest)
        manifest[key] = manifest_entry(digest)
        save_manifest(manifest)
        print(f"{status}, embedded {n} chunks: {key}")

    for key in set(manifest) - set(keys):
        collection.delete(where={"source": key})
        manifest.pop(key)
        save_manifest(manifest)
        print(f"Removed from S3, deleted chunks: {key}")


if __name__ == "__main__":
    main()
