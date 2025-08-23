from __future__ import annotations

import os
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionInfo
from qdrant_client.http.exceptions import UnexpectedResponse


def get_qdrant_client(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 60
) -> QdrantClient:
    """Create and return a Qdrant client."""
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        https=url.startswith('https://'),
        prefer_grpc=False  # Use HTTP for better compatibility
    )


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE
) -> bool:
    """Ensure the collection exists with proper configuration.
    
    Returns True if collection was created, False if it already existed.
    """
    try:
        collection_info = client.get_collection(collection_name)
        # Validate vector size matches
        if collection_info.config.params.vectors.size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size "
                f"{collection_info.config.params.vectors.size}, but expected {vector_size}"
            )
        return False  # Collection already exists
    except UnexpectedResponse as e:
        if e.status_code == 404:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            return True  # Collection was created
        raise  # Re-raise other errors


def validate_collection_config(
    client: QdrantClient,
    collection_name: str,
    expected_vector_size: int
) -> CollectionInfo:
    """Validate that the collection exists and has the correct configuration."""
    try:
        collection_info = client.get_collection(collection_name)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Please create it first or use ensure_collection_exists()."
            )
        raise

    actual_size = collection_info.config.params.vectors.size
    if actual_size != expected_vector_size:
        raise ValueError(
            f"Collection '{collection_name}' has vector size {actual_size}, "
            f"but expected {expected_vector_size}"
        )
    
    return collection_info