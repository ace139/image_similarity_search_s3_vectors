from __future__ import annotations

from typing import Dict

# S3 Vectors allows at most 10 metadata keys per vector. Keep a stable priority.
_S3V_META_PRIORITY = [
    "user_id", "meal_type", "ts",
    "s3_bucket", "s3_image_key", "s3_embedding_key",
    "model_id", "uploaded_filename", "content_type", "meal_time",
]


def merge_metadata(searchable: Dict, non_searchable: Dict, limit: int = 10) -> Dict:
    """Merge searchable and non-searchable metadata honoring a fixed priority order.

    S3 Vectors limits metadata to at most 10 keys per vector. This function
    ensures a consistent, stable selection of keys with priority given to fields
    that are important for filtering and display.
    """
    searchable = searchable or {}
    non_searchable = non_searchable or {}

    merged: Dict = {}
    # apply priority order first
    for k in _S3V_META_PRIORITY:
        if k in searchable:
            merged[k] = searchable[k]
        if k in non_searchable and k not in merged:
            merged[k] = non_searchable[k]
        if len(merged) >= limit:
            return merged
    # fill remaining slots with any leftover keys (stable order)
    for src in (searchable, non_searchable):
        for k, v in src.items():
            if k not in merged:
                merged[k] = v
                if len(merged) >= limit:
                    return merged
    return merged


def delete_orphan_vector_and_artifacts(
    s3_client,
    s3v_client,
    *,
    index_arn: str | None = None,
    vector_bucket: str | None = None,
    index_name: str | None = None,
    key: str,
    meta: dict,
) -> None:
    """Best-effort delete of an orphan vector and its embedding JSON.

    Requires s3vectors:DeleteVectors (on the index) and s3:DeleteObject (on the
    embedding JSON object). This function is idempotent and logs errors to stdout.
    """
    # Delete vector from S3 Vectors
    del_kwargs = {"keys": [key]}
    if index_arn:
        del_kwargs["indexArn"] = index_arn
    else:
        del_kwargs["vectorBucketName"] = vector_bucket
        del_kwargs["indexName"] = index_name
    try:
        s3v_client.delete_vectors(**del_kwargs)
    except Exception as e:
        print(f"[cleanup] delete_vectors failed for {key}: {e}")

    # Delete embedding JSON if we have it
    emb_bucket = meta.get("s3_bucket")
    emb_key = meta.get("s3_embedding_key")
    if emb_bucket and emb_key:
        try:
            s3_client.delete_object(Bucket=emb_bucket, Key=emb_key)
        except Exception as e:
            print(f"[cleanup] delete_object failed for s3://{emb_bucket}/{emb_key}: {e}")
