from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Mapping

from mmfood.aws.s3 import normalize_prefix
from mmfood.bedrock.ai import DEFAULT_CLAUDE_VISION_PROFILE


@dataclass
class AppConfig:
    region: str
    profile: Optional[str]

    bucket: str
    images_prefix: str
    embeddings_prefix: str

    model_id: str
    output_dim: int

    vector_bucket: Optional[str]
    index_name: Optional[str]
    index_arn: Optional[str]

    claude_vision_model_id: str

    def missing_required(self) -> List[str]:
        missing: List[str] = []
        if not self.bucket:
            missing.append("APP_S3_BUCKET")

        # For S3 Vectors, require either index_arn OR (vector_bucket AND index_name)
        if not self.index_arn:
            if not self.vector_bucket:
                missing.append("S3V_VECTOR_BUCKET")
            if not self.index_name:
                missing.append("S3V_INDEX_NAME")
        return missing

    def validate(self) -> None:
        missing = self.missing_required()
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing) +
                ". Provide either S3V_INDEX_ARN or both S3V_VECTOR_BUCKET and S3V_INDEX_NAME."
            )


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    env = os.environ if env is None else env

    # Region & profile
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"
    env_aki = env.get("AWS_ACCESS_KEY_ID")
    env_sak = env.get("AWS_SECRET_ACCESS_KEY")
    profile = None if (env_aki and env_sak) else (env.get("AWS_PROFILE") or None)

    # App S3
    bucket = env.get("APP_S3_BUCKET") or ""
    images_prefix = normalize_prefix(env.get("APP_IMAGES_PREFIX", "images/"))
    embeddings_prefix = normalize_prefix(env.get("APP_EMBEDDINGS_PREFIX", "embeddings/"))

    # Models
    model_id = env.get("MODEL_ID", "amazon.titan-embed-image-v1")
    output_dim = int(env.get("OUTPUT_EMBEDDING_LENGTH", "1024"))

    # S3 Vectors
    vector_bucket = env.get("S3V_VECTOR_BUCKET") or None
    index_name = env.get("S3V_INDEX_NAME") or None
    index_arn = env.get("S3V_INDEX_ARN") or None

    # Claude Vision inference profile (ID/ARN)
    claude_model = env.get("CLAUDE_VISION_MODEL_ID", DEFAULT_CLAUDE_VISION_PROFILE)

    return AppConfig(
        region=region,
        profile=profile,
        bucket=bucket,
        images_prefix=images_prefix,
        embeddings_prefix=embeddings_prefix,
        model_id=model_id,
        output_dim=output_dim,
        vector_bucket=vector_bucket,
        index_name=index_name,
        index_arn=index_arn,
        claude_vision_model_id=claude_model,
    )
