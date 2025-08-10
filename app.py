import os
import io
import json
import uuid
import base64
import hashlib
from datetime import datetime, date, time as dtime, timezone, timedelta
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import boto3
from botocore.exceptions import ClientError, BotoCoreError, ProfileNotFound
from dotenv import load_dotenv

# Helper to read boolean from environment variable strings
def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


# Load .env (if present)
load_dotenv()

APP_TITLE = "Image Upload to S3 with Titan Multimodal Embeddings"
DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
# Prefer explicit env credentials over a profile
ENV_AKI = os.getenv("AWS_ACCESS_KEY_ID")
ENV_SAK = os.getenv("AWS_SECRET_ACCESS_KEY")
ENV_STS = os.getenv("AWS_SESSION_TOKEN")
DEFAULT_PROFILE = None if (ENV_AKI and ENV_SAK) else (os.getenv("AWS_PROFILE") or None)
DEFAULT_MODEL_ID = "amazon.titan-embed-image-v1"
DEFAULT_OUTPUT_EMBEDDING_LENGTH = 1024

# App configuration from environment (no fallbacks - fail fast if missing)
ENV_BUCKET = os.getenv("APP_S3_BUCKET")
ENV_IMAGES_PREFIX = os.getenv("APP_IMAGES_PREFIX", "images/")
ENV_EMBEDDINGS_PREFIX = os.getenv("APP_EMBEDDINGS_PREFIX", "embeddings/")
ENV_VECTOR_BUCKET = os.getenv("S3V_VECTOR_BUCKET")
ENV_INDEX_NAME = os.getenv("S3V_INDEX_NAME")
ENV_INDEX_ARN = os.getenv("S3V_INDEX_ARN")
ENV_MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
ENV_OUTPUT_DIM = int(os.getenv("OUTPUT_EMBEDDING_LENGTH", str(DEFAULT_OUTPUT_EMBEDDING_LENGTH)))

# Use Claude 3.5 Sonnet v2 inference profile ID (required, not optional)
ENV_CLAUDE_VISION_MODEL_ID = os.getenv("CLAUDE_VISION_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")


# Helper for Bedrock inference profile/converse model ID normalization
def _is_inference_profile_identifier(s: str) -> bool:
    """Return True if the string looks like a Bedrock inference profile ID or ARN."""
    if not isinstance(s, str):
        return False
    return s.startswith("us.") or (s.startswith("arn:aws:bedrock:") and ":inference-profile/" in s)


def _resolve_converse_model_id(raw_id: Optional[str]) -> str:
    """Normalize model identifier for Converse.

    For some models (e.g., Anthropic Claude 3.5 Sonnet v2), Bedrock requires an
    **inference profile** identifier (ID or ARN) rather than the foundation model ID
    when using on-demand throughput. This function ensures we pass a valid inference
    profile identifier to `bedrock-runtime.converse`.
    """
    model = (raw_id or ENV_CLAUDE_VISION_MODEL_ID or "").strip()
    if not model:
        raise ValueError(
            "CLAUDE_VISION_MODEL_ID is not set. Set it to an inference profile ID (e.g., 'us.anthropic.claude-3-5-sonnet-20241022-v2:0') or its ARN."
        )
    if _is_inference_profile_identifier(model):
        return model

    # Known mappings: foundation model ID -> inference profile ID
    known_map = {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-haiku-20240307-v1:0": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    }
    if model in known_map:
        return known_map[model]

    # If it's some other foundation model ID, force a clear configuration error.
    raise ValueError(
        (
            "Bedrock Converse requires an inference profile ID/ARN for this model. "
            f"Got foundation model ID '{model}'. Set CLAUDE_VISION_MODEL_ID to an inference profile (e.g., "
            "'us.anthropic.claude-3-5-sonnet-20241022-v2:0') or the corresponding profile ARN from the Bedrock console."
        )
    )


def _get_boto3_session(region: Optional[str] = None, profile: Optional[str] = None):
    """Create a boto3 Session prioritizing environment credentials over profiles.

    Behavior:
    - If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in the environment, use them and ignore profile.
    - Otherwise, if a profile is provided, attempt to use it; if missing, fall back to default chain.
    - Finally, fall back to the default credential chain (env vars, shared config, role).
    """
    # Resolve at call-time to reflect latest environment state
    region = region or (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)

    aki = os.getenv("AWS_ACCESS_KEY_ID")
    sak = os.getenv("AWS_SECRET_ACCESS_KEY")
    sts = os.getenv("AWS_SESSION_TOKEN")

    # If explicit env credentials are present, prefer them and ignore any profile
    if aki and sak:
        return boto3.Session(
            aws_access_key_id=aki,
            aws_secret_access_key=sak,
            aws_session_token=sts,
            region_name=region,
        )

    # If no explicit creds, consider profile (explicit arg wins over env)
    prof = profile if profile is not None else (os.getenv("AWS_PROFILE") or None)
    if prof:
        try:
            return boto3.Session(profile_name=prof, region_name=region)
        except ProfileNotFound:
            print(f"[WARN] AWS profile '{prof}' not found. Falling back to default credentials.")

    # Default chain (env vars, shared config/default, role)
    return boto3.Session(region_name=region)


def get_bedrock_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("bedrock-runtime")


def get_s3_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("s3")


def get_s3vectors_client(region: Optional[str] = None, profile: Optional[str] = None):
    """Return a boto3 client for Amazon S3 Vectors."""
    session = _get_boto3_session(region, profile)
    return session.client("s3vectors")


def generate_image_embedding(image_bytes: bytes, model_id: str, output_dim: int, bedrock_client) -> list:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    body = json.dumps(
        {
            "inputImage": b64_image,
            "embeddingConfig": {"outputEmbeddingLength": output_dim},
        }
    )

    response = bedrock_client.invoke_model(
        body=body, modelId=model_id, accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    warning = response_body.get("message")
    embedding = response_body.get("embedding")
    if isinstance(embedding, list):
        if warning:
            print(f"[Titan warning:image] {warning}")
        return embedding
    if warning:
        raise RuntimeError(f"Model error: {warning}")
    raise RuntimeError("Embedding not found in model response.")



def generate_mm_embedding(
    *,
    bedrock_client,
    model_id: str,
    output_dim: int,
    input_text: Optional[str] = None,
    input_image_bytes: Optional[bytes] = None,
) -> list:
    """Generate an embedding using Titan Multimodal for text and/or image.

    - Caps input text to stay within Titan's ~128-token budget.
    - Treats Titan body `message` as a warning if an embedding is present.
    - Retries once with a tighter cap if Titan refuses due to token limits.
    """
    if not input_text and not input_image_bytes:
        raise ValueError("Either input_text or input_image_bytes must be provided")

    # Keep text within Titan's limit (conservative caps)
    safe_text = None
    if input_text:
        safe_text = _titan_mm_safe_text(input_text, max_words=80)

    def _invoke(text_for_body: Optional[str]):
        body_dict = {"embeddingConfig": {"outputEmbeddingLength": output_dim}}
        if text_for_body:
            body_dict["inputText"] = text_for_body
        if input_image_bytes:
            body_dict["inputImage"] = base64.b64encode(input_image_bytes).decode("utf-8")
        resp = bedrock_client.invoke_model(
            body=json.dumps(body_dict),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        body = json.loads(resp.get("body").read())
        return body

    body = _invoke(safe_text)
    warning = (body or {}).get("message")
    embedding = (body or {}).get("embedding")
    if isinstance(embedding, list):
        if warning:
            print(f"[Titan warning:mm] {warning}")
        return embedding

    # Retry once if token/limit related and we had text
    if warning and ("token" in warning.lower() or "limit" in warning.lower()) and safe_text:
        tighter = _titan_mm_safe_text(safe_text, max_words=60)
        body = _invoke(tighter)
        embedding = (body or {}).get("embedding")
        if isinstance(embedding, list):
            return embedding
        warning = (body or {}).get("message") or warning

    if warning:
        raise RuntimeError(f"Model error: {warning}")
    raise RuntimeError("Embedding not found in model response.")



def _titan_mm_safe_text(text: str, max_words: int = 80) -> str:
    """Clean and truncate text for Titan MM input."""
    if not text:
        return ""
    words = text.replace("\n", " ").split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return " ".join(words)


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _build_embed_text_from_struct(d: dict) -> str:
    # Build a compact, search-friendly string for Titan MM (<= ~60-80 words)
    items = ", ".join((d.get("items") or [])[:8])
    prep = ", ".join((d.get("preparation") or [])[:5])
    tags = ", ".join((d.get("dietary_tags") or [])[:10])
    health = (d.get("perceived_healthiness") or "").strip()
    parts = []
    if items:
        parts.append(f"items: {items}")
    if prep:
        parts.append(f"prep: {prep}")
    if health:
        parts.append(f"health: {health}")
    if tags:
        parts.append(f"tags: {tags}")
    compact = "; ".join(parts)
    return _titan_mm_safe_text(compact, max_words=60) or compact


def generate_image_description(
    image_bytes: bytes,
    meal_data: dict,
    bedrock_client,
    claude_model_id: str = None,
) -> Tuple[str, str]:
    """
    Generate a detailed description of the food image using Claude Vision.
    Combines the meal metadata to create rich, searchable descriptions.
    Returns a tuple: (display_text, embed_text)
    """
    claude_model_id = _resolve_converse_model_id(claude_model_id)
    try:
        prompt_text = (
            "You are a nutrition-conscious vision assistant. Analyze the food image. "
            "Return ONLY compact JSON with these keys: "
            "items (array), preparation (array), sauces_or_sides (array), "
            "perceived_healthiness (one of: 'healthy','moderate','indulgent'), "
            "health_reasons (string, <=25 words), dietary_tags (array; choose from: "
            "['high_protein','whole_grain','fresh_fruit','vegetables','nuts_seeds','dairy',"
            "'fried','refined_carbs','processed_meat','added_syrup','high_sugar','high_salt']), "
            "portion ('small'|'medium'|'large'). No prose, no markdown."
            f"\n\nContext: meal_type={meal_data.get('meal_type','meal')}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt_text},
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                ],
            }
        ]

        response = bedrock_client.converse(
            modelId=claude_model_id,
            messages=messages,
            inferenceConfig={"maxTokens": 160, "temperature": 0.1},
        )

        raw = response["output"]["message"]["content"][0]["text"].strip()
        data = _safe_json_loads(raw)
        if isinstance(data, dict):
            items = ", ".join((data.get("items") or [])[:6])
            health = data.get("perceived_healthiness") or "unknown"
            reason = (data.get("health_reasons") or "").strip()
            display = f"Items: {items}. Health: {health}. {reason}".strip()
            embed_text = _build_embed_text_from_struct(data)
            return display, embed_text
        return raw, _titan_mm_safe_text(raw, max_words=80)
    except Exception:
        # Surface the error to the caller so Streamlit shows it instead of silently falling back.
        raise


def to_unix_ts(dt: datetime) -> int:
    """Convert datetime to Unix timestamp (seconds). Defaults to UTC if naive."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())



def presign_url(s3_client, bucket: str, key: str, expires_in: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        ClientMethod="get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )


# Helper to fetch S3 object bytes directly (for image rendering)

def get_object_bytes(s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


# Helper to fetch S3 object bytes and minimal metadata for debug display

def get_object_bytes_and_meta(s3_client, bucket: str, key: str):
    """Fetch object bytes and minimal metadata for debug display."""
    meta = s3_client.head_object(Bucket=bucket, Key=key)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return data, {
        "content_type": meta.get("ContentType"),
        "content_length": meta.get("ContentLength"),
        "etag": meta.get("ETag"),
        "last_modified": str(meta.get("LastModified")),
    }


# Helper: Delete vector and its embedding JSON (if present)
def delete_orphan_vector_and_artifacts(s3_client, s3v_client, *, index_arn: str = None, vector_bucket: str = None, index_name: str = None, key: str, meta: dict):
    """Best-effort delete of an orphan vector and its embedding JSON.
    Requires s3vectors:DeleteVectors and s3:DeleteObject.
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


def upload_bytes_to_s3(s3_client, bucket: str, key: str, data: bytes, content_type: Optional[str] = None):
    extra = {"ContentType": content_type} if content_type else {}
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, **extra)


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return ""
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def ext_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return ""
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(mime.lower(), "")



def _md5_hex(b: bytes) -> str:
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()

# S3 Vectors allows at most 10 metadata keys per vector. Keep a stable priority.
_S3V_META_PRIORITY = [
    "user_id", "meal_type", "ts",
    "s3_bucket", "s3_image_key", "s3_embedding_key",
    "model_id", "uploaded_filename", "content_type", "meal_time",
]

def _s3v_merge_metadata(searchable: dict, non_searchable: dict, limit: int = 10) -> dict:
    merged = {}
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


def _validate_required_env_vars():
    """Validate that all required environment variables are set."""
    missing_vars = []
    
    # Check required S3 configuration
    if not ENV_BUCKET:
        missing_vars.append("APP_S3_BUCKET")
    
    # Check required S3 Vectors configuration
    if not ENV_VECTOR_BUCKET:
        missing_vars.append("S3V_VECTOR_BUCKET")
    if not ENV_INDEX_NAME:
        missing_vars.append("S3V_INDEX_NAME")
    if not ENV_INDEX_ARN:
        missing_vars.append("S3V_INDEX_ARN")
    
    if missing_vars:
        st.error(
            "❌ **Missing Required Environment Variables**\n\n"
            "The following variables must be set in your `.env` file:\n\n" +
            "\n".join([f"• `{var}`" for var in missing_vars]) +
            "\n\nPlease check your `.env` file and restart the application."
        )
        st.stop()


def main():
    st.set_page_config(page_title="S3 + Titan Embeddings", page_icon="🖼️", layout="centered")
    st.title(APP_TITLE)
    st.caption(
        "Upload an image, generate an embedding with Amazon Bedrock Titan Multimodal, and store both the image and embedding JSON in S3."
    )
    
    # Validate required environment variables first
    _validate_required_env_vars()

    # Load settings from environment (resolved at runtime to pick up .env changes)
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION
    profile = None if (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")) else (os.getenv("AWS_PROFILE") or None)
    bucket = ENV_BUCKET
    images_prefix = normalize_prefix(ENV_IMAGES_PREFIX)
    vectors_prefix = normalize_prefix(ENV_EMBEDDINGS_PREFIX)
    model_id = ENV_MODEL_ID
    output_dim = ENV_OUTPUT_DIM

    # Load S3 Vectors configuration from environment
    vector_bucket = ENV_VECTOR_BUCKET
    index_name = ENV_INDEX_NAME
    index_arn = ENV_INDEX_ARN

    # Main content tabs: Ingest and Search
    ingest_tab, search_tab = st.tabs(["Ingest", "Search"])

    with ingest_tab:
        st.subheader("Upload Food Image")
        
        # File uploader
        uploaded = st.file_uploader(
            "Select a food plate image",
            type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
            key="ingest_uploader",
            help="Upload a single image of your food plate"
        )

        # Metadata inputs
        st.markdown("**Metadata**")
        user_id = st.text_input("User ID (required)", value="", key="ingest_user_id").strip()
        
        col1, col2 = st.columns(2)
        with col1:
            meal_date: date = st.date_input("Meal date", value=datetime.now().date(), key="meal_date")
        with col2:
            meal_time_val: dtime = st.time_input("Meal time", value=datetime.now().time(), key="meal_time")
        
        meal_type = st.selectbox(
            "Meal type",
            ["breakfast", "lunch", "dinner", "snack", "other"],
            index=1,
            key="meal_type",
        )

    # Initialize session state for storing generated data
    if 'generated_description' not in st.session_state:
        st.session_state.generated_description = None
    if 'generated_embedding' not in st.session_state:
        st.session_state.generated_embedding = None
    if 'current_image_bytes' not in st.session_state:
        st.session_state.current_image_bytes = None
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None
    if 'last_image_hash' not in st.session_state:
        st.session_state.last_image_hash = None

    if uploaded is not None:
        # Read bytes once and keep them; also display a preview
        image_bytes = uploaded.read()
        st.session_state.current_image_bytes = image_bytes
        st.session_state.current_image_name = uploaded.name
        try:
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption=f"Preview: {uploaded.name}", use_container_width=True)
        except Exception:
            st.info("Preview unavailable, proceeding with raw bytes.")

        # Reset generated data ONLY if a different image was uploaded
        new_hash = _md5_hex(image_bytes)
        if st.session_state.last_image_hash and st.session_state.last_image_hash != new_hash:
            st.session_state.generated_description = None
            st.session_state.generated_embedding = None
        st.session_state.last_image_hash = new_hash

    with ingest_tab:
        # Step 1: Generate Description and Embedding
        st.markdown("---")
        st.markdown("### Step 1: Generate Description & Embedding")
        
        if st.button("🔍 Generate Description & Embedding", type="primary", key="generate_button", 
                    disabled=(uploaded is None or not user_id)):
            with st.spinner("Processing..."):
                try:
                    bedrock = get_bedrock_client(region, profile)
                    
                    # Prepare meal data for description generation (simplified)
                    meal_data = {
                        'meal_type': meal_type,
                        'tags': [],  # No tags field anymore
                        'protein_grams': 0,  # No protein field anymore
                    }
                    
                    # Generate detailed image description using Claude Vision
                    with st.spinner("Generating image description..."):
                        display_text, embed_text = generate_image_description(
                            image_bytes=st.session_state.current_image_bytes,
                            meal_data=meal_data,
                            bedrock_client=bedrock,
                        )
                        st.session_state.generated_description = display_text

                    # Generate multi-modal embedding (image + compact content tags)
                    with st.spinner("Generating multi-modal embedding..."):
                        embedding = generate_mm_embedding(
                            bedrock_client=bedrock,
                            model_id=model_id,
                            output_dim=output_dim,
                            input_text=embed_text,
                            input_image_bytes=st.session_state.current_image_bytes
                        )
                        st.session_state.generated_embedding = embedding
                    
                    st.success("✅ Description and embedding generated successfully!")
                    
                except (ClientError, BotoCoreError) as aws_err:
                    st.error(f"AWS error: {aws_err}")
                except Exception as e:
                    st.error(f"Failed to generate: {e}")
        
        # Display generated description if available
        if st.session_state.generated_description:
            st.markdown("**Generated Description:**")
            st.info(st.session_state.generated_description)
            
            st.markdown("**Embedding Info:**")
            st.write(f"• Dimension: {len(st.session_state.generated_embedding)}")
            st.write(f"• First 10 values: {st.session_state.generated_embedding[:10]}")
        
        # Step 2: Upload to S3 and Index
        st.markdown("---")
        st.markdown("### Step 2: Upload to S3 & Index")
        
        if st.button("☁️ Upload to S3", type="primary", key="upload_button",
                    disabled=(st.session_state.generated_embedding is None)):
            if not bucket:
                st.error("S3 bucket not configured. Please check your .env file.")
                st.stop()
            if not index_arn and (not vector_bucket or not index_name):
                st.error("Please provide either Index ARN or both Vector Bucket Name and Index Name for S3 Vectors.")
                st.stop()
            # Extra safety: if Streamlit reran and lost state, avoid NoneType errors
            if st.session_state.generated_embedding is None:
                st.error("No embedding available. Please run Step 1: Generate Description & Embedding.")
                st.stop()
            if st.session_state.current_image_bytes is None:
                st.error("Image bytes missing from session. Please re-upload the image and run Step 1 again.")
                st.stop()

            with st.spinner("Uploading to S3 and indexing..."):
                try:
                    s3 = get_s3_client(region, profile)
                    s3v = get_s3vectors_client(region, profile)

                    # Validate index dimension matches embedding length
                    try:
                        if index_arn:
                            idx_resp = s3v.get_index(indexArn=index_arn)
                        else:
                            idx_resp = s3v.get_index(vectorBucketName=vector_bucket, indexName=index_name)
                        index_dim = int(idx_resp["index"].get("dimension"))
                        if index_dim != len(st.session_state.generated_embedding):
                            st.error(
                                f"Index dimension ({index_dim}) does not match embedding length ({len(st.session_state.generated_embedding)}). "
                                f"Please check your .env configuration."
                            )
                            st.stop()
                    except Exception as ie:
                        st.warning(f"Could not validate vector index dimension: {ie}")

                    # Prepare S3 keys
                    image_id = str(uuid.uuid4())
                    content_type = getattr(uploaded, "type", None)
                    ext = ext_from_mime(content_type)
                    if not ext and isinstance(st.session_state.current_image_name, str) and "." in st.session_state.current_image_name:
                        ext = "." + st.session_state.current_image_name.rsplit(".", 1)[-1]

                    image_key = f"{images_prefix}{image_id}{ext}"
                    vector_key = f"{vectors_prefix}{image_id}.json"

                    # Upload image bytes
                    upload_bytes_to_s3(s3, bucket, image_key, st.session_state.current_image_bytes, content_type=content_type)

                    # Build metadata
                    meal_dt = datetime.combine(meal_date, meal_time_val)
                    ts = to_unix_ts(meal_dt)

                    base_record = {
                        "model_id": model_id,
                        "embedding_length": len(st.session_state.generated_embedding),
                        "s3_image_bucket": bucket,
                        "s3_image_key": image_key,
                        "uploaded_filename": st.session_state.current_image_name,
                        "content_type": content_type,
                        "output_embedding_length": output_dim,
                        "region": region,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        # domain metadata
                        "user_id": user_id,
                        "meal_type": meal_type,
                        "meal_time": meal_dt.isoformat(),
                        "ts": ts,
                        "generated_description": st.session_state.generated_description,
                        "embedding": st.session_state.generated_embedding
                    }

                    # Upload complete embedding record JSON to S3
                    upload_bytes_to_s3(
                        s3, bucket, vector_key,
                        json.dumps(base_record).encode("utf-8"),
                        content_type="application/json"
                    )

                    # S3 Vectors: Store minimal metadata
                    searchable_metadata = {
                        "user_id": user_id,
                        "meal_type": meal_type,
                        "ts": ts,
                    }

                    non_searchable_metadata = {
                        "s3_image_key": image_key,
                        "s3_embedding_key": vector_key,
                        "s3_bucket": bucket,
                        "model_id": model_id,
                        "uploaded_filename": st.session_state.current_image_name,
                        "content_type": content_type,
                        "meal_time": meal_dt.isoformat(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    # Insert embedding into S3 Vectors
                    compact_meta = _s3v_merge_metadata(searchable_metadata, non_searchable_metadata, limit=10)
                    put_kwargs = {
                        "vectors": [
                            {
                                "key": image_id,
                                "data": {"float32": [float(x) for x in st.session_state.generated_embedding]},
                                "metadata": compact_meta,
                            }
                        ]
                    }
                    if index_arn:
                        put_kwargs["indexArn"] = index_arn
                    else:
                        put_kwargs["vectorBucketName"] = vector_bucket
                        put_kwargs["indexName"] = index_name

                    s3v.put_vectors(**put_kwargs)

                    st.success("✅ Upload complete!")
                    st.write("**Upload Details:**")
                    st.write(f"• S3 Image Key: `{image_key}`")
                    st.write(f"• S3 Embedding Key: `{vector_key}`")
                    st.write(f"• S3 Vectors Index: `{index_arn or f'{vector_bucket}/{index_name}'}`")
                    st.write(f"• Vector ID: `{image_id}`")

                    # Clear session state after successful upload
                    st.session_state.generated_description = None
                    st.session_state.generated_embedding = None

                except (ClientError, BotoCoreError) as aws_err:
                    st.error(f"AWS error: {aws_err}")
                except Exception as e:
                    st.error(f"Failed to upload: {e}")

    with search_tab:
        st.subheader("Search (User-specific)")
        # Debug mode checkbox (can be pre-enabled by env var)
        debug_mode = st.checkbox("Debug image fetch", value=_env_bool("APP_DEBUG", False))
        # Query input
        query_mode = st.radio("Query type", ["Text", "Image"], horizontal=True, key="query_mode")
        query_text = ""
        query_image_bytes: Optional[bytes] = None
        if query_mode == "Text":
            query_text = st.text_input("Search text", value="", key="search_text").strip()
        else:
            q_up = st.file_uploader(
                "Upload query image",
                type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
                key="search_image_uploader",
            )
            if q_up is not None:
                query_image_bytes = q_up.read()

        st.markdown("**Filters**")
        user_id_f = st.text_input("User ID (required)", value="", key="search_user_id").strip()
        
        # Date range defaults to last 7 days
        today = datetime.now().date()
        default_start = today - timedelta(days=6)
        date_range = st.date_input("Date range", value=(default_start, today), key="date_range")
        
        meal_types = st.multiselect(
            "Meal types",
            options=["breakfast", "lunch", "dinner", "snack", "other"],
            default=[],
            key="meal_types_filter",
        )
        
        top_k = st.number_input("Results to return", min_value=1, max_value=20, value=5, step=1, key="top_k")

        if st.button("Run Search", type="primary", key="run_search"):
            if not user_id_f:
                st.error("Please provide a User ID for user-specific search.")
                st.stop()
            if not index_arn and (not vector_bucket or not index_name):
                st.error("Please provide either Index ARN or both Vector Bucket Name and Index Name for S3 Vectors.")
                st.stop()
            if (query_mode == "Text" and not query_text) and (query_mode == "Image" and not query_image_bytes):
                st.error("Provide a search text or upload a query image.")
                st.stop()

            with st.spinner("Embedding query and querying S3 Vectors..."):
                try:
                    bedrock = get_bedrock_client(region, profile)
                    s3 = get_s3_client(region, profile)
                    s3v = get_s3vectors_client(region, profile)

                    # Get index dimension to ensure matching embedding length
                    if index_arn:
                        idx_resp = s3v.get_index(indexArn=index_arn)
                    else:
                        idx_resp = s3v.get_index(vectorBucketName=vector_bucket, indexName=index_name)
                    index_dim = int(idx_resp["index"].get("dimension"))
                    index_metric = idx_resp["index"].get("distanceMetric")

                    # Create query embedding (text or image) using Titan Multimodal
                    q_embedding = generate_mm_embedding(
                        bedrock_client=bedrock,
                        model_id=model_id,
                        output_dim=index_dim,
                        input_text=query_text if query_mode == "Text" else None,
                        input_image_bytes=query_image_bytes if query_mode == "Image" else None,
                    )

                    # Build filter document
                    conds = []
                    conds.append({"user_id": {"$eq": user_id_f}})

                    # Date range handling
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_d, end_d = date_range
                        if isinstance(start_d, date) and isinstance(end_d, date):
                            start_dt = datetime.combine(start_d, dtime(0, 0, 0))
                            end_dt = datetime.combine(end_d, dtime(23, 59, 59))
                            conds.append({"ts": {"$gte": to_unix_ts(start_dt), "$lte": to_unix_ts(end_dt)}})

                    if meal_types:
                        conds.append({"meal_type": {"$in": meal_types}})

                    if len(conds) == 1:
                        filter_doc = conds[0]
                    else:
                        filter_doc = {"$and": conds}

                    q_kwargs = {
                        "topK": int(top_k),
                        "queryVector": {"float32": [float(x) for x in q_embedding]},
                        "returnMetadata": True,
                        "returnDistance": True,
                        "filter": filter_doc,
                    }
                    if index_arn:
                        q_kwargs["indexArn"] = index_arn
                    else:
                        q_kwargs["vectorBucketName"] = vector_bucket
                        q_kwargs["indexName"] = index_name

                    q_resp = s3v.query_vectors(**q_kwargs)
                    results = q_resp.get("vectors", [])

                    if not results:
                        st.info("No results.")
                    else:
                        st.success(f"Found {len(results)} result(s)")
                        for item in results:
                            key = item.get("key")
                            meta = item.get("metadata", {}) or {}
                            dist = item.get("distance")
                            img_bucket = meta.get("s3_bucket")  # Updated to use optimized metadata
                            img_key = meta.get("s3_image_key")
                            cols = st.columns([1, 2])
                            with cols[0]:
                                if img_bucket and img_key:
                                    if debug_mode:
                                        st.write(f"Attempting image fetch: s3://{img_bucket}/{img_key}")
                                    # Prefer server-side fetch
                                    try:
                                        data, meta_info = get_object_bytes_and_meta(s3, img_bucket, img_key)
                                        if debug_mode:
                                            st.write({
                                                "len_bytes": len(data),
                                                **(meta_info or {})
                                            })
                                        # Try PIL decode first for reliability
                                        try:
                                            img_obj = Image.open(io.BytesIO(data))
                                            if debug_mode:
                                                st.write({"pil_format": img_obj.format, "size": img_obj.size})
                                            st.image(img_obj, caption=f"{key}", use_container_width=True)
                                        except Exception as dec_err:
                                            if debug_mode:
                                                st.warning(f"PIL decode failed: {dec_err}")
                                            st.image(data, caption=f"{key}", use_container_width=True)
                                    except Exception as fetch_err:

                                        if debug_mode:
                                            st.warning(f"Direct S3 fetch failed: {fetch_err}")
                                        # If the object is missing (404/NoSuchKey), offer to clean up the orphan
                                        missing = False
                                        try:
                                            if isinstance(fetch_err, ClientError):
                                                code = fetch_err.response.get("Error", {}).get("Code")
                                                status = fetch_err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                                                missing = (code in {"404", "NoSuchKey", "NotFound"}) or (status == 404)
                                        except Exception:
                                            pass
                                        if missing:
                                            st.warning("Image object not found in S3. This looks like an **orphaned vector** (image deleted after indexing).")
                                            if st.button("🧹 Delete this vector & JSON", key=f"del_{key}"):
                                                try:
                                                    delete_orphan_vector_and_artifacts(
                                                        s3, s3v,
                                                        index_arn=index_arn,
                                                        vector_bucket=vector_bucket,
                                                        index_name=index_name,
                                                        key=key,
                                                        meta=meta,
                                                    )
                                                    st.success("Deleted vector and attempted to remove JSON artifact. Re-run the search.")
                                                except Exception as del_err:
                                                    st.error(f"Cleanup failed: {del_err}")
                                        else:
                                            # Fallback to a presigned URL if not a missing-object case
                                            try:
                                                url = presign_url(s3, img_bucket, img_key, expires_in=3600)
                                                if debug_mode:
                                                    st.write({"presigned_url": url[:80] + "..."})
                                                st.image(url, caption=f"{key}", use_container_width=True)
                                            except Exception as url_err:
                                                if debug_mode:
                                                    st.error(f"Presigned URL display failed: {url_err}")
                                                st.write(f"Image: s3://{img_bucket}/{img_key}")
                                else:
                                    st.write("(no image metadata)")
                            with cols[1]:
                                st.write(f"Key: {key}")
                                sim_text = None
                                if index_metric and index_metric.lower() == "cosine" and isinstance(dist, (int, float)):
                                    sim = 1.0 - float(dist)
                                    sim_text = f"Similarity: {sim:.4f}"
                                if dist is not None:
                                    st.write(f"Distance: {dist:.4f}")
                                if sim_text:
                                    st.write(sim_text)
                                st.json(meta)
                except (ClientError, BotoCoreError) as aws_err:
                    st.error(f"AWS error: {aws_err}")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    with st.expander("Help & Notes"):
        st.markdown(
            """
            **Multi-Modal Image Similarity Search:**
            - This app uses **Claude Vision** to generate detailed textual descriptions of your food images.
            - These descriptions are combined with the images to create **multi-modal embeddings** using Titan Multimodal.
            - This approach significantly improves search accuracy by capturing both visual and semantic content.
            
            **Configuration:**
            - Ensure your AWS credentials are configured in your `.env` file or via environment variables.
            - Amazon Bedrock must be enabled with access to:
              - Titan Multimodal Embeddings model (`amazon.titan-embed-image-v1`)
              - Claude Vision via an **inference profile**. Set `CLAUDE_VISION_MODEL_ID` to the inference profile **ID** (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`) or the **ARN**. The Converse API requires an inference profile for these Anthropic models.
              - To find the profile ID/ARN: Bedrock Console → Inference and assessment → **Cross-Region inference** → select the relevant profile (e.g., *US Anthropic Claude 3.5 Sonnet v2*) and copy its ID/ARN.
            - The S3 bucket and S3 Vectors configuration are loaded from your `.env` file.
            
            **Stored Objects:**
            - Images: `<images_prefix>/<uuid>.<ext>`
            - Embeddings JSON: `<embeddings_prefix>/<uuid>.json` (includes generated descriptions)
            - S3 Vectors: Multi-modal embeddings indexed by image UUID
            
            **Testing:**
            - Run `./tests/run_tests.sh quick` to verify your AWS setup anytime.
            """
        )


if __name__ == "__main__":
    main()
