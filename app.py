import os
import io
import json
import uuid
from datetime import datetime, date, time as dtime, timezone, timedelta
from typing import Optional

import streamlit as st
from PIL import Image
from botocore.exceptions import ClientError, BotoCoreError
from dotenv import load_dotenv
from mmfood.aws.s3 import (
    presign_url,
    get_object_bytes_and_meta,
    upload_bytes_to_s3,
    ext_from_mime,
)
from mmfood.utils.time import to_unix_ts
from mmfood.utils.crypto import md5_hex
from mmfood.aws.session import (
    get_bedrock_client,
    get_s3_client,
    get_s3vectors_client,
)
from mmfood.bedrock.ai import (
    generate_mm_embedding,
    generate_image_description,
)
from mmfood.s3vectors.utils import (
    merge_metadata as _s3v_merge_metadata,
    delete_orphan_vector_and_artifacts,
)
from mmfood.config import load_config, AppConfig

# Helper to read boolean from environment variable strings
def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


# Load .env (if present)
load_dotenv()

APP_TITLE = "Multi-Modal Food Image Search with AWS AI Stack"


# Bedrock helpers moved to mmfood.bedrock.ai


# AWS client helpers now imported from mmfood.aws.session


# generate_image_embedding moved to mmfood.bedrock.ai (not used directly)




# generate_mm_embedding now imported from mmfood.bedrock.ai



# _titan_mm_safe_text moved to mmfood.bedrock.ai


# _safe_json_loads moved to mmfood.bedrock.ai


# _build_embed_text_from_struct moved to mmfood.bedrock.ai


## generate_image_description moved to mmfood.bedrock.ai

# to_unix_ts moved to mmfood.utils.time


# Helper: Delete vector and its embedding JSON (if present)
# S3 Vectors helpers moved to mmfood.s3vectors.utils
 


def _validate_required_env_vars(cfg: AppConfig):
    """Validate that all required environment variables are set.

    Requires either `S3V_INDEX_ARN` OR both `S3V_VECTOR_BUCKET` and `S3V_INDEX_NAME`.
    """
    missing_vars = cfg.missing_required()
    if missing_vars:
        details = (
            "Provide either `S3V_INDEX_ARN` or both `S3V_VECTOR_BUCKET` and `S3V_INDEX_NAME`."
        )
        st.error(
            "❌ **Missing Required Environment Variables**\n\n"
            "The following variables must be set in your `.env` file:\n\n" +
            "\n".join([f"• `{var}`" for var in missing_vars]) +
            f"\n\n{details}\n\nPlease check your `.env` file and restart the application."
        )
        st.stop()


def main():
    st.set_page_config(page_title="Multi-Modal Food Search", page_icon="🍽️", layout="centered")
    st.title(APP_TITLE)
    st.caption(
        "🔍 **Search food images using text or images** • Powered by AWS S3 Vectors, Bedrock Titan Multi-Modal Embeddings, and Claude 3.5 Sonnet for AI-generated image descriptions"
    )
    
    # Load centralized configuration and validate
    cfg = load_config()
    _validate_required_env_vars(cfg)

    # Settings
    region = cfg.region
    profile = cfg.profile
    bucket = cfg.bucket
    images_prefix = cfg.images_prefix
    vectors_prefix = cfg.embeddings_prefix
    model_id = cfg.model_id
    output_dim = cfg.output_dim

    # S3 Vectors
    vector_bucket = cfg.vector_bucket
    index_name = cfg.index_name
    index_arn = cfg.index_arn

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
        
        # Show preview immediately below the uploader (Ingest tab only)
        if uploaded is not None:
            # Read bytes once and keep them; also display a preview
            image_bytes = uploaded.read()
            st.session_state.current_image_bytes = image_bytes
            st.session_state.current_image_name = uploaded.name
            st.session_state.ingest_preview_bytes = image_bytes
            try:
                img = Image.open(io.BytesIO(image_bytes))
                st.image(img, caption=f"Preview: {uploaded.name}", use_container_width=True)
            except Exception:
                st.info("Preview unavailable, proceeding with raw bytes.")
            
            # Reset generated data ONLY if a different image was uploaded
            new_hash = md5_hex(image_bytes)
            if st.session_state.last_image_hash and st.session_state.last_image_hash != new_hash:
                st.session_state.generated_description = None
                st.session_state.generated_embedding = None
            st.session_state.last_image_hash = new_hash

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
    if 'ingest_preview_bytes' not in st.session_state:
        st.session_state.ingest_preview_bytes = None

    # Preview handled within ingest tab immediately after uploader

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
                            claude_model_id=cfg.claude_vision_model_id,
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
                    st.session_state.current_image_bytes = None
                    st.session_state.current_image_name = None
                    st.session_state.last_image_hash = None
                    st.session_state.ingest_preview_bytes = None
                    # Note: Do not attempt to clear the file_uploader via session_state; Streamlit forbids modifying
                    # a widget's value programmatically after instantiation. We rely on preview-state cleanup instead.

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
                # No preview in Search tab by design

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
