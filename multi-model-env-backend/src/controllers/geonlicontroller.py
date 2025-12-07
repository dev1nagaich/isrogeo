"""
GeoNLI Evaluation Controller
Handles GeoNLI evaluation API requests

This controller orchestrates the complete evaluation pipeline:
1. Downloads and validates input images
2. Routes queries to appropriate AI services
3. Aggregates and formats results
4. Handles errors gracefully with fallbacks
"""
import modal
from fastapi import HTTPException, status
from src.models.geonlimodel import (
    GeoNLIEvalRequest,
    GeoNLIEvalResponse,
    QueryResponses,
    CaptionResponse,
    GroundingResponse,
    AttributeQueryResponse,
    BinaryAttributeResponse,
    NumericAttributeResponse,
    SemanticAttributeResponse,
    OrientedBoundingBox
)
from src.services import get_caption_service, get_vqa_service, get_grounding_service
from src.utils.image_utils import download_image, save_temp_image, cleanup_temp_image, validate_image_dimensions
import os
import traceback
from typing import Optional
from datetime import datetime
from modal_app import run_caption, run_vqa, run_grounding


# Model paths configuration from environment variables
CAPTION_MODEL_PATH = os.getenv("CAPTION_MODEL_PATH", None)
VQA_MODEL_PATH = os.getenv("VQA_MODEL_PATH", None)
GROUNDING_MODEL_PATH = os.getenv("GROUNDING_MODEL_PATH", None)
GROUNDING_MODEL_BASE = os.getenv("GROUNDING_MODEL_BASE", None)

# Grounding configuration
GROUNDING_CONV_MODE = os.getenv("GROUNDING_CONV_MODE", "llava_v1")
GROUNDING_MAX_BOXES = int(os.getenv("GROUNDING_MAX_BOXES", "10"))

# Performance configuration
MAX_CAPTION_TOKENS = int(os.getenv("MAX_CAPTION_TOKENS", "512"))
MAX_VQA_TOKENS = int(os.getenv("MAX_VQA_TOKENS", "128"))
CAPTION_TEMPERATURE = float(os.getenv("CAPTION_TEMPERATURE", "0.7"))
VQA_TEMPERATURE = float(os.getenv("VQA_TEMPERATURE", "0.7"))
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "30"))



async def evaluate_geonli(eval_request: GeoNLIEvalRequest) -> GeoNLIEvalResponse:
    """
    Process GeoNLI evaluation request
    
    This is the main entry point for GeoNLI evaluation. It orchestrates:
    1. Image acquisition and validation
    2. Query processing across multiple AI services
    3. Result aggregation and formatting
    4. Error handling and cleanup
    
    Args:
        eval_request: GeoNLI evaluation request with image and queries
        
    Returns:
        GeoNLIEvalResponse with all query results
        
    Raises:
        HTTPException: If processing fails critically
    """
    start_time = datetime.now()
    temp_image_path = None
    
    # Log request details
    _log_request_start(eval_request)
    
    try:
        # ===================================================================
        # STEP 1: IMAGE ACQUISITION AND VALIDATION
        # ===================================================================
        print("\n[STEP 1/5] 📥 Downloading and validating image...")
        
        try:
            image = download_image(
                eval_request.input_image.image_url,
                timeout=IMAGE_DOWNLOAD_TIMEOUT
            )
            print(f"  ✅ Image downloaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            error_msg = f"Failed to download image from {eval_request.input_image.image_url}"
            print(f"  ❌ {error_msg}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{error_msg}. Please verify the URL is accessible."
            )
        
        # Validate image dimensions
        expected_width = eval_request.input_image.metadata.width
        expected_height = eval_request.input_image.metadata.height
        actual_width, actual_height = image.size
        
        if not validate_image_dimensions(image, expected_width, expected_height):
            print(f"  ⚠️  Warning: Dimension mismatch!")
            print(f"      Expected: {expected_width}x{expected_height}")
            print(f"      Actual:   {actual_width}x{actual_height}")
            # Note: We proceed with warning, not error
        else:
            print(f"  ✅ Dimensions validated: {actual_width}x{actual_height}")
        
        # Save to temporary file for model processing
        try:
            temp_image_path = save_temp_image(image, eval_request.input_image.image_id)
            print(f"  ✅ Image saved to: {temp_image_path}")
        except Exception as e:
            print(f"  ❌ Failed to save temporary image: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process image file."
            )
        
        # ===================================================================
        # STEP 2: INITIALIZE RESPONSE STRUCTURE
        # ===================================================================
        print("\n[STEP 2/5] 🏗️  Initializing response structure...")
        response_queries = QueryResponses()
        query_count = 0
        
        if eval_request.queries.caption_query:
            query_count += 1
        if eval_request.queries.grounding_query:
            query_count += 1
        if eval_request.queries.attribute_query:
            if eval_request.queries.attribute_query.binary:
                query_count += 1
            if eval_request.queries.attribute_query.numeric:
                query_count += 1
            if eval_request.queries.attribute_query.semantic:
                query_count += 1
        
        print(f"  ✅ Processing {query_count} total queries")
        
        # ===================================================================
        # STEP 3: PROCESS CAPTION QUERY
        # ===================================================================
        if eval_request.queries.caption_query:
            print("\n[STEP 3/5] 📝 Processing caption query...")
            response_queries.caption_query = await _process_caption_query(
                temp_image_path,
                eval_request.queries.caption_query.instruction
            )
            if response_queries.caption_query:
                caption_preview = response_queries.caption_query.response[:100]
                print(f"  ✅ Caption generated: {caption_preview}...")
            else:
                print(f"  ⚠️  Caption generation returned empty result")
        else:
            print("\n[STEP 3/5] ⏭️  Skipping caption query (not requested)")
        
        # ===================================================================
        # STEP 4: PROCESS GROUNDING QUERY
        # ===================================================================
        if eval_request.queries.grounding_query:
            print("\n[STEP 4/5] 🎯 Processing grounding query...")
            response_queries.grounding_query = await _process_grounding_query(
                temp_image_path,
                eval_request.queries.grounding_query.instruction
            )
            if response_queries.grounding_query:
                obj_count = len(response_queries.grounding_query.response)
                print(f"  ✅ Detected {obj_count} objects")
                for idx, bbox in enumerate(response_queries.grounding_query.response[:3], 1):
                    print(f"      Object {bbox.object_id}: {bbox.obbox}")
                if obj_count > 3:
                    print(f"      ... and {obj_count - 3} more")
            else:
                print(f"  ⚠️  Grounding returned no objects")
        else:
            print("\n[STEP 4/5] ⏭️  Skipping grounding query (not requested)")
        
        # ===================================================================
        # STEP 5: PROCESS ATTRIBUTE QUERIES
        # ===================================================================
        if eval_request.queries.attribute_query:
            print("\n[STEP 5/5] ❓ Processing attribute queries...")
            response_queries.attribute_query = await _process_attribute_queries(
                temp_image_path,
                eval_request.queries.attribute_query
            )
            
            if response_queries.attribute_query:
                if response_queries.attribute_query.binary:
                    print(f"  ✅ Binary: {response_queries.attribute_query.binary.response}")
                if response_queries.attribute_query.numeric:
                    print(f"  ✅ Numeric: {response_queries.attribute_query.numeric.response}")
                if response_queries.attribute_query.semantic:
                    semantic_preview = response_queries.attribute_query.semantic.response[:50]
                    print(f"  ✅ Semantic: {semantic_preview}...")
        else:
            print("\n[STEP 5/5] ⏭️  Skipping attribute queries (not requested)")
        
        # ===================================================================
        # BUILD FINAL RESPONSE
        # ===================================================================
        print("\n[FINAL] 🎁 Building response...")
        response = GeoNLIEvalResponse(
            input_image=eval_request.input_image,
            queries=response_queries
        )
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"  Image ID: {eval_request.input_image.image_id}")
        print(f"  Queries processed: {query_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print("="*70 + "\n")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (already formatted)
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GeoNLI evaluation failed: {str(e)}"
        )
    
    finally:
        # ===================================================================
        # CLEANUP
        # ===================================================================
        if temp_image_path:
            try:
                cleanup_temp_image(temp_image_path)
                print(f"🧹 Cleaned up temporary image: {temp_image_path}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to cleanup temp image: {e}")


def _log_request_start(eval_request: GeoNLIEvalRequest):
    """Log the start of a GeoNLI evaluation request"""
    print("\n" + "="*70)
    print("🚀 GEONLI EVALUATION REQUEST")
    print("="*70)
    print(f"  Image ID:    {eval_request.input_image.image_id}")
    print(f"  Image URL:   {eval_request.input_image.image_url}")
    print(f"  Dimensions:  {eval_request.input_image.metadata.width}x{eval_request.input_image.metadata.height}")
    print(f"  Resolution:  {eval_request.input_image.metadata.spatial_resolution_m}m")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    queries = []
    if eval_request.queries.caption_query:
        queries.append("Caption")
    if eval_request.queries.grounding_query:
        queries.append("Grounding")
    if eval_request.queries.attribute_query:
        attr_queries = []
        if eval_request.queries.attribute_query.binary:
            attr_queries.append("Binary")
        if eval_request.queries.attribute_query.numeric:
            attr_queries.append("Numeric")
        if eval_request.queries.attribute_query.semantic:
            attr_queries.append("Semantic")
        if attr_queries:
            queries.append(f"Attribute ({', '.join(attr_queries)})")
    
    print(f"  Queries:     {', '.join(queries) if queries else 'None'}")
    print("="*70)


async def _process_caption_query(image_path: str, instruction: str) -> CaptionResponse:
    try:
        print("  📝 Calling Modal Caption GPU service...")

        # Read image as bytes for Modal GPU function
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Call modal GPU service
        result = await run_caption.remote.aio(image_bytes, MAX_CAPTION_TOKENS, CAPTION_TEMPERATURE)

        caption = result.get("caption", "")

        if not caption or len(caption.strip()) == 0:
            caption = "Unable to generate a detailed caption for this image."

        return CaptionResponse(
            instruction=instruction,
            response=caption
        )

    except Exception as e:
        print(f"  ❌ Caption GPU error: {e}")
        traceback.print_exc()
        return CaptionResponse(
            instruction=instruction,
            response="Caption failed due to processing error."
        )



async def _process_grounding_query(image_path: str, instruction: str) -> GroundingResponse:
    try:
        print("  🎯 Calling Modal Grounding GPU service...")

        # Read image as bytes for Modal GPU function
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        result = await run_grounding.remote.aio(image_bytes, instruction, GROUNDING_MAX_BOXES)
        detections = result.get("detections", [])

        bboxes = [
            OrientedBoundingBox(
                object_id=str(obj["object_id"]),
                obbox=obj["obbox"]
            )
            for obj in detections
        ]

        return GroundingResponse(
            instruction=instruction,
            response=bboxes
        )

    except Exception as e:
        print(f"  ❌ Grounding GPU error: {e}")
        traceback.print_exc()
        return GroundingResponse(instruction=instruction, response=[])



async def _process_attribute_queries(image_path: str, attribute_query) -> AttributeQueryResponse:
    try:
        print("  ❓ Calling Modal VQA GPU service...")

        questions = []
        mapping = {}

        if attribute_query.binary:
            mapping['binary'] = attribute_query.binary.instruction
            questions.append(attribute_query.binary.instruction)

        if attribute_query.numeric:
            mapping['numeric'] = attribute_query.numeric.instruction
            questions.append(attribute_query.numeric.instruction)

        if attribute_query.semantic:
            mapping['semantic'] = attribute_query.semantic.instruction
            questions.append(attribute_query.semantic.instruction)

        # Read image as bytes for Modal GPU function
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        result = await run_vqa.remote.aio(image_bytes, "\n".join(questions), MAX_VQA_TOKENS, VQA_TEMPERATURE)
        answer = result.get("answer", "")

        response = AttributeQueryResponse()

        idx = 0
        if 'binary' in mapping:
            response.binary = BinaryAttributeResponse(
                instruction=mapping['binary'],
                response="Yes" if "yes" in answer.lower() else "No"
            )
            idx += 1

        if 'numeric' in mapping:
            num = 0.0
            try:
                num = float(''.join(filter(str.isdigit, answer)) or 0)
            except:
                pass
            response.numeric = NumericAttributeResponse(
                instruction=mapping['numeric'],
                response=num
            )
            idx += 1

        if 'semantic' in mapping:
            response.semantic = SemanticAttributeResponse(
                instruction=mapping['semantic'],
                response=answer.strip()
            )

        return response

    except Exception as e:
        print(f"  ❌ VQA GPU error: {e}")
        traceback.print_exc()
        return AttributeQueryResponse()
