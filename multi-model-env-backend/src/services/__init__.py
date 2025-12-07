"""
Services package
Contains AI model services for GeoNLI evaluation
"""

# Lazy imports to avoid cross-environment dependencies
# Each service will be imported only when needed in its respective Modal function

__all__ = [
    'Florence2CaptionService',
    'Florence2VQAService',
    'GroundingService',
    'get_caption_service',
    'get_vqa_service',
    'get_grounding_service',
]

def __getattr__(name):
    """Lazy load services only when accessed"""
    if name == 'Florence2CaptionService' or name == 'get_caption_service':
        from .florence2_caption_service import Florence2CaptionService, get_caption_service
        return get_caption_service if name == 'get_caption_service' else Florence2CaptionService
    elif name == 'Florence2VQAService' or name == 'get_vqa_service':
        from .florence2_vqa_service import Florence2VQAService, get_vqa_service
        return get_vqa_service if name == 'get_vqa_service' else Florence2VQAService
    elif name == 'GroundingService' or name == 'get_grounding_service':
        from .grounding_service import GroundingService, get_grounding_service
        return get_grounding_service if name == 'get_grounding_service' else GroundingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
