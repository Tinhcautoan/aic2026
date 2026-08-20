"""
Pydantic models for the Video Retrieval System API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """What do you want to find?"""
    query: str = Field(..., description="Describe what you're looking for! Like 'a dog running' or 'someone smiling' 🐕", max_length=500)
    top_k: int = Field(default=10, ge=1, le=100, description="How many matching pictures do you want? (1-100)")
    video_filter: Optional[str] = Field(default=None, description="Only look in one specific video? Leave empty to search all")

class SearchResult(BaseModel):
    """One matching picture from a video."""
    id: int = Field(..., description="Unique number for this picture")
    video_id: str = Field(..., description="Which video this picture came from")
    frame_idx: int = Field(..., description="Picture number inside the video")
    timestamp: float = Field(..., description="When in the video this picture appears (in seconds)")
    clip_score: float = Field(..., description="How well this picture matches what you searched (0=not at all, 1=perfect match!)")
    siglip2_score: float = Field(..., description="A second opinion on the match")
    confidence: float = Field(default=0.0, description="0-1 confidence shown in the UI (sigmoid of the reranker logit)")
    final_score: float = Field(..., description="⭐ The FINAL score! Higher = better match!")
    metadata: Optional[dict] = Field(default=None, description="Extra stuff we found in the picture")

class QueryResponse(BaseModel):
    """Here are the matching pictures we found!"""
    query: str = Field(..., description="What you searched for")
    model: str = Field(..., description="Which brain we used")
    task_type: str = Field(..., description="What kind of task this was")
    latency_ms: float = Field(..., description="How fast we found the results (in milliseconds)")
    total_results: int = Field(..., description="How many matching pictures we found")
    results: List[SearchResult] = Field(default_factory=list, description="The matching pictures, best ones first!")

class VideoProcessRequest(BaseModel):
    """🎬 Upload your video here! Just tell us where the file is and we'll do the rest!"""
    video_path: str = Field(
        ..., 
        description="📍 Where is your video? Example: data/videos/my_awesome_video.mp4",
        examples=["data/videos/my_awesome_video.mp4"]
    )

class VideoProcessResponse(BaseModel):
    """✅ Your video is ready! Here's what we did."""
    video_id: str = Field(..., description="A special name we gave your video so we can find it later")
    video_path: str = Field(..., description="Where your video file lives on the computer")
    keyframes: int = Field(..., description="How many important pictures we took from your video 📸")
    embeddings: int = Field(..., description="How many memories we made from these pictures 🧠")
    ocr_enabled: bool = Field(..., description="Did we try to read any words in the video?")
    asr_enabled: bool = Field(..., description="Did we try to understand any spoken words?")
    processing_time: float = Field(..., description="How many seconds it took to watch and learn your video ⏱️")
    status: str = Field(..., description='Did everything go OK? "success" means YES! 🎉')
    error: Optional[str] = Field(default=None, description="If something went wrong, here's what happened")

class BatchProcessRequest(BaseModel):
    """Add lots of videos all at once!"""
    video_paths: List[str] = Field(..., description="A list of all the video files you want to add 📚")
    save_keyframes: bool = Field(default=False, description="Do you want to save the pictures we take? (Takes more space!)")

class BatchProcessResponse(BaseModel):
    """How many videos went through OK?"""
    total_videos: int = Field(..., description="How many videos you sent us")
    successful: int = Field(..., description="How many we learned successfully ✅")
    failed: int = Field(..., description="How many didn't work out ❌")
    results: List[VideoProcessResponse] = Field(default_factory=list, description="Details about each video")

class SystemStatus(BaseModel):
    """How is the system feeling right now?"""
    status: str = Field(..., description="Are we feeling good? 'healthy' = YES! 'degraded' = a bit sick 🤒")
    milvus: dict = Field(..., description="Info about our memory bank")
    redis: dict = Field(..., description="Info about our quick-access notepad")
    gpu_available: bool = Field(..., description="Do we have a super-fast brain booster? (GPU)")
    models_loaded: List[str] = Field(..., description="Which smart brains are ready to go")
    uptime_seconds: float = Field(..., description="How long since we last woke up (in seconds)")

class ErrorResponse(BaseModel):
    """Oops! Something went wrong."""
    error: str = Field(..., description="What went wrong 😢")
    detail: Optional[str] = Field(default=None, description="More details about what happened")


class QaSearchRequest(BaseModel):
    description: str = Field(..., description="Description of the scene to search for", max_length=2000)
    question: str = Field(..., description="Question to answer about the found frames", max_length=1000)
    top_k: int = Field(default=20, ge=1, le=100, description="Number of candidate frames to retrieve")


class QaSearchResultItem(BaseModel):
    video_id: str = ""
    frame_idx: int = 0
    answer: str = ""
    final_score: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    clip_score: float = 0.0
    siglip2_score: float = 0.0


class QaSearchResponse(BaseModel):
    query: str = ""
    question: str = ""
    total_results: int = 0
    latency_ms: float = 0.0
    results: List[QaSearchResultItem] = Field(default_factory=list)


class TrakeRequest(BaseModel):
    description: str = Field(..., description="Main description of the event sequence", max_length=2000)
    events: List[str] = Field(..., description="Ordered list of event moment descriptions", min_length=1)
    top_k: int = Field(default=50, ge=1, le=200, description="Number of candidate videos to search")


class TrakeResultItem(BaseModel):
    video_id: str = ""
    frame_idx: int = 0
    frames: List[int] = Field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    event_frames: List[dict] = Field(default_factory=list)


class TrakeResponse(BaseModel):
    query: str = ""
    total_results: int = 0
    latency_ms: float = 0.0
    results: List[TrakeResultItem] = Field(default_factory=list)


class VQARequest(BaseModel):
    """Single VideoQA question."""
    id: Optional[int] = Field(default=None, description="Row id from test.csv")
    video: Optional[str] = Field(default=None, description="Video filename e.g. clip_001.mp4 (None for global search)")
    question: str = Field(..., description="Question about the video", max_length=1000)


class VQABatchRequest(BaseModel):
    """Batch VideoQA questions."""
    items: List[VQARequest] = Field(default_factory=list)


class VQAResponse(BaseModel):
    id: Optional[int] = None
    video: str
    question: str
    answer: str
    evidence_video_id: Optional[str] = None
    evidence_frame_idx: Optional[int] = None
    evidence_timestamp: Optional[float] = None
    confidence: float = 0.0
    source: str = "metadata"
