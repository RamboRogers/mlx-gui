# MLX-GUI API Documentation

MLX-GUI provides a comprehensive REST API for managing and running MLX models on Apple Silicon. The API follows RESTful conventions and includes OpenAI-compatible endpoints for easy integration.

## Base URL

```
http://localhost:8000
```

## Authentication

### API Keys (OpenAI Compatible)

MLX-GUI accepts API keys for OpenAI compatibility but **accepts any key**:

**Bearer Token:**
```bash
curl -H "Authorization: Bearer sk-your-api-key-here" http://localhost:8000/v1/chat/completions
```

**X-API-Key Header:**
```bash
curl -H "x-api-key: sk-your-api-key-here" http://localhost:8000/v1/chat/completions
```

**No Authentication:**
```bash
curl http://localhost:8000/v1/chat/completions  # Also works
```

### HuggingFace Integration

Set HuggingFace tokens via environment variables for enhanced model discovery:
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

---

## Core Endpoints

### Health & Status

#### `GET /`
Root endpoint with basic server information.

**Response:**
```json
{
  "name": "MLX-GUI API",
      "version": "1.2.0",
  "status": "running"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `GET /v1/system/status`
Comprehensive system status including memory usage and loaded models.

**Response:**
```json
{
  "status": "running",
  "system": {
    "platform": "Darwin",
    "architecture": "arm64",
    "processor": "Apple M2 Pro",
    "is_apple_silicon": true,
    "mlx_compatible": true,
    "memory": {
      "total_gb": 32.0,
      "available_gb": 18.5,
      "used_gb": 13.5,
      "percent_used": 42.2
    },
    "gpu_memory": {
      "total_gb": 9.6,
      "available_gb": 9.6,
      "device_name": "Apple M2 Pro"
    }
  },
  "model_manager": {
    "loaded_models_count": 1,
    "max_concurrent_models": 3,
    "queue_size": 0,
    "total_model_memory_gb": 8.5,
    "memory_usage_percent": 26.6,
    "max_tokens_limit": 16384,
    "auto_unload_enabled": true,
    "inactivity_timeout_minutes": 30
  },
  "mlx_compatible": true
}
```

#### `POST /v1/system/shutdown`
Gracefully shutdown the server.

**Response:**
```json
{
  "message": "Server shutting down"
}
```

#### `POST /v1/system/restart`
Restart the server with updated settings.

**Response:**
```json
{
  "message": "Server restarting with updated settings"
}
```

---

## Memory Management

MLX-GUI features advanced memory management with automatic model unloading:

### Three-Layer Memory Protection

1. **Proactive Management**
   - Automatic inactivity-based unloading (configurable timeout, default: 5 minutes)
   - Background cleanup worker (60-second intervals)

2. **Concurrent Limit Management**
   - Hard limit of concurrent models (default: 3, configurable)
   - LRU (Least Recently Used) eviction when limit reached

3. **Emergency Memory Recovery**
   - Automatic MLX memory error detection
   - Up to 3 retry attempts with model unloading between attempts
   - Memory cleanup and cache clearing

### Memory Error Detection

The system automatically detects memory-related errors:
- Error messages: "out of memory", "memory allocation", "insufficient memory", "metal out of memory"
- Error types: `OutOfMemoryError`, `MemoryError`, `RuntimeError` with memory keywords
- Automatic retry with LRU model unloading

### Auto-Unload Behavior

When memory constraints are reached:
1. **Concurrent Limit**: Automatically unloads oldest model when trying to load beyond the limit
2. **Memory Errors**: Detects MLX memory failures and retries with automatic cleanup
3. **Inactivity**: Unloads models after configurable timeout period

---

## Model Management

### List Models

#### `GET /v1/models`
List all models in the database.

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "llama-3.2-3b-instruct",
      "type": "text",
      "status": "loaded",
      "memory_required_gb": 8,
      "use_count": 15,
      "last_used_at": "2025-01-04T10:30:00",
      "created_at": "2025-01-04T09:00:00"
    }
  ]
}
```

### Get Model Details

#### `GET /v1/models/{model_name}`
Get detailed information about a specific model.

**Response:**
```json
{
  "id": 1,
  "name": "llama-3.2-3b-instruct",
  "path": "/path/to/model",
  "version": "1.0",
  "type": "text",
  "status": "loaded",
  "memory_required_gb": 8,
  "use_count": 15,
  "last_used_at": "2025-01-04T10:30:00",
  "created_at": "2025-01-04T09:00:00",
  "updated_at": "2025-01-04T10:30:00",
  "error_message": null,
  "metadata": {}
}
```

### Load Model

#### `POST /v1/models/{model_name}/load`
Load a model into memory. If concurrent model limits are reached, the oldest model will be automatically unloaded to make space.

**Parameters:**
- `priority` (query, optional): Loading priority (default: 0)

**Response:**
```json
{
  "message": "Model 'llama-3.2-3b-instruct' loaded successfully",
  "status": "loaded"
}
```

**Memory Management:**
- Automatically unloads LRU model if concurrent limit (default: 3) is reached
- Retries up to 3 times with memory cleanup if MLX memory errors occur
- Memory warnings are logged but don't prevent loading

**Error Responses:**
- `400`: Model not compatible with system or hardware requirements not met
- `404`: Model not found
- `500`: Loading failed after all retry attempts

### Unload Model

#### `POST /v1/models/{model_name}/unload`
Unload a model from memory.

**Response:**
```json
{
  "message": "Model 'llama-3.2-3b-instruct' unloaded successfully",
  "status": "unloaded"
}
```

### Delete Model

#### `DELETE /v1/models/{model_name}`
Delete a model from the database and remove downloaded files.

**Parameters:**
- `remove_files` (query, optional): Whether to delete downloaded files (default: true)

**Response:**
```json
{
  "message": "Model 'llama-3.2-3b-instruct' deleted successfully",
  "removed_files": true,
  "status": "deleted"
}
```

**Error Responses:**
- `404`: Model not found
- `500`: Deletion failed

### Model Health

#### `GET /v1/models/{model_name}/health`
Check if a model is healthy and ready for inference.

**Response:**
```json
{
  "model": "llama-3.2-3b-instruct",
  "status": "loaded",
  "healthy": true,
  "last_used": "2025-01-04T10:30:00"
}
```

---

## Text Generation

### Generate Text

#### `POST /v1/models/{model_name}/generate`
Generate text using a loaded model.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "repetition_context_size": 20,
  "seed": null
}
```

**Parameters:**
- `prompt` (required): Input text prompt
- `max_tokens` (optional): Maximum tokens to generate (default: 8192, max: 16384)
- `temperature` (optional): Sampling temperature (default: 0.0)
- `top_p` (optional): Nucleus sampling parameter (default: 1.0)
- `top_k` (optional): Top-k sampling parameter (default: 0)
- `repetition_penalty` (optional): Repetition penalty (default: 1.0)
- `repetition_context_size` (optional): Context size for repetition penalty (default: 20)
- `seed` (optional): Random seed for reproducible generation

**Response:**
```json
{
  "model": "llama-3.2-3b-instruct",
  "prompt": "Once upon a time",
  "text": " there was a young princess who lived in a castle...",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 96,
    "total_tokens": 100
  },
  "timing": {
    "generation_time_seconds": 2.5,
    "tokens_per_second": 38.4
  }
}
```

**Error Responses:**
- `400`: Model not loaded or invalid parameters
- `404`: Model not found
- `500`: Generation failed

---

## Model Discovery

### Search Models

#### `GET /v1/discover/models`
Discover MLX-compatible models from HuggingFace.

**Parameters:**
- `query` (optional): Search query
- `limit` (optional): Number of results (default: 20)
- `sort` (optional): Sort by 'downloads', 'likes', 'created', 'updated' (default: 'downloads')

**Response:**
```json
{
  "models": [
    {
      "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "name": "Llama-3.2-3B-Instruct-4bit",
      "author": "mlx-community",
      "downloads": 12543,
      "likes": 89,
      "model_type": "text",
      "size_gb": 2.1,
      "estimated_memory_gb": 2.5,
      "mlx_compatible": true,
      "has_mlx_version": true,
      "mlx_repo_id": null,
      "tags": ["mlx", "llama", "instruct"],
      "description": "Llama 3.2 3B Instruct quantized to 4-bit",
      "updated_at": "2025-01-01T12:00:00"
    }
  ],
  "total": 1
}
```

### Popular Models

#### `GET /v1/discover/popular`
Get popular MLX models by download count.

**Parameters:**
- `limit` (optional): Number of results (default: 20)

**Response:** Same format as `/v1/discover/models`

### Model Categories

#### `GET /v1/discover/categories`
Get categorized model lists.

**Response:**
```json
{
  "categories": {
    "Popular Text Models": [
      "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ],
    "Vision Models": [
      "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"
    ],
    "Multimodal Models": [
      "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    ],
    "Code Models": [
      "mlx-community/CodeLlama-7B-Instruct-4bit"
    ],
    "Chat Models": [
      "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    ],
    "Small Models (< 10GB)": [
      "mlx-community/Phi-3.5-mini-instruct-4bit"
    ],
    "Large Models (> 50GB)": [
      "mlx-community/Llama-3.1-70B-Instruct-4bit"
    ]
  }
}
```

### Embedding Models

#### `GET /v1/discover/embeddings`
Discover MLX-compatible embedding models from HuggingFace.

**Parameters:**
- `query` (optional): Search query for embedding models
- `limit` (optional): Number of results (default: 20)

**Response:**
```json
{
  "models": [
    {
      "id": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
      "name": "Qwen3-Embedding-0.6B-4bit-DWQ",
      "author": "mlx-community",
      "downloads": 1543,
      "likes": 12,
      "model_type": "feature-extraction",
      "size_gb": 0.8,
      "estimated_memory_gb": 1.2,
      "mlx_compatible": true,
      "has_mlx_version": true,
      "tags": ["mlx", "embedding", "feature-extraction"],
      "description": "Qwen3 embedding model quantized to 4-bit for MLX",
      "updated_at": "2025-01-01T12:00:00"
    }
  ],
  "total": 1
}
```

### Compatible Models

#### `GET /v1/discover/compatible`
Find models compatible with current system memory.

**Parameters:**
- `query` (optional): Search query
- `max_memory_gb` (optional): Maximum memory constraint (default: system available memory)

**Response:** Same format as `/v1/discover/models` with additional `memory_fit` field

### Model Details

#### `GET /v1/discover/models/{model_id}`
Get detailed information about a HuggingFace model.

**Response:**
```json
{
  "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "name": "Llama-3.2-3B-Instruct-4bit",
  "author": "mlx-community",
  "downloads": 12543,
  "likes": 89,
  "created_at": "2024-12-01T10:00:00",
  "updated_at": "2025-01-01T12:00:00",
  "model_type": "text",
  "library_name": "transformers",
  "pipeline_tag": "text-generation",
  "tags": ["mlx", "llama", "instruct", "4-bit"],
  "size_gb": 2.1,
  "estimated_memory_gb": 2.5,
  "mlx_compatible": true,
  "has_mlx_version": true,
  "mlx_repo_id": null,
  "description": "Llama 3.2 3B Instruct quantized to 4-bit with MLX",
  "system_compatible": true,
  "compatibility_message": "✅ Compatible - 18.5GB available, 2.5GB required"
}
```

---

## Model Manager

### List Models (Internal Format)

#### `GET /v1/manager/models`
List all models with detailed status information (internal format with status field).

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "llama-3.2-3b-instruct",
      "type": "text",
      "status": "loaded",
      "memory_required_gb": 8,
      "use_count": 15,
      "last_used_at": "2025-01-04T10:30:00",
      "created_at": "2025-01-04T09:00:00",
      "huggingface_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "author": "mlx-community"
    }
  ]
}
```

**Note:** This endpoint includes the `status` field, unlike the OpenAI-compatible `/v1/models` endpoint.

### Manager Status

#### `GET /v1/manager/status`
Get detailed model manager status including loaded models and queue.

**Response:**
```json
{
  "loaded_models": {
    "llama-3.2-3b-instruct": {
      "loaded_at": "2025-01-04T09:15:00",
      "last_used_at": "2025-01-04T10:30:00",
      "memory_usage_gb": 8.5,
      "config": {
        "model_type": "text-generation",
        "estimated_memory_gb": 8.5
      }
    }
  },
  "system_status": {
    "loaded_models_count": 1,
    "max_concurrent_models": 3,
    "queue_size": 0,
    "total_model_memory_gb": 8.5,
    "memory_usage_percent": 26.6,
    "auto_unload_enabled": true,
    "inactivity_timeout_minutes": 30
  },
  "queue_status": []
}
```

### Model Status

#### `GET /v1/manager/models/{model_name}/status`
Get detailed status of a specific model.

**Response:**
```json
{
  "name": "llama-3.2-3b-instruct",
  "status": "loaded",
  "loaded": true,
  "loaded_at": "2025-01-04T09:15:00",
  "last_used_at": "2025-01-04T10:30:00",
  "memory_usage_gb": 8.5,
  "queue_position": null
}
```

### Update Model Priority

#### `POST /v1/manager/models/{model_name}/priority`
Update model loading priority in queue.

**Request Body:**
```json
{
  "priority": 10
}
```

**Response:**
```json
{
  "model": "llama-3.2-3b-instruct",
  "priority": 10,
  "message": "Priority update requested"
}
```

---

## Settings

### Get Settings

#### `GET /v1/settings`
Get all application settings.

**Response:**
```json
{
  "server_port": 8000,
  "max_concurrent_requests": 5,
  "auto_unload_inactive_models": true,
  "model_inactivity_timeout_minutes": 30,
  "enable_system_tray": true,
  "log_level": "INFO",
  "huggingface_cache_dir": "",
  "enable_gpu_acceleration": true,
  "bind_to_all_interfaces": false
}
```

### Update Setting

#### `PUT /v1/settings/{key}`
Update a specific setting.

**Request Body:**
```json
{
  "value": 60
}
```

**Response:**
```json
{
  "key": "model_inactivity_timeout_minutes",
  "value": 60,
  "updated": true
}
```

---

## Audio API

### Audio Transcription

#### `POST /v1/audio/transcriptions`
OpenAI-compatible audio transcription endpoint using MLX-optimized Whisper models for high-performance speech-to-text on Apple Silicon.

**Request (multipart/form-data):**
- `file` (required): Audio file (wav, mp3, m4a, flac, ogg, webm)
- `model` (required): Model to use for transcription
- `language` (optional): Language of the audio (ISO-639-1 format, e.g., 'en', 'es', 'fr')
- `prompt` (optional): Initial prompt to guide transcription
- `response_format` (optional): Format of response (`json`, `text`, `verbose_json`)
- `temperature` (optional): Sampling temperature (0.0 to 1.0)
- `word_timestamps` (optional): Include word-level timestamps (default: false)

**Supported Models:**
| Model Name | MLX Model | Size | Memory | Speed | Accuracy |
|------------|-----------|------|--------|-------|----------|
| `whisper-1` | `mlx-community/whisper-tiny` | ~39MB | Low | Fastest | Good |
| `whisper-tiny` | `mlx-community/whisper-tiny` | ~39MB | Low | Fastest | Good |
| `whisper-base` | `mlx-community/whisper-base` | ~74MB | Low | Fast | Better |
| `whisper-small` | `mlx-community/whisper-small` | ~244MB | Medium | Medium | Good |
| `whisper-medium` | `mlx-community/whisper-medium` | ~769MB | High | Slower | Very Good |
| `whisper-large` | `mlx-community/whisper-large-v3` | ~1.5GB | Very High | Slowest | Best |
| `whisper-large-v2` | `mlx-community/whisper-large-v2` | ~1.5GB | Very High | Slowest | Excellent |
| `whisper-large-v3` | `mlx-community/whisper-large-v3` | ~1.5GB | Very High | Slowest | Best |

**Supported Audio Formats:**
- **WAV** - Uncompressed audio (recommended)
- **MP3** - Compressed audio  
- **MP4/M4A** - MPEG-4 audio
- **FLAC** - Lossless compression
- **OGG** - Ogg Vorbis
- **WebM** - Web audio format

**Basic Example:**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-small"
```

**With Language Specification:**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-medium" \
  -F "language=es" \
  -F "response_format=verbose_json"
```

**Response (JSON format):**
```json
{
  "text": "Hello, this is a test of the audio transcription system."
}
```

**Response (text format):**
```
Hello, this is a test of the audio transcription system.
```

**Response (verbose_json format):**
```json
{
  "text": "Hello, this is a test of the audio transcription system.",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello, this is a test of the audio transcription system."
    }
  ],
  "model": "whisper-small"
}
```

**Python Client Example:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # MLX-GUI accepts any key
    base_url="http://localhost:8000/v1"
)

with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-small",
        file=audio_file,
        language="en"  # Optional
    )

print(transcript.text)
```

**Audio Model Installation:**
Install Whisper models before first use:

```bash
# Install recommended model
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/whisper-small",
    "name": "whisper-small"
  }'

# Install tiny model for fast transcription
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/whisper-tiny", 
    "name": "whisper-tiny"
  }'

# Install large model for best accuracy
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/whisper-large-v3",
    "name": "whisper-large-v3"
  }'
```

**Performance Guidelines:**
- **Quick transcription**: Use `whisper-tiny` or `whisper-base`
- **Balanced performance**: Use `whisper-small` (recommended default)
- **High accuracy**: Use `whisper-medium` or `whisper-large-v3`
- **Resource constrained**: Start with `whisper-tiny`

**Auto-Loading:** Models automatically load when needed and unload based on memory management policies.

**Error Responses:**
- `400`: Invalid audio file, unsupported format, or model not compatible
- `404`: Audio model not found - install it first
- `500`: Transcription failed or model loading error

**Memory Management:** Audio models use GPU memory on Apple Silicon and are subject to the same auto-unloading policies as text models.

### Audio Speech (Text-to-Speech)

#### `POST /v1/audio/speech`
Generate speech from text (endpoint defined, implementation pending).

**Request Body:**
```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Response:** Audio file (implementation pending)

---

## Audio Model Support

### Supported Audio Libraries
MLX-GUI supports multiple audio processing libraries:

- **MLX-Whisper** (`mlx-whisper>=0.12.0`)
  - OpenAI Whisper models optimized for Apple Silicon
  - Automatic speech recognition (ASR)
  - Multiple language support

- **Parakeet-MLX** (`parakeet-mlx`)
  - Advanced speech-to-text models
  - High accuracy transcription
  - Optimized for MLX framework

### Audio Model Types
- **Whisper Models**: `whisper-tiny`, `whisper-small`, `whisper-medium`, `whisper-large`
- **Parakeet Models**: `parakeet-tdt-0.6b-v2` (Time-Delay Transformer)
- **Custom Audio Models**: Any MLX-compatible audio model

### Installation Requirements
Audio support requires additional dependencies:

```bash
# Install audio dependencies
pip install mlx-whisper>=0.12.0
pip install parakeet-mlx

# Or install with audio support
pip install mlx-gui[audio]
```

### Supported Audio Formats
- **WAV** (recommended)
- **MP3**
- **M4A**
- **FLAC**
- **OGG**
- **WEBM**

**Note:** Audio files are automatically converted to the appropriate format for processing.

---

## Multimodal Support

### Vision/Image Models

MLX-GUI now supports vision and multimodal models using MLX-VLM for image understanding. These models can process both text and images in chat conversations.

#### Supported Vision Models

- **Qwen2-VL series**: `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen2-VL-7B-Instruct-4bit`
- **Qwen3-VL series**: `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-8bit`
- **LLaVA models**: Various LLaVA variants optimized for MLX
- **Idefics3**: Multi-modal conversation models
- **Gemma-3 Vision**: `mlx-community/gemma-3n-E4B-it-bf16` and related models

#### Installation

Vision model support requires MLX-VLM:

```bash
# Install with vision support
pip install mlx-gui[vision]

# Or install MLX-VLM separately
pip install mlx-vlm>=0.1.0
```

#### Chat Completions with Images

Send images via the OpenAI-compatible chat completions endpoint using base64 data URLs or image URLs:

**Example with Base64 Image:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-any-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-vl-2b-instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What do you see in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
            }
          }
        ]
      }
    ],
    "max_tokens": 500
  }'
```

**Example with Image URL:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-any-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3n-e4b-it",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in detail."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

**Multiple Images:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-vl-7b-instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Compare these two images and tell me the differences."
          },
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
          },
          {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}
          }
        ]
      }
    ]
  }'
```

#### Response Format

Vision models return the same OpenAI-compatible response format as text models:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1751234567,
  "model": "qwen2-vl-2b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I can see a beautiful landscape with mountains in the background and a lake in the foreground. The sky appears to be partly cloudy with warm lighting suggesting either sunrise or sunset..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 89,
    "completion_tokens": 156,
    "total_tokens": 245
  }
}
```

#### Supported Image Formats

- **JPEG/JPG** (recommended)
- **PNG**
- **GIF** (static)
- **WebP**
- **Base64 encoded images** (data URLs)
- **HTTP/HTTPS image URLs**

#### Model Auto-Detection

MLX-GUI automatically detects vision models based on:

- Model path containing keywords: `vision`, `vlm`, `multimodal`, `llava`, `qwen2-vl`, `qwen3-vl`, `idefics`, `gemma-3`
- Model architecture in `config.json`
- Model type metadata

When images are provided to a non-vision model, they are ignored with a warning.

#### Installation Example

**Install a vision model:**
```bash
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "name": "qwen2-vl-2b-instruct"
  }'
```

#### Limitations

- **Streaming**: Vision models don't support streaming responses yet
- **File uploads**: Only base64 data URLs and HTTP image URLs are supported
- **Video**: Currently only static images are supported, not video files
- **Large images**: Very large images may be automatically resized by MLX-VLM

#### Architecture

MLX-GUI uses MLX-VLM for vision model support:
- **MLX-LM**: For text-only models
- **MLX-VLM**: For vision/multimodal models with image processing
- **Automatic fallback**: If MLX-VLM fails, falls back to MLX-LM text generation

---

## OpenAI Compatibility

### Current Support

MLX-GUI provides **full OpenAI API compatibility** for drop-in replacement:

#### ✅ **Fully Implemented:**
- **`POST /v1/chat/completions`** - Chat completions with streaming support
- **`GET /v1/models`** - List models (OpenAI format)
- **API Key Authentication** - Accepts any API key (Bearer token or x-api-key header)
- **Streaming Responses** - Server-Sent Events with proper chunking
- **OpenAI Response Format** - Exact compatible JSON structure
- **Error Handling** - OpenAI-compatible error responses
- **Token Usage Stats** - Prompt/completion/total token counts
- **Model Installation** - `POST /v1/models/install` for dynamic model loading

#### ✅ **Audio Support:**
- **`POST /v1/audio/transcriptions`** - Audio transcription with Whisper and Parakeet models
- **`POST /v1/audio/speech`** - Text-to-speech (endpoint defined, implementation pending)

#### ❌ **Not Yet Implemented:**
- Function calling
- Embeddings
- Image generation

### Drop-in Replacement

MLX-GUI is now **fully OpenAI-compatible**! Simply:

1. **Replace base URL**: Change from `https://api.openai.com` to `http://localhost:8000`
2. **Install models**: Use `POST /v1/models/install` to add models dynamically
3. **Use as normal**: All existing OpenAI client code works unchanged

### OpenAI Chat Completions

**Example with curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-any-key-works" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-6bit",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1751234567,
  "model": "qwen3-8b-6bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 88,
    "total_tokens": 100
  }
}
```

### Streaming Support

**Enable streaming:**
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-any-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-6bit",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

**Streaming Response:**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1751234567,"model":"qwen3-8b-6bit","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1751234567,"model":"qwen3-8b-6bit","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: [DONE]
```

### Model Installation

**Install any MLX-compatible model:**
```bash
curl -X POST http://localhost:8000/v1/models/install \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "name": "qwen-7b-4bit"
  }'
```

**Response:**
```json
{
  "message": "Model 'qwen-7b-4bit' installed successfully",
  "model_name": "qwen-7b-4bit",
  "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "status": "installed"
}
```

### OpenAI Models List

**Get models in OpenAI format:**
```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-any-key"
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-8b-6bit",
      "object": "model",
      "created": 1751234567,
      "owned_by": "mlx-gui",
      "permission": [],
      "root": "qwen3-8b-6bit",
      "parent": null
    }
  ]
}
```

### OpenAI Embeddings

**Generate text embeddings:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer sk-any-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you today?"],
    "model": "qwen3-embedding-0-6b-4bit",
    "encoding_format": "float"
  }'
```

**Request Parameters:**
- `input` (required): String or array of strings to embed
- `model` (required): Name of the embedding model to use
- `encoding_format` (optional): Format for embeddings (`float` or `base64`, default: `float`)
- `dimensions` (optional): Number of dimensions to truncate embeddings to
- `user` (optional): User identifier for tracking

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1234, -0.5678, 0.9012, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.2345, -0.6789, 0.8901, ...],
      "index": 1
    }
  ],
  "model": "qwen3-embedding-0-6b-4bit",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

**Base64 Encoding:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Text to embed",
    "model": "qwen3-embedding-0-6b-4bit",
    "encoding_format": "base64"
  }'
```

**Supported Embedding Models:**
- Install with: `POST /v1/models/install`
  ```json
  {
    "model_id": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "name": "qwen3-embedding-0-6b-4bit"
  }
  ```

**Features:**
- **Queuing**: Automatic queuing when models are busy
- **Auto-loading**: Models load automatically when needed
- **Batch processing**: Handle multiple texts in one request
- **Usage tracking**: Token counting and metrics
- **Memory management**: Efficient model loading/unloading

---

## Error Handling

### Standard HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters, model not loaded)
- `404`: Not Found (model/endpoint not found)
- `500`: Internal Server Error (system/processing errors)

### Error Response Format

```json
{
  "detail": "Model 'invalid-model' not found"
}
```

### Common Errors

1. **Model Not Found (404)**
   ```json
   {
     "detail": "Model 'nonexistent-model' not found"
   }
   ```

2. **Model Not Loaded (400)**
   ```json
   {
     "detail": "Model 'llama-3.2-3b-instruct' is not loaded. Load it first with POST /v1/models/llama-3.2-3b-instruct/load"
   }
   ```

3. **Insufficient Memory (400)**
   ```json
   {
     "detail": "❌ Insufficient memory - This model requires 32.0GB but you have 18.5GB available"
   }
   ```

4. **MLX Not Compatible (400)**
   ```json
   {
     "detail": "❌ MLX requires Apple Silicon (M1/M2/M3) hardware"
   }
   ```

---

## Rate Limiting

Currently, no rate limiting is implemented. HuggingFace API calls may be rate-limited based on your token tier.

---

## Environment Variables

### Production and Development
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`: HuggingFace token for enhanced model discovery
- `MLX_GUI_PORT`: Server port (default: 8000)
- `MLX_GUI_HOST`: Server host (default: 127.0.0.1)
  - Note: The bundled app auto-detects and configures ffmpeg/ffprobe internally; no ffmpeg-related env vars are required in production builds.

### Development and CI
- `PATH`: For development/tests, put Homebrew first to prefer arm64 tools (e.g., `PATH=/opt/homebrew/bin:$PATH`).
- `FFMPEG_BINARY` and `FFMPEG_PROBE` are optional overrides. The app build and runtime auto-detect ffmpeg/ffprobe; set these only if you need to force a specific binary.

---

## Examples

### Complete Workflow

```bash
# 1. Check system status
curl http://localhost:8000/v1/system/status

# 2. Discover compatible models
curl "http://localhost:8000/v1/discover/compatible?query=llama"

# 3. Load a model
curl -X POST http://localhost:8000/v1/models/llama-3.2-3b-instruct/load

# 4. Check model status
curl http://localhost:8000/v1/models/llama-3.2-3b-instruct/health

# 5. Generate text
curl -X POST http://localhost:8000/v1/models/llama-3.2-3b-instruct/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about Apple Silicon:",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# 6. Unload model when done
curl -X POST http://localhost:8000/v1/models/llama-3.2-3b-instruct/unload

# 7. Delete model and files (default)
curl -X DELETE http://localhost:8000/v1/models/llama-3.2-3b-instruct

# 8. Delete model but keep files (optional)
curl -X DELETE "http://localhost:8000/v1/models/llama-3.2-3b-instruct?remove_files=false"
```

### Python Example

```python
import requests

base_url = "http://localhost:8000"

# Load model
response = requests.post(f"{base_url}/v1/models/llama-3.2-3b-instruct/load")
print(response.json())

# Generate text
response = requests.post(
    f"{base_url}/v1/models/llama-3.2-3b-instruct/generate",
    json={
        "prompt": "Hello, world!",
        "max_tokens": 20,
        "temperature": 0.7
    }
)
result = response.json()
print(f"Generated: {result['text']}")
```

---

## Future OpenAI Compatibility

Planned OpenAI-compatible endpoints:

- `GET /v1/models` - List available models (OpenAI format)
- `POST /v1/chat/completions` - Chat completions endpoint
- `POST /v1/completions` - Text completions (exact OpenAI format)
- Server-Sent Events for streaming
- Function calling support
- Embeddings endpoint

This will enable drop-in replacement for many OpenAI-powered applications.