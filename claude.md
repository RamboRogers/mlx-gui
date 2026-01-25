# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-GUI is a **production-ready** lightweight RESTful wrapper around Apple's MLX engine for dynamically loading and serving MLX-compatible models. The project is a complete inference server with GUI, system tray integration, and comprehensive model management capabilities.

## Current Status

**🎉 Production Release v1.2.4** - This is a mature, feature-complete application with:
- Full OpenAI-compatible API endpoints
- Advanced audio intelligence (Whisper, Parakeet)
- Production-ready embeddings (BGE, MiniLM, Qwen3, Arctic, E5)
- Mistral Small integration with vision capabilities
- Enterprise-grade memory management
- Standalone macOS app bundle

## Key Architecture Components

### 🎯 **Universal AI Capabilities**
- **🧠 MLX Engine Integration** - Native Apple Silicon acceleration via MLX
- **🎙️ Advanced Audio Intelligence** - Complete Whisper & Parakeet support with multi-format processing
- **🔢 Production Embeddings** - Multi-architecture support (BGE, MiniLM, Qwen3, Arctic, E5)
- **🖼️ Vision Models** - Image understanding with Gemma-3n, Qwen2-VL & Qwen3-VL, Mistral Small
- **🤖 Large Language Models** - Full support for instruction-tuned and reasoning models

### 🛠️ **Enterprise-Grade Infrastructure**
- **🔄 Intelligent Memory Management** - Advanced auto-unload system with LRU eviction
- **🛡️ Three-Layer Memory Protection** - Proactive cleanup, concurrent limits, emergency recovery
- **⚡ OpenAI Compatibility** - Drop-in replacement for OpenAI API endpoints
- **🌐 REST API Server** - Complete API for model management and inference
- **📊 Real-Time Monitoring** - System status, memory usage, and model performance

### 🎨 **User Experience**
- **🎨 Beautiful Admin Interface** - Modern web GUI for model management
- **🔍 HuggingFace Integration** - Discover and install MLX-compatible models
- **🍎 macOS System Tray** - Native menu bar integration
- **📱 Standalone App** - Packaged macOS app bundle (no Python required)

## Development Setup

The project is fully implemented with proper packaging:

1. **Python Environment**: Requires Python 3.11+ with MLX dependencies
2. **Package Structure**: Complete Python package with `pyproject.toml`
3. **Dependencies**: All dependencies managed including MLX, FastAPI, SQLite, audio/vision support
4. **CLI Interface**: Full CLI with `mlx-gui start --port 8000` and `mlx-gui tray`

## API Design

The project implements a complete OpenAI-compatible REST API:

### 🎯 **Core AI Services**
- `POST /v1/chat/completions` - OpenAI-compatible chat (text + images + Mistral Small)
- `POST /v1/embeddings` - Multi-architecture embeddings (BGE, MiniLM, Qwen3, Arctic)
- `POST /v1/audio/transcriptions` - Enhanced audio transcription (Whisper Turbo, Parakeet)

### 🛠️ **Model Management**
- `GET /v1/models` - List installed models
- `POST /v1/models/install` - Install from HuggingFace
- `POST /v1/models/{name}/load` - Load model into memory
- `POST /v1/models/{name}/unload` - Unload model from memory
- `DELETE /v1/models/{name}` - Delete model and files
- `GET /v1/models/{name}/health` - Model health checks

### 🔍 **Model Discovery**
- `GET /v1/discover/models` - Search HuggingFace for MLX models
- `GET /v1/discover/embeddings` - Search for embedding models
- `GET /v1/discover/stt` - Search for audio transcription models
- `GET /v1/discover/vision` - Search for vision models
- `GET /v1/discover/popular` - Get popular MLX models
- `GET /v1/discover/trending` - Get trending MLX models

### 📊 **System Operations**
- `GET /v1/system/status` - System and memory status
- `GET /v1/manager/status` - Detailed model manager status
- `GET /admin` - Web admin interface

## Memory Management

**Revolutionary auto-unload system** with:
- **LRU eviction** - Automatically unload least recently used models
- **Memory limits** - Configurable concurrent model limits (default: 3)
- **Proactive cleanup** - Three-layer memory protection
- **Emergency recovery** - Handles memory pressure gracefully

Critical features include checking system RAM before loading models with clear error messages and automatic model eviction when limits are reached.

## Important Considerations

- **Apple Silicon Focus**: Optimized for Apple Silicon with MLX-LM>=0.24.0
- **Multimodal Support**: Complete support for text, audio, image, and embedding inputs
- **HuggingFace Integration**: Full discovery and installation of MLX-compatible models
- **User Experience**: Beautiful GUI with intuitive system tray integration
- **State Management**: SQLite database in standard user application directory
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Model Support

### Text Models
- Qwen3, DeepSeek R1, SmolLM3, Mistral Small, Gemma 3
- All instruction-tuned and reasoning models

### Audio Models
- **Whisper**: Complete MLX-Whisper integration (Turbo, Large v3, all variants)
- **Parakeet**: Ultra-fast speech-to-text with parakeet-tdt-0.6b-v2

### Vision Models
- **Gemma-3n**: Native vision capabilities
- **Qwen2-VL & Qwen3-VL**: Advanced multimodal understanding
- **Mistral Small**: Vision-text capability via MLX-VLM

### Embedding Models
- **BGE**: High-quality embeddings (384 dimensions)
- **MiniLM**: Efficient embeddings (384 dimensions)
- **Qwen3**: Large embeddings (2560 dimensions)
- **Arctic**: Specialized embeddings (1024 dimensions)
- **E5**: Multilingual embeddings

## Development Workflow

When implementing features:
1. The core API endpoints are fully implemented and stable
2. Memory management system is production-ready
3. Database schema is complete with proper migrations
4. GUI components are fully built and functional
5. System tray integration is complete for macOS
6. Comprehensive error messages and user feedback are implemented
7. Full test suite ensures reliability

## Testing

The project includes a comprehensive test suite (`tests/test_unified_mlx.py`) that tests:
- All model types (text, audio, vision, embedding)
- Memory management and auto-unload functionality
- API endpoint compatibility
- Admin interface functionality
- Model discovery and installation

## Deployment

Available as:
1. **Standalone macOS App** - No Python required, drag-and-drop installation
2. **PyPI Package** - `pip install mlx-gui`
3. **Source Installation** - Full development setup

The project is production-ready and actively maintained with regular releases.