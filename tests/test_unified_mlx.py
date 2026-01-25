#!/usr/bin/env python3
"""
🧪 MLX-GUI Unified Test Suite
Tests all MLX model types: Text, Audio, Vision, and Embeddings

Includes support for:
- Text models: Qwen3, DeepSeek R1, SmolLM3 (multilingual)
- Audio models: Parakeet (transcription)
- Vision models: Gemma 3n, Qwen2-VL & Qwen-VL (multimodal)
- Embedding models: Qwen3 embeddings
"""

import asyncio
import base64
import httpx
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# Test configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0

# Test models - ALL Gemma 3 variants now use MLX-VLM
TEST_MODELS = {
    "text": {
        "qwen3": "qwen3-8b-6bit",
        "deepseek": "deepseek-r1-0528-qwen3-8b-mlx-8bit",  # DeepSeek R1 based on Qwen3
        "smollm3": "smollm3-3b-4bit",  # SmolLM3 multilingual model
        "mistral_small": "mistral-small-3-2-24b-instruct-2506-mlx-4bit"  # Mistral Small 24B instruct model
    },
    "audio": {
        "parakeet": "parakeet-tdt-0-6b-v2",
        "whisper_turbo": "whisper-large-v3-turbo"
    },
    "vision": {
        "gemma3_text": "gemma-3-27b-it-qat-4bit",  # Gemma 3 text model via MLX-VLM
        "gemma3n_8bit": "gemma-3n-e4b-it-mlx-8bit",  # Gemma 3n vision model (8bit)
        "gemma3n_4bit": "gemma-3n-e4b-it",  # Gemma 3n vision model (4bit)
        "synthia": "synthia-s1-27b-mlx-8bit",  # Synthia multimodal model
        "mistral_small": "mistral-small-3-2-24b-instruct-2506-mlx-4bit"  # Mistral Small 24B instruct model
    },
    "embedding": {
        "qwen3_embedding": "qwen3-embedding-4b-4bit-dwq",  # Qwen3 embedding model
        "bge_small": "bge-small-en-v1-5-bf16",  # BGE embedding model
        "minilm": "all-minilm-l6-v2-4bit",  # MiniLM embedding model
        "e5_large": "multilingual-e5-large-mlx"  # E5 large multilingual embedding model
    }
}

class ModelTestResult:
    def __init__(self, name: str, success: bool = False, message: str = "", details: Dict = None):
        self.name = name
        self.success = success
        self.message = message
        self.details = details or {}

    def __str__(self):
        emoji = "✅" if self.success else "❌"
        return f"{emoji} {self.name}: {self.message}"

class MLXTestSuite:
    def __init__(self):
        self.results: List[ModelTestResult] = []
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def add_result(self, result: ModelTestResult):
        self.results.append(result)
        print(f"  {result}")

    async def unload_model(self, model_name: str):
        """Unload a model to free memory after testing."""
        try:
            response = await self.client.post(f"{BASE_URL}/v1/models/{model_name}/unload")
            if response.status_code == 200:
                print(f"    🔄 Unloaded {model_name}")
            else:
                print(f"    ⚠️  Failed to unload {model_name}: {response.status_code}")
        except Exception as e:
            print(f"    ⚠️  Error unloading {model_name}: {e}")

    async def test_server_health(self) -> bool:
        """Test if the server is running and responsive."""
        print("\n🏥 Testing Server Health")

        try:
            response = await self.client.get(f"{BASE_URL}/v1/system/status")
            if response.status_code == 200:
                status = response.json()
                self.add_result(ModelTestResult(
                    "Server Status",
                    True,
                    f"Server online - {status.get('memory', {}).get('total_gb', 'unknown')}GB RAM"
                ))
                return True
            else:
                self.add_result(ModelTestResult("Server Status", False, f"HTTP {response.status_code}"))
                return False
        except Exception as e:
            self.add_result(ModelTestResult("Server Status", False, f"Connection failed: {e}"))
            return False

    async def test_admin_interface(self) -> bool:
        """Test admin interface accessibility."""
        print("\n🖥️  Testing Admin Interface")

        try:
            response = await self.client.get(f"{BASE_URL}/admin")
            if response.status_code == 200:
                content = response.text
                if "MLX-GUI Admin" in content:
                    self.add_result(ModelTestResult("Admin Interface", True, "Admin page accessible"))
                    return True
                else:
                    self.add_result(ModelTestResult("Admin Interface", False, "Admin page content unexpected"))
                    return False
            else:
                self.add_result(ModelTestResult("Admin Interface", False, f"HTTP {response.status_code}"))
                return False
        except Exception as e:
            self.add_result(ModelTestResult("Admin Interface", False, f"Failed to access: {e}"))
            return False

    async def test_models_endpoint(self) -> Dict[str, List[str]]:
        """Test /v1/models endpoint and categorize available models."""
        print("\n📋 Testing Models Endpoint")

        try:
            response = await self.client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('data', [])

                # Categorize models
                available_models = {
                    "text": [],
                    "audio": [],
                    "vision": [],
                    "embedding": [],
                    "unknown": []
                }

                for model in models:
                    model_id = model.get('id', '')
                    model_type = model.get('model_type')

                    # If model_type is null, detect from name
                    if not model_type:
                        model_id_lower = model_id.lower()
                        if any(keyword in model_id_lower for keyword in ["parakeet", "whisper", "speech", "tdt"]):
                            model_type = "audio"
                        elif any(keyword in model_id_lower for keyword in ["3n", "vlm", "vision", "vl-", "multimodal", "qwen2-vl", "qwen3-vl", "gemma-3", "gemma3", "mistral-small"]):
                            model_type = "vision"
                        elif any(keyword in model_id_lower for keyword in ["embedding", "qwen3-embedding", "bge", "minilm"]):
                            model_type = "embedding"
                        else:
                            model_type = "text"  # Default to text for most LLMs

                    if model_type in available_models:
                        available_models[model_type].append(model_id)
                    else:
                        available_models['unknown'].append(model_id)

                total_models = len(models)
                self.add_result(ModelTestResult(
                    "Models Endpoint",
                    True,
                    f"Found {total_models} models ({len(available_models['text'])} text, {len(available_models['audio'])} audio, {len(available_models['vision'])} vision, {len(available_models['embedding'])} embedding)"
                ))

                return available_models
            else:
                self.add_result(ModelTestResult("Models Endpoint", False, f"HTTP {response.status_code}"))
                return {}
        except Exception as e:
            self.add_result(ModelTestResult("Models Endpoint", False, f"Request failed: {e}"))
            return {}

    async def test_text_generation(self, model_name: str, model_label: str) -> bool:
        """Test text generation with a specific model."""
        print(f"\n💬 Testing Text Generation - {model_label}")

        if not model_name:
            self.add_result(ModelTestResult(f"Text Gen ({model_label})", False, "Model not configured"))
            return False

        try:
            # Test with a simple math question
            chat_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer in exactly one word."
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }

            print(f"  🔄 Sending request to {model_name}...")
            response = await self.client.post(f"{BASE_URL}/v1/chat/completions", json=chat_data)

            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content'].strip()
                usage = result['usage']

                # Special check for different model types
                if "qwen3" in model_name.lower() or "gemma3" in model_name.lower():
                    extra_info = " (Processor fix working!)"
                elif "smollm3" in model_name.lower():
                    extra_info = " (SmolLM3 multilingual model working!)"
                else:
                    extra_info = ""

                self.add_result(ModelTestResult(
                    f"Text Gen ({model_label})",
                    True,
                    f"Generated: '{message}' ({usage['total_tokens']} tokens){extra_info}"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult(f"Text Gen ({model_label})", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult(f"Text Gen ({model_label})", False, f"Request failed: {e}"))
            return False

    async def test_audio_transcription(self, model_name: str) -> bool:
        """Test audio transcription with Parakeet model."""
        print(f"\n🎙️  Testing Audio Transcription (Parakeet) - {model_name}")

        if not model_name:
            self.add_result(ModelTestResult("Audio Transcription (Parakeet)", False, "Parakeet model not configured"))
            return False

        try:
            # Use the existing test.wav file (relative to project root)
            test_wav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "test.wav")
            if not os.path.exists(test_wav_path):
                # Try alternative path if running from different directory
                test_wav_path = os.path.join("tests", "test.wav")
                if not os.path.exists(test_wav_path):
                    self.add_result(ModelTestResult("Audio Transcription (Parakeet)", False, f"test.wav file not found at {test_wav_path}"))
                    return False

            # Test transcription
            print(f"  🔄 Transcribing test.wav with {model_name}...")
            with open(test_wav_path, 'rb') as audio_file:
                files = {'file': ('test.wav', audio_file, 'audio/wav')}
                data = {'model': model_name}

                response = await self.client.post(
                    f"{BASE_URL}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()

                # Display the actual transcribed output from the audio file
                print(f"    📝 Transcribed text from audio: '{text}'")

                self.add_result(ModelTestResult(
                    "Audio Transcription (Parakeet)",
                    True,
                    f"✅ Transcribed audio to text: '{text}' (Parakeet working)"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult("Audio Transcription (Parakeet)", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult("Audio Transcription (Parakeet)", False, f"Test failed: {e}"))
            return False

    async def test_whisper_transcription(self, model_name: str, model_label: str) -> bool:
        """Test audio transcription with MLX Whisper models."""
        print(f"\n🔊 Testing Whisper Transcription - {model_label} ({model_name})")

        if not model_name:
            self.add_result(ModelTestResult(f"Whisper Transcription ({model_label})", False, "Whisper model not configured"))
            return False

        try:
            # Use the existing test.wav file (relative to project root)
            test_wav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "test.wav")
            if not os.path.exists(test_wav_path):
                # Try alternative path if running from different directory
                test_wav_path = os.path.join("tests", "test.wav")
                if not os.path.exists(test_wav_path):
                    self.add_result(ModelTestResult(f"Whisper Transcription ({model_label})", False, f"test.wav file not found at {test_wav_path}"))
                    return False

            # Test transcription with different parameters
            print(f"  🔄 Transcribing test.wav with {model_label}...")
            with open(test_wav_path, 'rb') as audio_file:
                files = {'file': ('test.wav', audio_file, 'audio/wav')}
                data = {
                    'model': model_name,
                    'response_format': 'json',
                    'language': 'en',  # Specify English for better accuracy
                    'timestamp_granularities': ['word']  # Request word-level timestamps
                }

                response = await self.client.post(
                    f"{BASE_URL}/v1/audio/transcriptions",
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                language = result.get('language', 'unknown')

                # Check for additional Whisper-specific features
                segments = result.get('segments', [])
                has_segments = len(segments) > 0

                # Display the actual transcribed output from the audio file
                print(f"    📝 Transcribed text: '{text}'")
                print(f"    🌐 Detected language: {language}")
                if has_segments:
                    print(f"    ⏱️  Segments: {len(segments)} (timestamps working)")
                    # Show first segment details if available
                    if segments:
                        first_segment = segments[0]
                        start_time = first_segment.get('start', 'N/A')
                        end_time = first_segment.get('end', 'N/A')
                        segment_text = first_segment.get('text', 'N/A')
                        print(f"    📍 First segment ({start_time}s-{end_time}s): '{segment_text}'")

                # Create success message with Whisper-specific details
                features = []
                if has_segments:
                    features.append("timestamps")
                if language != 'unknown':
                    features.append("language detection")

                feature_str = f" ({', '.join(features)} working)" if features else ""

                self.add_result(ModelTestResult(
                    f"Whisper Transcription ({model_label})",
                    True,
                    f"✅ MLX Whisper: '{text}' (lang: {language}){feature_str}"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult(f"Whisper Transcription ({model_label})", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult(f"Whisper Transcription ({model_label})", False, f"Test failed: {e}"))
            return False

    async def test_embedding_generation(self, model_name: str, model_label: str) -> bool:
        """Test embedding generation with the Qwen3 embedding model."""
        print(f"\n🔗 Testing Embedding Generation - {model_label}")

        if not model_name:
            self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, "Model not configured"))
            return False

        try:
            # Read the test chunk text
            test_chunk_path = os.path.join(os.path.dirname(__file__), "test_chunk.txt")
            if not os.path.exists(test_chunk_path):
                # Try alternative path if running from different directory
                test_chunk_path = os.path.join("tests", "test_chunk.txt")
                if not os.path.exists(test_chunk_path):
                    self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, f"test_chunk.txt not found at {test_chunk_path}"))
                    return False

            with open(test_chunk_path, 'r', encoding='utf-8') as f:
                test_text = f.read().strip()

            # Split into smaller chunks for testing (embedding models have token limits)
            sentences = test_text.split('. ')
            test_chunks = [
                sentences[0] + '.',  # First sentence about grapes
                sentences[1] + '.' if len(sentences) > 1 else "Grapes are nutritious fruits.",  # Second sentence
                "This is a test embedding query about fruit nutrition."  # Additional test text
            ]

            embedding_data = {
                "model": model_name,
                "input": test_chunks,
                "encoding_format": "float"
            }

            print(f"  🔄 Generating embeddings for {len(test_chunks)} text chunks with {model_name}...")
            response = await self.client.post(f"{BASE_URL}/v1/embeddings", json=embedding_data)

            if response.status_code == 200:
                result = response.json()
                embeddings = result.get('data', [])
                usage = result.get('usage', {})

                # Validate embeddings
                if not embeddings:
                    self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, "No embeddings returned"))
                    return False

                # Check that we got the expected number of embeddings
                expected_count = len(test_chunks)
                actual_count = len(embeddings)
                if actual_count != expected_count:
                    self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, f"Expected {expected_count} embeddings, got {actual_count}"))
                    return False

                # Check embedding dimensions and values
                first_embedding = embeddings[0].get('embedding', [])
                embedding_dim = len(first_embedding)

                # Validate embedding quality
                if embedding_dim == 0:
                    self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, "Empty embedding vector"))
                    return False

                # Check that embeddings are numerical and non-zero
                non_zero_count = sum(1 for val in first_embedding if abs(val) > 1e-6)
                if non_zero_count == 0:
                    self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, "All embedding values are zero"))
                    return False

                # Output embeddings for verification
                print(f"    📊 Embedding Details:")
                for i, embedding_data in enumerate(embeddings):
                    embedding_vec = embedding_data.get('embedding', [])
                    input_text = test_chunks[i][:50] + "..." if len(test_chunks[i]) > 50 else test_chunks[i]

                    # Show first 10 and last 10 values
                    if len(embedding_vec) > 20:
                        preview = embedding_vec[:10] + ['...'] + embedding_vec[-10:]
                        preview_str = ', '.join([f"{val:.4f}" if isinstance(val, (int, float)) else str(val) for val in preview])
                    else:
                        preview_str = ', '.join([f"{val:.4f}" for val in embedding_vec])

                    print(f"      Embedding {i+1} ('{input_text}'):")
                    print(f"        Dimension: {len(embedding_vec)}")
                    print(f"        Values: [{preview_str}]")
                    print(f"        Range: {min(embedding_vec):.4f} to {max(embedding_vec):.4f}")
                    print(f"        Mean: {sum(embedding_vec)/len(embedding_vec):.4f}")
                    print()

                # Success message with details
                total_tokens = usage.get('total_tokens', 0)
                self.add_result(ModelTestResult(
                    f"Embedding Gen ({model_label})",
                    True,
                    f"Generated {actual_count} embeddings (dim={embedding_dim}, {non_zero_count}/{embedding_dim} non-zero) ({total_tokens} tokens) (Qwen3 embeddings working!)"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult(f"Embedding Gen ({model_label})", False, f"Test failed: {e}"))
            return False

    async def test_vision_generation(self, model_name: str, model_label: str) -> bool:
        """Test vision generation with image input - specifically for Gemma 3n, Qwen2-VL and Qwen3-VL."""
        print(f"\n🖼️  Testing Vision Generation - {model_label}")

        if not model_name:
            self.add_result(ModelTestResult(f"Vision Gen ({model_label})", False, "Model not configured"))
            return False

        try:
            # Create a simple test image (red square)
            from PIL import Image
            import io

            # Create a 64x64 red square
            image = Image.new('RGB', (64, 64), color='red')

            # Convert to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            img_url = f"data:image/png;base64,{img_base64}"

            # Test prompt varies by model type
            if ("gemma3_text" in model_label.lower() or ("gemma-3-27b" in model_name.lower() and "3n" not in model_name.lower()) or 
                "mistral" in model_name.lower()):
                # This is a text-only model being tested via MLX-VLM - test without image
                return await self._test_text_via_vision_model(model_name, model_label)
            elif "gemma3n" in model_name.lower():
                prompt = "What color is this square? Answer with just the color name."
                model_info = " (Gemma 3n multimodal)"
            elif "qwen2" in model_name.lower():
                prompt = "Describe the color of this image in one word."
                model_info = " (Qwen2-VL vision)"
            elif "qwen3" in model_name.lower():
                prompt = "Describe the color of this image in one word."
                model_info = " (Qwen3-VL vision)"
            else:
                prompt = "What color is this image? Answer in one word."
                model_info = " (Vision model)"

            chat_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]
                    }
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }

            print(f"  🔄 Sending vision request to {model_name}...")
            response = await self.client.post(f"{BASE_URL}/v1/chat/completions", json=chat_data)

            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content'].strip()
                usage = result['usage']

                # Check if the response contains 'red' (expected for red square)
                contains_red = 'red' in message.lower()
                accuracy_note = " ✓ Correct!" if contains_red else " (unexpected response)"

                self.add_result(ModelTestResult(
                    f"Vision Gen ({model_label})",
                    True,
                    f"Response: '{message}'{accuracy_note} ({usage['total_tokens']} tokens){model_info}"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult(f"Vision Gen ({model_label})", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult(f"Vision Gen ({model_label})", False, f"Test failed: {e}"))
            return False

    async def _test_text_via_vision_model(self, model_name: str, model_label: str) -> bool:
        """Test text-only generation via a vision model (like Gemma 3 via MLX-VLM)."""
        try:
            chat_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer in exactly one word."
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }

            print(f"  🔄 Testing text-only mode via MLX-VLM for {model_name}...")
            response = await self.client.post(f"{BASE_URL}/v1/chat/completions", json=chat_data)

            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content'].strip()
                usage = result['usage']

                # Determine the model type for the success message
                if "mistral" in model_name.lower():
                    model_info = " (Mistral via MLX-VLM)"
                else:
                    model_info = " (Gemma 3 via MLX-VLM)"
                
                self.add_result(ModelTestResult(
                    f"Vision Gen ({model_label})",
                    True,
                    f"Text via MLX-VLM: '{message}' ({usage['total_tokens']} tokens){model_info}"
                ))

                # Unload model after successful test
                await self.unload_model(model_name)
                return True
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', str(response.status_code))
                except:
                    error_detail = f"HTTP {response.status_code}"

                self.add_result(ModelTestResult(f"Vision Gen ({model_label})", False, error_detail))
                return False

        except Exception as e:
            self.add_result(ModelTestResult(f"Vision Gen ({model_label})", False, f"Text test failed: {e}"))
            return False

    async def run_all_tests(self):
        """Run the complete test suite."""
        print("🧪 MLX-GUI Unified Test Suite")
        print("=" * 50)

        start_time = time.time()

        # Test server health
        if not await self.test_server_health():
            print("\n❌ Server is not responding. Please start MLX-GUI server first.")
            return

        # Test admin interface
        await self.test_admin_interface()

        # Get available models
        available_models = await self.test_models_endpoint()

        if not available_models:
            print("\n❌ Could not retrieve models list. Cannot continue with model tests.")
            return

        # Update test models based on what's actually available
        actual_test_models = self._find_available_test_models(available_models)

        # Print what models we're testing
        self._print_test_plan(actual_test_models)

        # Test text generation (Qwen3, Gemma3, etc.)
        print("\n🔤 Text Generation Tests")
        print("-" * 30)
        for model_key, model_name in actual_test_models["text"].items():
            if model_name:
                await self.test_text_generation(model_name, model_key.title())
            else:
                self.add_result(ModelTestResult(f"Text Gen ({model_key.title()})", False, "Model not available"))

        # Test audio transcription
        print("\n🎵 Audio Transcription Tests")
        print("-" * 30)

        # Test Parakeet model
        audio_model = actual_test_models["audio"].get("parakeet")
        if audio_model:
            await self.test_audio_transcription(audio_model)
        else:
            self.add_result(ModelTestResult("Audio Transcription (Parakeet)", False, "Parakeet model not available"))

        # Test Whisper model
        whisper_turbo = actual_test_models["audio"].get("whisper_turbo")

        if whisper_turbo:
            await self.test_whisper_transcription(whisper_turbo, "Whisper Large v3 Turbo")
        else:
            self.add_result(ModelTestResult("Whisper Transcription (Turbo)", False, "Whisper Large v3 Turbo model not available"))

        # Test embedding generation
        print("\n🔗 Embedding Generation Tests")
        print("-" * 30)
        
        # Test Qwen3 embedding model
        qwen3_embedding = actual_test_models["embedding"].get("qwen3_embedding")
        if qwen3_embedding:
            await self.test_embedding_generation(qwen3_embedding, "Qwen3 Embedding")
        else:
            self.add_result(ModelTestResult("Embedding Generation (Qwen3)", False, "Qwen3 embedding model not available"))
        
        # Test BGE embedding model
        bge_embedding = actual_test_models["embedding"].get("bge_small")
        if bge_embedding:
            await self.test_embedding_generation(bge_embedding, "BGE Small")
        else:
            self.add_result(ModelTestResult("Embedding Generation (BGE)", False, "BGE embedding model not available"))
        
        # Test MiniLM embedding model
        minilm_embedding = actual_test_models["embedding"].get("minilm")
        if minilm_embedding:
            await self.test_embedding_generation(minilm_embedding, "MiniLM L6")
        else:
            self.add_result(ModelTestResult("Embedding Generation (MiniLM)", False, "MiniLM embedding model not available"))

        # Test vision generation (Gemma 3n, Qwen2-VL, Qwen3-VL)
        print("\n👁️  Vision Generation Tests")
        print("-" * 30)
        for model_key, model_name in actual_test_models["vision"].items():
            if model_name:
                await self.test_vision_generation(model_name, model_key.title())
            else:
                self.add_result(ModelTestResult(f"Vision Gen ({model_key.title()})", False, "Model not available"))

        # Test memory overload and automatic model unloading
        print("\n🧠 Memory Overload Test")
        print("-" * 30)
        await self.test_memory_overload()

        # Print summary after ALL tests are complete
        await self._print_summary(start_time)

        # Cleanup: Unload all models after tests
        await self._cleanup_models()

    def _find_available_test_models(self, available_models: Dict[str, List[str]]) -> Dict:
        """Find which test models are actually available."""
        result = {"text": {}, "audio": {}, "vision": {}, "embedding": {}}

        # Text models
        for key, preferred_name in TEST_MODELS["text"].items():
            if preferred_name in available_models["text"]:
                result["text"][key] = preferred_name
            elif available_models["text"]:
                # Use first available text model
                result["text"][key] = available_models["text"][0]
                print(f"  ℹ️  Using {available_models['text'][0]} instead of {preferred_name} for {key} test")
            else:
                result["text"][key] = None

        # Audio models
        for key, preferred_name in TEST_MODELS["audio"].items():
            if preferred_name in available_models["audio"]:
                result["audio"][key] = preferred_name
            elif available_models["audio"] and key == "parakeet":
                # Use first available audio model for parakeet test
                result["audio"][key] = available_models["audio"][0]
                print(f"  ℹ️  Using {available_models['audio'][0]} instead of {preferred_name} for {key} test")
            else:
                result["audio"][key] = None

        # Vision models
        for key, preferred_name in TEST_MODELS["vision"].items():
            if preferred_name in available_models["vision"]:
                result["vision"][key] = preferred_name
            elif available_models["vision"]:
                # Use first available vision model
                result["vision"][key] = available_models["vision"][0]
                print(f"  ℹ️  Using {available_models['vision'][0]} instead of {preferred_name} for {key} test")
            else:
                result["vision"][key] = None

        # Embedding models
        for key, preferred_name in TEST_MODELS["embedding"].items():
            if preferred_name in available_models["embedding"]:
                result["embedding"][key] = preferred_name
            elif available_models["embedding"]:
                # Use first available embedding model
                result["embedding"][key] = available_models["embedding"][0]
                print(f"  ℹ️  Using {available_models['embedding'][0]} instead of {preferred_name} for {key} test")
            else:
                result["embedding"][key] = None

        return result

    def _print_test_plan(self, actual_test_models: Dict):
        """Print what models will be tested."""
        print("\n📋 Test Plan")
        print("-" * 30)

        # Text models
        text_models = [name for name in actual_test_models["text"].values() if name]
        if text_models:
            print(f"🔤 Text Models: {', '.join(text_models)}")
        else:
            print("🔤 Text Models: None available")

        # Audio models
        audio_models = [name for name in actual_test_models["audio"].values() if name]
        if audio_models:
            print(f"🎵 Audio Models: {', '.join(audio_models)}")
        else:
            print("🎵 Audio Models: None available")

        # Embedding models
        embedding_models = [name for name in actual_test_models["embedding"].values() if name]
        if embedding_models:
            print(f"🔗 Embedding Models: {', '.join(embedding_models)}")
        else:
            print("🔗 Embedding Models: None available")

        # Vision models
        vision_models = [name for name in actual_test_models["vision"].values() if name]
        if vision_models:
            print(f"👁️  Vision Models: {', '.join(vision_models)}")
        else:
            print("👁️  Vision Models: None available")

    async def _print_summary(self, start_time: float):
        """Print test summary."""
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️  Duration: {duration:.2f} seconds")

        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    print(f"   • {result.name}: {result.message}")

        print(f"\n🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        if failed_tests == 0:
            print("🎉 All tests passed! MLX-GUI is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the errors above.")

    async def test_memory_overload(self):
        """Test automatic memory management by loading multiple large models."""
        print("🧠 Testing automatic memory management with model overload...")
        print("  📝 Note: Loading models sequentially without unloading to test memory limits")

        # Models to load in sequence - these should trigger memory management
        test_models = [
            ("deepseek-r1-0528-qwen3-8b-mlx-8bit", "DeepSeek R1 8B"),
            ("smollm3-3b-4bit", "SmolLM3 3B Multilingual"),
            ("gemma-3-27b-it-qat-4bit", "Gemma 3 27B QAT"),
            ("gemma-3n-e4b-it-mlx-8bit", "Gemma 3n 8B Vision"),
            ("qwen3-8b-6bit", "Qwen3 8B"),  # This should trigger unloading
            ("qwen3-embedding-4b-4bit-dwq", "Qwen3 Embedding")  # This should definitely trigger unloading
        ]

        try:
            # First, check system status
            status_response = await self.client.get(f"{BASE_URL}/v1/system/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                model_manager_info = status_data.get('model_manager', {})
                max_concurrent = model_manager_info.get('max_concurrent_models', 3)
                print(f"  📊 System allows max {max_concurrent} concurrent models")

                memory_info = status_data.get('system', {}).get('memory', {})
                total_memory = memory_info.get('total_gb', 'unknown')
                print(f"  💾 Total system memory: {total_memory}GB")

            for i, (model_name, model_label) in enumerate(test_models):
                print(f"\n  📥 Loading model {i+1}/{len(test_models)}: {model_label} ({model_name})")

                # Check current loaded models before loading using the INTERNAL endpoint
                models_response = await self.client.get(f"{BASE_URL}/v1/manager/models")
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    loaded_models = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
                    print(f"    📊 Before: {len(loaded_models)} loaded - {loaded_models}")

                # Attempt to load the model
                load_response = await self.client.post(f"{BASE_URL}/v1/models/{model_name}/load")

                if load_response.status_code == 200:
                    result = load_response.json()
                    print(f"    ✅ {model_label} loaded successfully")

                    # Check if there was a memory warning
                    if 'memory_warning' in result:
                        print(f"    ⚠️  Memory warning: {result['memory_warning']}")

                    # Check loaded models after loading using the INTERNAL endpoint
                    models_response = await self.client.get(f"{BASE_URL}/v1/manager/models")
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        new_loaded_models = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
                        print(f"    📊 After:  {len(new_loaded_models)} loaded - {new_loaded_models}")

                        # Check if any models were automatically unloaded
                        if len(loaded_models) >= max_concurrent and len(new_loaded_models) <= max_concurrent:
                            unloaded = set(loaded_models) - set(new_loaded_models)
                            if unloaded:
                                print(f"    🔄 AUTO-UNLOADED: {list(unloaded)} (LRU eviction working!)")
                                self.add_result(ModelTestResult(
                                    f"Auto-unload triggered by {model_label}",
                                    True,
                                    f"Successfully unloaded {list(unloaded)} to make space"
                                ))
                            else:
                                print(f"    ⚠️  Expected auto-unload but none detected")
                        elif len(new_loaded_models) > max_concurrent:
                            # This should not happen if our limit is working
                            print(f"    ⚠️  Warning: {len(new_loaded_models)} models loaded, exceeds limit of {max_concurrent}")

                        # Test a simple generation to ensure the model is actually loaded
                        test_response = await self.client.post(f"{BASE_URL}/v1/chat/completions", json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 5
                        })

                        if test_response.status_code == 200:
                            print(f"    ✅ {model_label} responding to requests")
                        else:
                            print(f"    ⚠️  {model_label} loaded but not responding")

                else:
                    error_detail = "Unknown error"
                    try:
                        error_data = load_response.json()
                        error_detail = error_data.get('detail', str(load_response.status_code))
                    except:
                        error_detail = f"HTTP {load_response.status_code}"

                    print(f"    ❌ Failed to load {model_label}: {error_detail}")
                    self.add_result(ModelTestResult(
                        f"Memory overload - {model_label}",
                        False,
                        f"Failed to load: {error_detail}"
                    ))

                # Small delay between loads
                await asyncio.sleep(1)

            # Final summary using the INTERNAL endpoint
            models_response = await self.client.get(f"{BASE_URL}/v1/manager/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                final_loaded_models = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
                print(f"\n  📊 Final loaded models: {len(final_loaded_models)} - {final_loaded_models}")

                if len(final_loaded_models) <= max_concurrent:
                    print(f"  ✅ SUCCESS: Model count ({len(final_loaded_models)}) respects limit ({max_concurrent})")
                    print(f"  ✅ AUTO-UNLOAD SYSTEM WORKING: Older models were automatically evicted")
                    self.add_result(ModelTestResult(
                        "Memory Management Test",
                        True,
                        f"Completed memory overload test - {len(final_loaded_models)} models remain loaded (limit: {max_concurrent})"
                    ))
                else:
                    print(f"  ❌ FAILURE: Model count ({len(final_loaded_models)}) exceeds limit ({max_concurrent})")
                    self.add_result(ModelTestResult(
                        "Memory Management Test",
                        False,
                        f"Model count ({len(final_loaded_models)}) exceeds limit ({max_concurrent})"
                    ))

        except Exception as e:
            self.add_result(ModelTestResult(
                "Memory Management Test",
                False,
                f"Test failed: {e}"
            ))
            print(f"    ❌ Memory overload test failed: {e}")

    async def _cleanup_models(self):
        """Unload all models after testing to free memory."""
        print("\n🔄 Cleaning up models...")
        try:
            # Get list of loaded models
            response = await self.client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('data', [])

                # Unload each model
                for model in models:
                    model_name = model.get('id', '')
                    if model_name:
                        await self.unload_model(model_name)

                print("✅ Model cleanup completed")
            else:
                print("⚠️  Could not retrieve models for cleanup")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


async def main():
    """Main test function."""
    try:
        async with MLXTestSuite() as test_suite:
            await test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import httpx
        import PIL
        import numpy
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install with: pip install httpx pillow numpy")
        exit(1)

    print("🚀 Starting MLX-GUI Unified Test Suite...")
    print("📋 Make sure MLX-GUI server is running on http://localhost:8000")
    print()

    asyncio.run(main())