"""
MLX-LM integration layer for MLX-GUI.
Handles actual model loading, tokenization, and inference using MLX-LM.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import load as load_utils
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from huggingface_hub import snapshot_download
import numpy as np

# Audio model imports (optional)
try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

try:
    import mlx_audio
    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False

# Embedding model imports (optional)
try:
    from mlx_embedding_models import EmbeddingModel
    HAS_MLX_EMBEDDING_MODELS = True
except ImportError:
    HAS_MLX_EMBEDDING_MODELS = False

# MLX embeddings library (for MLX community embedding models)
try:
    from mlx_embeddings import load as mlx_embeddings_load, generate as mlx_embeddings_generate
    HAS_MLX_EMBEDDINGS = True
except ImportError:
    HAS_MLX_EMBEDDINGS = False

try:
    import parakeet_mlx
    HAS_PARAKEET_MLX = True
except ImportError:
    HAS_PARAKEET_MLX = False

# Vision/multimodal model imports (optional)
try:
    import mlx_vlm
    HAS_MLX_VLM = True
except Exception as e:  # broader than ImportError; transformers/metadata issues surface here
    HAS_MLX_VLM = False
    # Defer heavy logging until logger is configured; store a bootstrap message
    try:
        import logging as _bootstrap_logging
        _bootstrap_logging.getLogger(__name__).warning(
            "mlx-vlm unavailable: %s. Vision/multimodal models will be disabled.", str(e)
        )
    except Exception:
        pass

from mlx_gui.huggingface_integration import get_huggingface_client

logger = logging.getLogger(__name__)

# Log library status after logger is defined
if not HAS_MLX_WHISPER:
    logger.warning("mlx-whisper not installed - Whisper models not supported")
if not HAS_MLX_AUDIO:
    logger.warning("mlx-audio not installed - Audio models not supported")
if not HAS_PARAKEET_MLX:
    logger.warning("parakeet-mlx not installed - Parakeet models not supported")
if not HAS_MLX_VLM:
    logger.warning("mlx-vlm not installed - Vision/multimodal models not supported")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    prompt: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    generation_time_seconds: float
    tokens_per_second: float


class MLXModelWrapper:
    """Base wrapper for MLX models with unified interface."""

    def __init__(self, model, tokenizer, model_path: str, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.model_type = config.get("model_type", "text-generation")

        # Only apply tokenizer fixes if tokenizer is not None (audio models don't have tokenizers)
        if tokenizer is not None:
            # Workaround for processors (Gemma3Processor, Qwen3Processor, etc.) missing methods
            if not hasattr(tokenizer, 'eos_token_id'):
                # Try to get from wrapped tokenizer first
                if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'eos_token_id'):
                    tokenizer.eos_token_id = tokenizer.tokenizer.eos_token_id
                    logger.debug(f"Set eos_token_id from wrapped tokenizer: {tokenizer.eos_token_id}")
                # Try to get from model config
                elif hasattr(tokenizer, 'vocab_size') and 'eos_token_id' in config:
                    tokenizer.eos_token_id = config['eos_token_id']
                    logger.debug(f"Set eos_token_id from model config: {tokenizer.eos_token_id}")
                # Check for common attribute names
                elif hasattr(tokenizer, 'eos_id'):
                    tokenizer.eos_token_id = tokenizer.eos_id
                    logger.debug(f"Set eos_token_id from eos_id: {tokenizer.eos_token_id}")
                else:
                    # Fallback: use common default
                    tokenizer.eos_token_id = 2
                    logger.warning(f"Using fallback eos_token_id: {tokenizer.eos_token_id}")

            # Comprehensive workaround for processors missing common tokenizer methods/attributes
            if hasattr(tokenizer, 'tokenizer'):
                inner_tokenizer = tokenizer.tokenizer

                # Add missing methods
                for method_name in ['encode', 'decode', 'apply_chat_template']:
                    if not hasattr(tokenizer, method_name) and hasattr(inner_tokenizer, method_name):
                        setattr(tokenizer, method_name, getattr(inner_tokenizer, method_name))
                        logger.debug(f"Set {method_name} method from wrapped tokenizer")

                # Add missing attributes
                for attr_name in ['vocab', 'vocab_size', 'bos_token_id', 'pad_token_id', 'bos_token', 'eos_token', 'pad_token']:
                    if not hasattr(tokenizer, attr_name) and hasattr(inner_tokenizer, attr_name):
                        setattr(tokenizer, attr_name, getattr(inner_tokenizer, attr_name))
                        logger.debug(f"Set {attr_name} attribute from wrapped tokenizer")

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text synchronously."""
        import time
        start_time = time.time()

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Create sampler with temperature and top_p
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )

        # Create logits processors for repetition penalty
        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size
        )

        # Generate with MLX-LM
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Debug logging
        logger.debug(f"Prompt: {repr(prompt)}")
        logger.debug(f"Full response: {repr(response)}")
        logger.debug(f"Response length: {len(response)}, Prompt length: {len(prompt)}")

        # Extract generated text (remove prompt)
        if response.startswith(prompt):
            generated_text = response[len(prompt):]
        else:
            # MLX-LM might return only the generated text
            generated_text = response
            logger.debug("Response doesn't start with prompt, using full response as generated text")

        # Count tokens
        completion_tokens = len(self.tokenizer.encode(generated_text))
        total_tokens = len(prompt_tokens) + completion_tokens
        tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0

        return GenerationResult(
            text=generated_text,
            prompt=prompt,
            total_tokens=total_tokens,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=completion_tokens,
            generation_time_seconds=generation_time,
            tokens_per_second=tokens_per_second
        )

    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        import asyncio

        # Create sampler and logits processors
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )

        logits_processors = make_logits_processors(
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size
        )

        # Use MLX-LM stream_generate
        for response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors
        ):
            yield response.text
            # Allow other coroutines to run
            await asyncio.sleep(0)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # This is a basic implementation using the model's forward pass
        # For proper embedding models, this should be overridden
        import time

        start_time = time.time()
        embeddings = []

        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Convert to MLX array and add batch dimension
            input_ids = mx.array([tokens])

            # Get model outputs (hidden states)
            # For embedding models, we typically want the last hidden state
            # This is a simplified approach - real embedding models may need different handling
            try:
                # Forward pass through the model
                outputs = self.model(input_ids)

                # For embedding models, we typically take the mean of the last hidden states
                # or use the [CLS] token representation
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    # Some models return tuple of (logits, hidden_states)
                    hidden_states = outputs[0]
                else:
                    # If model returns logits directly, we need to get hidden states differently
                    # This is a fallback that may not work for all models
                    hidden_states = outputs

                # For embedding models, we want to pool across the sequence length (axis=1)
                # to get a fixed-size representation per text
                if hidden_states.ndim == 3:  # [batch, seq_len, hidden_dim]
                    # Mean pooling across sequence length (dim=1) 
                    # This gives us [batch, hidden_dim]
                    embedding = mx.mean(hidden_states, axis=1).squeeze()
                elif hidden_states.ndim == 2:  # [seq_len, hidden_dim] - single batch
                    # Mean pooling across sequence length (dim=0)
                    # This gives us [hidden_dim]
                    embedding = mx.mean(hidden_states, axis=0)
                else:
                    # Fallback for unexpected shapes
                    embedding = hidden_states.squeeze()

                # Ensure we have a 1D array and convert to list
                if embedding.ndim == 0:
                    embedding = mx.expand_dims(embedding, axis=0)
                elif embedding.ndim > 1:
                    # If still multi-dimensional, flatten to 1D
                    embedding = embedding.flatten()

                embedding_list = embedding.tolist()
                embeddings.append(embedding_list)

            except Exception as e:
                logger.error(f"Error generating embedding for text: {e}")
                # Return a zero embedding as fallback
                embeddings.append([0.0] * 768)  # Default embedding size

        end_time = time.time()
        logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")

        return embeddings


class MLXUniversalEmbeddingWrapper(MLXModelWrapper):
    """Universal wrapper for various embedding model architectures."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Load model and tokenizer
        try:
            model, tokenizer = load(model_path)
            super().__init__(model, tokenizer, model_path, config)
            self.model_type = "embedding"
            self.architecture = self._detect_architecture(model_path, model)
            logger.info(f"Loaded {self.architecture} embedding model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load universal embedding model: {e}")
            # Re-raise the exception so it can fall back to other loading methods
            raise RuntimeError(f"Universal embedding wrapper failed to load {model_path}: {e}") from e
    
    def _detect_architecture(self, model_path: str, model) -> str:
        """Detect the model architecture for proper embedding extraction."""
        path_lower = model_path.lower()
        
        # Check by model name patterns
        if "qwen" in path_lower and "embedding" in path_lower:
            return "qwen"
        elif "gte" in path_lower and "qwen" in path_lower:
            return "gte_qwen"
        elif "modernbert" in path_lower:
            return "modernbert"
        elif "arctic" in path_lower:
            return "xlm-roberta"  # Arctic models use xlm-roberta architecture
        elif "e5" in path_lower:
            return "e5"
        elif "minilm" in path_lower:
            return "minilm"
        elif "bge" in path_lower:
            return "bge"
        
        # Check by model class name
        model_class = type(model).__name__.lower()
        if "qwen" in model_class:
            return "qwen"
        elif "xlmroberta" in model_class or "xlm_roberta" in model_class:
            return "xlm-roberta"
        elif "bert" in model_class:
            return "bert"
        
        # Default fallback
        return "generic"
    
    def _get_hidden_states(self, input_ids) -> mx.array:
        """Extract hidden states based on model architecture."""
        if self.architecture in ["qwen", "gte_qwen"]:
            # For Qwen-based models, extract hidden states manually
            x = self.model.model.embed_tokens(input_ids)
            for layer in self.model.model.layers:
                x = layer(x)
            if hasattr(self.model.model, 'norm'):
                x = self.model.model.norm(x)
            return x
        
        elif self.architecture in ["modernbert", "bert", "e5", "minilm", "bge", "xlm-roberta"]:
            # For BERT-based models, try to access hidden states
            outputs = self.model(input_ids)
            
            # Check various ways BERT models return hidden states
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                return outputs.hidden_states[-1]  # Last layer
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # Some models return (logits, hidden_states)
                return outputs[1] if outputs[1].ndim == 3 else outputs[0]
            else:
                # If we only get logits, try to access the model's encoder
                if hasattr(self.model, 'encoder'):
                    return self.model.encoder(input_ids)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
                    return self.model.model.encoder(input_ids)
                else:
                    # Last resort: use the raw output (may be logits)
                    return outputs
        
        elif self.architecture == "arctic":
            # Arctic models may have specific architecture
            outputs = self.model(input_ids)
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            else:
                return outputs
        
        else:
            # Generic approach
            outputs = self.model(input_ids)
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            else:
                return outputs
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using architecture-specific extraction."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Universal embedding model not properly loaded")
        
        import time
        start_time = time.time()
        embeddings = []

        for text in texts:
            try:
                # Tokenize the text
                tokens = self.tokenizer.encode(text)
                input_ids = mx.array([tokens])
                
                # Get hidden states using architecture-specific method
                hidden_states = self._get_hidden_states(input_ids)
                
                # Pool across sequence length to get fixed-size representation
                if hidden_states.ndim == 3:  # [batch, seq_len, hidden_dim]
                    embedding = mx.mean(hidden_states, axis=1).squeeze()
                elif hidden_states.ndim == 2:  # [seq_len, hidden_dim]
                    embedding = mx.mean(hidden_states, axis=0)
                else:
                    embedding = hidden_states.squeeze()

                # Normalize the embedding vector for similarity tasks
                embedding_norm = mx.sqrt(mx.sum(embedding * embedding))
                if embedding_norm > 1e-8:
                    embedding = embedding / embedding_norm

                # Convert to Python list
                embedding_list = embedding.tolist()
                
                # Ensure proper format
                if not isinstance(embedding_list, list):
                    embedding_list = [float(embedding_list)]
                elif len(embedding_list) == 0:
                    # Fallback embedding size based on architecture
                    default_size = {
                        "qwen": 2560, "gte_qwen": 4096, "modernbert": 768,
                        "arctic": 1024, "e5": 768, "minilm": 384, "bge": 384
                    }.get(self.architecture, 768)
                    embedding_list = [0.0] * default_size
                
                embeddings.append(embedding_list)

            except Exception as e:
                logger.error(f"Error generating {self.architecture} embedding for text: {e}")
                # Return appropriate fallback embedding
                default_size = {
                    "qwen": 2560, "gte_qwen": 4096, "modernbert": 768,
                    "arctic": 1024, "e5": 768, "minilm": 384, "bge": 384
                }.get(self.architecture, 768)
                embeddings.append([0.0] * default_size)

        end_time = time.time()
        logger.info(f"Generated {len(embeddings)} {self.architecture} embeddings in {end_time - start_time:.2f}s")
        
        # Log embedding stats for debugging
        if embeddings and len(embeddings[0]) > 0:
            sample_vals = embeddings[0][:5]
            all_vals = [val for emb in embeddings for val in emb]
            logger.info(f"{self.architecture} embedding stats - dims: {len(embeddings[0])}, sample: {sample_vals}, range: {min(all_vals):.3f} to {max(all_vals):.3f}")

        return embeddings


class MLXQwen3EmbeddingWrapper(MLXModelWrapper):
    """Specialized wrapper for Qwen3 embedding models."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Load model and tokenizer for Qwen3 embeddings
        try:
            model, tokenizer = load(model_path)
            super().__init__(model, tokenizer, model_path, config)
            self.model_type = "embedding"
            logger.info(f"Loaded Qwen3 embedding model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Qwen3 embedding model: {e}")
            super().__init__(None, None, model_path, config)
            self.model_type = "embedding"

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Qwen3 model with proper embedding extraction."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen3 embedding model not properly loaded")
        
        import time
        start_time = time.time()
        embeddings = []

        for text in texts:
            try:
                # Tokenize the text
                tokens = self.tokenizer.encode(text)
                
                # Convert to MLX array and add batch dimension
                input_ids = mx.array([tokens])
                
                # For Qwen3 embeddings, we need to get hidden states, not logits
                # The model() call returns logits, but we need to access the hidden states
                # from the last layer before the language modeling head
                
                # Get the hidden states by calling the model's layers directly
                x = self.model.model.embed_tokens(input_ids)
                
                # Apply each transformer layer
                for layer in self.model.model.layers:
                    x = layer(x)
                
                # Apply final layer norm
                if hasattr(self.model.model, 'norm'):
                    x = self.model.model.norm(x)
                
                # Now x contains the hidden states, not the logits
                hidden_states = x

                # For embedding models, we want to pool across the sequence length
                if hidden_states.ndim == 3:  # [batch, seq_len, hidden_dim]
                    # Mean pooling across sequence length to get fixed-size representation
                    embedding = mx.mean(hidden_states, axis=1).squeeze()
                elif hidden_states.ndim == 2:  # [seq_len, hidden_dim]
                    # Mean pooling across sequence length
                    embedding = mx.mean(hidden_states, axis=0)
                else:
                    # If 1D, use as-is (shouldn't happen for typical embedding models)
                    embedding = hidden_states.squeeze()

                # Normalize the embedding vector (important for similarity tasks)
                # This makes it compatible with sentence-transformer expectations
                embedding_norm = mx.sqrt(mx.sum(embedding * embedding))
                if embedding_norm > 1e-8:  # Avoid division by zero
                    embedding = embedding / embedding_norm

                # Convert to Python list
                embedding_list = embedding.tolist()
                
                # Ensure it's the right dimensionality (Qwen3-4B should be ~4096 dims)
                if not isinstance(embedding_list, list):
                    embedding_list = [float(embedding_list)]
                elif len(embedding_list) == 0:
                    # Fallback to reasonable embedding size
                    embedding_list = [0.0] * 4096
                
                embeddings.append(embedding_list)

            except Exception as e:
                logger.error(f"Error generating Qwen3 embedding for text: {e}")
                # Return a zero embedding as fallback (4096 dims for Qwen3-4B)
                embeddings.append([0.0] * 4096)

        end_time = time.time()
        logger.info(f"Generated {len(embeddings)} Qwen3 embeddings in {end_time - start_time:.2f}s")
        
        # Log embedding stats for debugging
        if embeddings and len(embeddings[0]) > 0:
            sample_vals = embeddings[0][:10]
            all_vals = [val for emb in embeddings for val in emb]
            logger.info(f"Qwen3 embedding stats - dims: {len(embeddings[0])}, sample: {sample_vals[:5]}, range: {min(all_vals):.3f} to {max(all_vals):.3f}")

        return embeddings


class MLXMiniLMWrapper(MLXModelWrapper):
    """Specialized wrapper for MiniLM models using mlx_embeddings library."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        # MiniLM models using mlx_embeddings don't use standard tokenizers
        super().__init__(None, None, model_path, config)
        self.model_type = "embedding"
        self.embedding_model = None
        self.tokenizer = None

    def load_embedding_model(self):
        """Load the MiniLM model using mlx_embeddings."""
        if not HAS_MLX_EMBEDDINGS:
            raise ImportError("mlx_embeddings is required for MiniLM models. Install with: pip install mlx_embeddings")
        
        try:
            logger.info(f"Loading MiniLM model using mlx_embeddings: {self.model_path}")
            self.embedding_model, self.tokenizer = mlx_embeddings_load(self.model_path)
            logger.info("Successfully loaded MiniLM model via mlx_embeddings")
        except Exception as e:
            logger.error(f"Failed to load MiniLM model with mlx_embeddings: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using mlx_embeddings library."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        try:
            # Use mlx_embeddings generate function as shown in reference
            output = mlx_embeddings_generate(self.embedding_model, self.tokenizer, texts=texts)
            
            # Extract normalized embeddings
            embeddings = output.text_embeds
            
            # Convert MLX array to Python list
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            # Ensure proper format (list of lists)
            if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                if not isinstance(embeddings_list[0], list):
                    # Single embedding, wrap in list
                    embeddings_list = [embeddings_list]
            
            logger.info(f"Generated {len(embeddings_list)} MiniLM embeddings via mlx_embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating MiniLM embeddings: {e}")
            # Return fallback zero embeddings (384 dims for MiniLM-L6)
            return [[0.0] * 384 for _ in texts]


class MLXSentenceTransformerWrapper(MLXModelWrapper):
    """Wrapper for models that require sentence-transformers (like XLM-RoBERTa Arctic models)."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Initialize without loading MLX model since we'll use sentence-transformers
        super().__init__(None, None, model_path, config)
        self.model_type = "embedding"
        self.sentence_model = None
        self.load_sentence_transformer_model()
    
    def load_sentence_transformer_model(self):
        """Load model using sentence-transformers library."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map MLX community model paths to original paths
            model_path_mapping = {
                'mlx-community/snowflake-arctic-embed-l-v2.0-4bit': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                'mlx-community/snowflake-arctic-embed-l-v2.0-bf16': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                'snowflake-arctic-embed-l-v2-0-4bit': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                'snowflake-arctic-embed-l-v2-0-bf16': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                'snowflake-arctic-embed-l-v2.0-4bit': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                'snowflake-arctic-embed-l-v2.0-bf16': 'Snowflake/snowflake-arctic-embed-l-v2.0',
                # Add MiniLM mappings
                'mlx-community/all-MiniLM-L6-v2-4bit': 'sentence-transformers/all-MiniLM-L6-v2',
                'mlx-community/all-MiniLM-L6-v2-bf16': 'sentence-transformers/all-MiniLM-L6-v2',
                'all-minilm-l6-v2-4bit': 'sentence-transformers/all-MiniLM-L6-v2',
                'all-minilm-l6-v2-bf16': 'sentence-transformers/all-MiniLM-L6-v2',
            }
            
            # Use mapped path if available, otherwise use original
            actual_model_path = model_path_mapping.get(self.model_path, self.model_path)
            logger.info(f"Loading model via sentence-transformers: {self.model_path} -> {actual_model_path}")
            
            self.sentence_model = SentenceTransformer(actual_model_path)
            logger.info(f"Successfully loaded model via sentence-transformers")
        except ImportError:
            raise RuntimeError("sentence-transformers library is required. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load model with sentence-transformers: {e}")
            raise RuntimeError(f"Failed to load model with sentence-transformers: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        if self.sentence_model is None:
            raise RuntimeError("Sentence transformer model not loaded")
        
        try:
            import numpy as np
            embeddings = self.sentence_model.encode(texts, convert_to_numpy=True)
            # Convert numpy array to list of lists
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings with sentence-transformers: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")


class MLXEmbeddingWrapper(MLXModelWrapper):
    """Specialized wrapper for MLX embedding models using mlx_embedding_models."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Embedding models don't use tokenizers in the same way
        super().__init__(None, None, model_path, config)
        self.model_type = "embedding"
        self.embedding_model = None
        self.model_name = None
        
        # Extract model name from path for registry lookup
        if "/" in model_path:
            self.model_name = model_path.split("/")[-1]
        else:
            self.model_name = model_path

    def load_embedding_model(self):
        """Load the embedding model using mlx_embedding_models."""
        if not HAS_MLX_EMBEDDING_MODELS:
            raise ImportError("mlx_embedding_models is required for embedding models. Install with: pip install mlx_embedding_models")
        
        try:
            # Try to load from registry first (for supported models like BGE)
            if "bge" in self.model_name.lower():
                # Map to registry names
                if "bge-small" in self.model_name.lower():
                    registry_name = "bge-small"
                elif "bge-base" in self.model_name.lower():
                    registry_name = "bge-base"
                elif "bge-large" in self.model_name.lower():
                    registry_name = "bge-large"
                else:
                    registry_name = "bge-small"  # Default fallback
                
                logger.info(f"Loading BGE embedding model from registry: {registry_name}")
                self.embedding_model = EmbeddingModel.from_registry(registry_name)
            
            elif "minilm" in self.model_name.lower():
                # Map MLX community MiniLM models to original sentence-transformers
                logger.info("Loading MiniLM embedding model from sentence-transformers")
                self.embedding_model = EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            else:
                # For other models, try loading from path
                logger.info(f"Loading embedding model from path: {self.model_path}")
                self.embedding_model = EmbeddingModel.from_pretrained(self.model_path)
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to parent class behavior
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using mlx_embedding_models."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        try:
            # Use the proper MLX embedding API
            embeddings = self.embedding_model.encode(texts)
            
            # Convert MLX array to Python list
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif hasattr(embeddings, '__array__'):
                # Handle numpy arrays
                import numpy as np
                embeddings = np.array(embeddings).tolist()
            
            # Ensure we return a list of lists
            if isinstance(embeddings, list) and len(embeddings) > 0:
                if not isinstance(embeddings[0], list):
                    # Single embedding, wrap in list
                    embeddings = [embeddings]
                    
                # Ensure all embeddings are lists of floats
                result = []
                for emb in embeddings:
                    if isinstance(emb, list):
                        result.append([float(x) for x in emb])
                    else:
                        result.append([float(x) for x in emb.tolist()] if hasattr(emb, 'tolist') else [float(emb)])
                
                return result
            
            # If we get here, something unexpected happened
            logger.warning(f"Unexpected embeddings format: {type(embeddings)}")
            return embeddings if isinstance(embeddings, list) else [embeddings]
            
        except Exception as e:
            logger.error(f"Error generating embeddings with mlx_embedding_models: {e}")
            # Fallback to parent class behavior if available
            return super().generate_embeddings(texts)


class MLXWhisperWrapper(MLXModelWrapper):
    """Specialized wrapper for MLX-Whisper models."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Whisper models don't use tokenizers in the same way
        super().__init__(None, None, model_path, config)
        self.model_type = "whisper"

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio file using MLX-Whisper."""
        if not HAS_MLX_WHISPER:
            raise ImportError("mlx-whisper is required for Whisper models. Install with: pip install mlx-whisper")

        try:
            logger.info(f"Transcribing audio with Whisper model at: {self.model_path}")
            # Use the model path directly - mlx_whisper.transcribe loads the model internally
            result = mlx_whisper.transcribe(
                audio=audio_file_path,
                path_or_hf_repo=self.model_path,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise


class MLXParakeetWrapper(MLXModelWrapper):
    """Specialized wrapper for Parakeet models using parakeet-mlx."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(None, None, model_path, config)
        self.model_type = "audio"
        self.parakeet_model = None

    def load_parakeet_model(self):
        """Load the Parakeet model using parakeet-mlx."""
        if not HAS_PARAKEET_MLX:
            raise ImportError("parakeet-mlx is required for Parakeet models. Install with: pip install parakeet-mlx")

        try:
            import parakeet_mlx

            logger.info(f"Loading Parakeet model from: {self.model_path}")

            # Load the Parakeet model using from_pretrained
            self.parakeet_model = parakeet_mlx.from_pretrained(self.model_path)

            logger.info(f"Successfully loaded Parakeet model")
            return True
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            return False

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio using Parakeet model."""
        if not self.parakeet_model:
            if not self.load_parakeet_model():
                raise RuntimeError("Parakeet model not loaded")

        try:
            # Filter kwargs to only include supported parameters for Parakeet
            supported_params = {}
            if 'chunk_duration' in kwargs:
                supported_params['chunk_duration'] = kwargs['chunk_duration']
            if 'overlap_duration' in kwargs:
                supported_params['overlap_duration'] = kwargs['overlap_duration']

            # Use the Parakeet model to transcribe
            result = self.parakeet_model.transcribe(audio_file_path, **supported_params)

            # The result is an AlignedResult object, extract the text
            if hasattr(result, 'sentences'):
                # Combine all sentences into one text
                text_parts = []
                for sentence in result.sentences:
                    if hasattr(sentence, 'text'):
                        text_parts.append(sentence.text)
                    elif hasattr(sentence, 'content'):
                        text_parts.append(sentence.content)
                transcribed_text = " ".join(text_parts)
                return {"text": transcribed_text}
            elif hasattr(result, 'text'):
                return {"text": result.text}
            else:
                return {"text": str(result)}

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            raise


class MLXAudioWrapper(MLXModelWrapper):
    """Specialized wrapper for other MLX-Audio models (non-Parakeet)."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(None, None, model_path, config)
        self.model_type = "audio"
        self.audio_model = None

    def load_audio_model(self):
        """Load the audio model using mlx-audio."""
        if not HAS_MLX_AUDIO:
            raise ImportError("mlx-audio is required for audio models. Install with: pip install mlx-audio")

        try:
            logger.info(f"Audio model ready: {self.model_path}")
            self.model_subtype = "stt"
            self.audio_model = "loaded"  # Placeholder
            return True
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            return False

    def transcribe_audio(self, audio_file_path: str, **kwargs):
        """Transcribe audio using MLX-Audio STT models."""
        if not self.audio_model:
            if not self.load_audio_model():
                raise RuntimeError("Audio model not loaded")

        try:
            import tempfile
            import os
            from mlx_audio.stt.generate import generate

            # Create a temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_output_path = temp_file.name

            try:
                # Call the generate function with model path and audio file
                generate(
                    model_path=self.model_path,
                    audio_path=audio_file_path,
                    output_path=temp_output_path,
                    format="txt",
                    verbose=False
                )

                # Read the transcribed text from the output file
                with open(temp_output_path, 'r') as f:
                    transcribed_text = f.read().strip()

                return {"text": transcribed_text}

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise


class MLXVisionWrapper(MLXModelWrapper):
    """Specialized wrapper for MLX-VLM vision/multimodal models."""

    def __init__(self, model, processor, model_path: str, config: Dict[str, Any]):
        super().__init__(model, processor, model_path, config)
        self.model_type = "vision"
        self.processor = processor
        self.model_config = model.config  # Use model.config for MLX-VLM operations

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text for a vision model with text-only input."""
        # For text-only, the prompt is a string. Wrap it in the message structure.
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_images(messages, images=[], config=config)

    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Simulate streaming for vision models by running a full generation and yielding the single result."""
        import asyncio
        # Run the synchronous generate method for text-only. This will block, but it prevents crashing.
        result = self.generate(prompt, config)
        yield result.text
        await asyncio.sleep(0)  # Allow other tasks to run.

    def generate_with_images(self, messages: List[Dict[str, Any]], images: List[str], config: GenerationConfig) -> GenerationResult:
        """Generate text with optional image inputs using MLX-VLM, accepting structured messages."""
        # Ensure mlx-vlm is available
        if not HAS_MLX_VLM:
            raise ImportError("mlx-vlm is required for vision models. Install with: pip install mlx-vlm")

        import time
        from mlx_vlm.prompt_utils import apply_chat_template

        start_time = time.time()

        try:
            logger.info(f"Generating with {len(images)} images and {len(messages)} messages.")

            # ------------------------------------------------------------------
            # Processor sanity-patches (Gemma3/Qwen3 variants may miss methods)
            # ------------------------------------------------------------------
            # 1. Ensure a .tokenizer attribute exists so downstream utilities work
            if not hasattr(self.processor, "tokenizer") or self.processor.tokenizer is None:
                self.processor.tokenizer = self.processor  # Use self as a fallback

            # 2. Copy encode/decode from inner tokenizer if missing
            for _method in ("encode", "decode"):
                if not hasattr(self.processor, _method):
                    inner_tok = getattr(self.processor, "tokenizer", None)
                    if inner_tok is not None and hasattr(inner_tok, _method):
                        setattr(self.processor, _method, getattr(inner_tok, _method))
                        logger.debug(f"Patched processor with '{_method}' from inner tokenizer")

            # ------------------------------------------------------------------
            # Format prompt & generate
            # ------------------------------------------------------------------
            formatted_prompt = apply_chat_template(
                processor=self.processor,
                config=self.model_config,
                prompt=messages,
                num_images=len(images)
            )

            image_list = images  # already a list of file paths

            logger.info(f"Using images: {image_list if image_list else 'text-only'}")
            logger.debug(f"Formatted prompt: {repr(formatted_prompt)}")

            from mlx_vlm import generate as vlm_generate
            result = vlm_generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                image=image_list,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                verbose=True
            )

            logger.debug(f"MLX-VLM raw result: {repr(result)}")

            end_time = time.time()
            generation_time = end_time - start_time

            # Extract text and usage stats from the verbose result object
            generated_text = result.text.strip()
            prompt_tokens = getattr(result, 'prompt_tokens', 0)
            completion_tokens = getattr(result, 'generation_tokens', 0)
            total_tokens = getattr(result, 'total_tokens', prompt_tokens + completion_tokens)
            tokens_per_second = getattr(result, 'generation_tps', 0)

            # Fallback if attributes are not present or zero
            if total_tokens == 0:
                # Simple approximation for fallback
                prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                completion_tokens = len(generated_text.split())
                total_tokens = prompt_tokens + completion_tokens

            if tokens_per_second == 0 and generation_time > 0:
                tokens_per_second = completion_tokens / generation_time

            return GenerationResult(
                text=generated_text,
                prompt=str(messages),  # Store the structured messages as a string for logging
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_time_seconds=generation_time,
                tokens_per_second=tokens_per_second
            )

        except Exception as e:
            logger.error(f"MLX-VLM generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class MLXLoader:
    """Handles loading models with MLX-LM."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "mlx-gui")
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_model(self, model_id: str, token: Optional[str] = None) -> str:
        """Download model from HuggingFace Hub with progress tracking."""
        try:
            logger.info(f"Downloading model {model_id}")

            # Initialize progress tracking
            if hasattr(self, '_download_progress'):
                self._download_progress[model_id] = {
                    'status': 'downloading',
                    'progress': 0,
                    'downloaded_mb': 0,
                    'total_mb': 0,
                    'speed_mbps': 0,
                    'eta_seconds': 0,
                    'stage': 'Starting download...'
                }
            else:
                self._download_progress = {
                    model_id: {
                        'status': 'downloading',
                        'progress': 0,
                        'downloaded_mb': 0,
                        'total_mb': 0,
                        'speed_mbps': 0,
                        'eta_seconds': 0,
                        'stage': 'Starting download...'
                    }
                }

            # Increase file descriptor limit temporarily
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
            except ValueError:
                logger.warning("Could not increase file descriptor limit")

            # Update progress to show download starting
            self._download_progress[model_id].update({
                'stage': 'Downloading model files...',
                'progress': 5
            })

            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                token=token,
                local_files_only=False,
                max_workers=4  # Limit concurrent downloads
            )

            # Update progress to show download complete
            if model_id in self._download_progress:
                self._download_progress[model_id].update({
                    'status': 'download_complete',
                    'progress': 95,
                    'stage': 'Download complete, loading into memory...'
                })

            # Restore original limit
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
            except ValueError:
                pass

            logger.info(f"Model {model_id} downloaded to {local_path}")
            return local_path

        except Exception as e:
            # Clean up progress tracking on error
            if hasattr(self, '_download_progress') and model_id in self._download_progress:
                self._download_progress[model_id].update({
                    'status': 'failed',
                    'stage': f'Download failed: {str(e)}'
                })
            logger.error(f"Error downloading model {model_id}: {e}")
            raise
        finally:
            # Clean up progress tracking when done (success or failure)
            if hasattr(self, '_download_progress') and model_id in self._download_progress:
                # Keep progress for a short time for final status check
                import threading
                def cleanup_progress():
                    import time
                    time.sleep(30)  # Keep for 30 seconds
                    if hasattr(self, '_download_progress') and model_id in self._download_progress:
                        del self._download_progress[model_id]

                cleanup_thread = threading.Thread(target=cleanup_progress)
                cleanup_thread.daemon = True
                cleanup_thread.start()

    def get_download_progress(self, model_id: str) -> dict:
        """Get current download progress for a model."""
        if hasattr(self, '_download_progress') and model_id in self._download_progress:
            return self._download_progress[model_id].copy()
        return {
            'status': 'not_downloading',
            'progress': 0,
            'downloaded_mb': 0,
            'total_mb': 0,
            'speed_mbps': 0,
            'eta_seconds': 0,
            'stage': 'No download in progress'
        }

    def load_model(self, model_path: str) -> MLXModelWrapper:
        """Load a model using appropriate MLX library."""
        try:
            logger.info(f"Loading MLX model from {model_path}")

            # Detect model type from path/config first (before checking if path exists)
            model_type = self._detect_model_type(model_path)
            logger.info(f"Detected model type: {model_type}")

            # For parakeet models, we can load directly from HuggingFace ID
            if model_type == "audio" and "parakeet" in model_path.lower():
                logger.info(f"Loading Parakeet model from HuggingFace ID: {model_path}")
                config = {"estimated_memory_gb": 2.0}  # Default estimate for parakeet
                wrapper = MLXParakeetWrapper(model_path=model_path, config=config)
                if not wrapper.load_parakeet_model():
                    raise RuntimeError("Failed to load Parakeet model")
                return wrapper

            # For vision models, we can load directly from HuggingFace ID
            if model_type == "vision" and not os.path.exists(model_path):
                logger.info(f"Loading Vision model from HuggingFace ID: {model_path}")
                try:
                    from mlx_vlm import load as vlm_load
                    model, processor = vlm_load(model_path)
                    config = {"estimated_memory_gb": 8.0}  # Default estimate for vision models
                    wrapper = MLXVisionWrapper(
                        model=model,
                        processor=processor,
                        model_path=model_path,
                        config=config
                    )
                    return wrapper
                except Exception as e:
                    logger.error(f"Failed to load vision model from HuggingFace: {e}")
                    raise RuntimeError(f"Failed to load vision model: {e}")

            # Check if path exists (for local models)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Load config first to get additional info
            config = {}
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

            # Estimate memory usage
            memory_usage = self._estimate_model_memory(model_path)
            config['estimated_memory_gb'] = memory_usage

            # Load appropriate model type
            if model_type == "whisper":
                logger.info("Loading as Whisper model")
                wrapper = MLXWhisperWrapper(model_path=model_path, config=config)
                # MLX-Whisper models don't need pre-loading - they load on transcription

            elif model_type == "audio":
                if "parakeet" in model_path.lower():
                    logger.info("Loading as Parakeet model")
                    wrapper = MLXParakeetWrapper(model_path=model_path, config=config)
                    if not wrapper.load_parakeet_model():
                        raise RuntimeError("Failed to load Parakeet model")
                else:
                    logger.info("Loading as Audio model")
                    wrapper = MLXAudioWrapper(model_path=model_path, config=config)
                    if not wrapper.load_audio_model():
                        raise RuntimeError("Failed to load audio model")

            elif model_type == "vision":
                # Vision/multimodal model using MLX-VLM
                logger.info("Loading as Vision/VLM model")
                try:
                    # MLX-VLM needs both model and processor
                    from mlx_vlm import load as vlm_load
                    model, processor = vlm_load(model_path)
                    wrapper = MLXVisionWrapper(
                        model=model,
                        processor=processor,
                        model_path=model_path,
                        config=config
                    )
                except Exception as e:
                    # Fallback to regular text model if VLM loading fails
                    logger.warning(f"Failed to load as VLM model, trying as text model: {e}")
                    try:
                        model, tokenizer = load(model_path)
                        wrapper = MLXModelWrapper(
                            model=model,
                            tokenizer=tokenizer,
                            model_path=model_path,
                            config=config
                        )
                    except Exception as fallback_e:
                        raise RuntimeError(f"Failed to load as both VLM and text model. VLM error: {e}, Text error: {fallback_e}")

            elif model_type == "embedding":
                logger.info("Loading as embedding model")
                
                # Try specialized embedding approaches in order of preference
                
                # 1. For MiniLM models, use specialized mlx_embeddings wrapper
                if "minilm" in model_path.lower():
                    logger.info("Detected MiniLM embedding model, using mlx_embeddings")
                    try:
                        wrapper = MLXMiniLMWrapper(model_path=model_path, config=config)
                        wrapper.load_embedding_model()
                        logger.info("Successfully loaded MiniLM embedding model via mlx_embeddings")
                    except Exception as e:
                        logger.error(f"Failed to load MiniLM embedding model: {e}")
                        # Fallback to sentence-transformers for MiniLM if mlx_embeddings fails
                        # This is especially important for PyInstaller bundles where BERT model type might not be properly registered
                        if "model type bert not supported" in str(e).lower():
                            logger.warning("BERT model type not supported in mlx_embeddings, trying sentence-transformers fallback")
                            try:
                                wrapper = MLXSentenceTransformerWrapper(model_path=model_path, config=config)
                                logger.info("Successfully loaded MiniLM model via sentence-transformers fallback")
                            except Exception as st_error:
                                logger.error(f"Sentence-transformers fallback also failed: {st_error}")
                                # Final fallback to universal wrapper
                                try:
                                    wrapper = MLXUniversalEmbeddingWrapper(model_path=model_path, config=config)
                                    logger.info("Successfully loaded MiniLM model via universal wrapper fallback")
                                except Exception as universal_error:
                                    logger.error(f"All MiniLM loading methods failed: {universal_error}")
                                    raise e
                        else:
                            raise e
                
                # 2. For BGE models, try mlx_embedding_models (most reliable)
                elif "bge" in model_path.lower():
                    logger.info("Detected BGE embedding model, using mlx_embedding_models")
                    try:
                        wrapper = MLXEmbeddingWrapper(model_path=model_path, config=config)
                        wrapper.load_embedding_model()
                        logger.info("Successfully loaded BGE embedding model via mlx_embedding_models")
                    except Exception as e:
                        logger.error(f"Failed to load BGE embedding model: {e}")
                        raise
                
                # 2. Special handling for Arctic models (XLM-RoBERTa)
                elif "arctic" in model_path.lower():
                    logger.info("Detected Arctic model, trying sentence-transformers bypass")
                    try:
                        wrapper = MLXSentenceTransformerWrapper(model_path=model_path, config=config)
                        logger.info("Successfully loaded Arctic model via sentence-transformers")
                    except Exception as e:
                        logger.warning(f"Failed to load Arctic model with sentence-transformers: {e}")
                        # Fallback to universal wrapper
                        try:
                            wrapper = MLXUniversalEmbeddingWrapper(model_path=model_path, config=config)
                            logger.info("Successfully loaded Arctic model via universal wrapper")
                        except Exception as universal_e:
                            logger.error(f"Failed to load Arctic model with universal wrapper: {universal_e}")
                            raise
                
                # 3. For other known architectures, try universal wrapper
                elif any(arch in model_path.lower() for arch in ["qwen", "gte", "modernbert", "e5", "minilm"]):
                    logger.info("Detected custom architecture embedding model, using universal wrapper")
                    try:
                        wrapper = MLXUniversalEmbeddingWrapper(model_path=model_path, config=config)
                        logger.info("Successfully loaded embedding model via universal wrapper")
                    except Exception as e:
                        logger.warning(f"Failed to load with universal wrapper, trying fallback: {e}")
                        # Fallback to Qwen3 wrapper for Qwen models
                        if "qwen3" in model_path.lower() and "embedding" in model_path.lower():
                            try:
                                wrapper = MLXQwen3EmbeddingWrapper(model_path=model_path, config=config)
                                logger.info("Successfully loaded Qwen3 embedding model via legacy wrapper")
                            except Exception as qwen_e:
                                logger.error(f"Failed to load Qwen3 embedding model: {qwen_e}")
                                raise
                        else:
                            raise
                
                # 3. Try mlx_embedding_models for unknown models
                else:
                    logger.info("Unknown embedding model, trying mlx_embedding_models")
                    try:
                        wrapper = MLXEmbeddingWrapper(model_path=model_path, config=config)
                        wrapper.load_embedding_model()
                        logger.info("Successfully loaded embedding model via mlx_embedding_models")
                    except Exception as e:
                        # 4. Final fallback: universal wrapper
                        logger.warning(f"Failed to load with mlx_embedding_models, trying universal wrapper: {e}")
                        try:
                            wrapper = MLXUniversalEmbeddingWrapper(model_path=model_path, config=config)
                            logger.info("Successfully loaded embedding model via universal wrapper fallback")
                        except Exception as universal_e:
                            # 5. Last resort: standard MLX-LM (will produce logits, not embeddings)
                            logger.warning(f"Failed to load with universal wrapper, trying standard mlx-lm: {universal_e}")
                            try:
                                model, tokenizer = load(model_path)
                                wrapper = MLXModelWrapper(
                                    model=model,
                                    tokenizer=tokenizer,
                                    model_path=model_path,
                                    config=config
                                )
                                logger.warning("Loaded embedding model via standard mlx-lm (may produce logits instead of embeddings)")
                            except Exception as fallback_e:
                                if "not supported" in str(fallback_e).lower():
                                    logger.warning(f"Model architecture not supported by mlx-lm, treating as text model: {fallback_e}")
                                    model_type = "text"
                                else:
                                    logger.error(f"Failed to load embedding model with all methods: {fallback_e}")
                                    raise
                # If model_type changed to "text", fall through to standard text loading

            if model_type == "text":  # Handle both original text models and embedding fallbacks
                # Standard text generation model (or embedding model treated as text)
                logger.info(f"Loading as text generation model")
                try:
                    model, tokenizer = load(model_path)
                except KeyError as e:
                    if str(e) == "'model'":
                        # This is a known bug in MLX-LM's gemma3n implementation
                        raise ValueError(f"Model '{model_path}' has incompatible format for MLX-LM. This appears to be a gemma3n model with a known MLX-LM compatibility issue.")
                    else:
                        raise

                wrapper = MLXModelWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=model_path,
                    config=config
                )
                
                # If this was originally an embedding model, preserve that information
                original_model_type = self._detect_model_type(model_path)
                if original_model_type == "embedding":
                    wrapper.model_type = "embedding"
                    logger.info("Preserving embedding model type for BERT-based model loaded as text")

            logger.info(f"Successfully loaded {model_type} model from {model_path}")
            logger.info(f"Estimated memory usage: {memory_usage:.1f}GB")

            return wrapper

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def _detect_model_type(self, model_path: str) -> str:
        """Detect the type of model from path and contents."""
        # Check path for model type indicators
        path_lower = model_path.lower()

        # Whisper models
        if "whisper" in path_lower:
            return "whisper"

        # Parakeet and other audio models (including HuggingFace IDs)
        if any(keyword in path_lower for keyword in ["parakeet", "speech", "stt", "asr", "tdt"]):
            return "audio"

        # Embedding models - includes BGE, BERT-based models
        if any(keyword in path_lower for keyword in ["embedding", "bge-", "e5-", "all-minilm", "sentence", "bert", "arctic-embed"]):
            return "embedding"

        # Vision/multimodal models - includes Gemma 3 vision variants
        if any(keyword in path_lower for keyword in [
            "vision", "vlm", "multimodal", "llava", "qwen2-vl", "qwen3-vl", "idefics",
            "gemma-3n", "gemma3n"  # Gemma 3 vision variants use MLX-VLM
        ]):
            return "vision"

        # Check config.json for additional hints
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check architecture types
                arch = config.get("architectures", [])
                if arch:
                    arch_str = str(arch[0]).lower()
                    if "whisper" in arch_str:
                        return "whisper"
                    if any(keyword in arch_str for keyword in ["speech", "audio", "parakeet"]):
                        return "audio"
                    if any(keyword in arch_str for keyword in ["vision", "vlm", "multimodal", "llava", "qwen2vl", "idefics"]):
                        return "vision"
                    if any(keyword in arch_str for keyword in ["bert", "embedding", "bge"]):
                        return "embedding"

                # Check model type field
                model_type_field = config.get("model_type", "").lower()
                if "whisper" in model_type_field:
                    return "whisper"
                if any(keyword in model_type_field for keyword in ["speech", "audio"]):
                    return "audio"
                if any(keyword in model_type_field for keyword in ["vision", "multimodal"]):
                    return "vision"
                if "embedding" in model_type_field:
                    return "embedding"

        except Exception as e:
            logger.debug(f"Could not read config for model type detection: {e}")

        # Default to text generation
        return "text"

    def load_from_hub(self, model_id: str, token: Optional[str] = None) -> MLXModelWrapper:
        """Load a model directly from HuggingFace Hub."""
        try:
            # Check if model is MLX compatible (with special handling for Arctic models)
            hf_client = get_huggingface_client(token)
            model_info = hf_client.get_model_details(model_id)

            # Allow Arctic models to bypass MLX compatibility check since we have custom handling
            is_arctic_model = "arctic" in model_id.lower()
            
            if not model_info or (not model_info.mlx_compatible and not is_arctic_model):
                raise ValueError(f"Model {model_id} is not MLX compatible")

            # Use MLX repo if available
            if model_info.mlx_repo_id:
                logger.info(f"Using MLX version: {model_info.mlx_repo_id}")
                model_id = model_info.mlx_repo_id

            # Download and load
            local_path = self.download_model(model_id, token)
            return self.load_model(local_path)

        except Exception as e:
            logger.error(f"Error loading model {model_id} from hub: {e}")
            raise

    def _estimate_model_memory(self, model_path: str) -> float:
        """Estimate model memory usage from model files."""
        total_size = 0

        try:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt')):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)

            # Convert to GB and add overhead (MLX typically needs less memory than the file size)
            memory_gb = (total_size / (1024**3)) * 0.8  # 80% of file size
            return max(memory_gb, 0.5)  # Minimum 0.5GB

        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            return 2.0  # Default estimate


class MLXInferenceEngine:
    """High-level inference engine for MLX models."""

    def __init__(self):
        self.loader = MLXLoader()
        self._loaded_models: Dict[str, MLXModelWrapper] = {}

    def load_model(self, model_name: str, model_path: str, token: Optional[str] = None) -> MLXModelWrapper:
        """Load a model by name."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        # Check for HF token in environment if not provided
        if not token:
            import os
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')

        # Determine if this is a local path or HuggingFace model ID
        if os.path.exists(model_path):
            wrapper = self.loader.load_model(model_path)
        else:
            # Assume it's a HuggingFace model ID
            wrapper = self.loader.load_from_hub(model_path, token)

        self._loaded_models[model_name] = wrapper
        return wrapper

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self._loaded_models:
            # MLX models are automatically garbage collected
            del self._loaded_models[model_name]

            # Force garbage collection
            import gc
            gc.collect()

            # Clear MLX memory
            try:
                mx.clear_cache()
            except AttributeError:
                mx.metal.clear_cache()

            logger.info(f"Unloaded model {model_name}")

    def generate(self, model_name: str, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using a loaded model."""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} is not loaded")

        wrapper = self._loaded_models[model_name]
        return wrapper.generate(prompt, config)

    async def generate_stream(self, model_name: str, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model {model_name} is not loaded")

        wrapper = self._loaded_models[model_name]
        async for token in wrapper.generate_stream(prompt, config):
            yield token

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._loaded_models.keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._loaded_models

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        if model_name not in self._loaded_models:
            return None

        wrapper = self._loaded_models[model_name]
        return {
            "model_path": wrapper.model_path,
            "model_type": wrapper.model_type,
            "config": wrapper.config,
            "estimated_memory_gb": wrapper.config.get("estimated_memory_gb", 0)
        }

    def get_download_progress(self, model_id: str) -> dict:
        """Get current download progress for a model."""
        return self.loader.get_download_progress(model_id)


# Global inference engine
_inference_engine: Optional[MLXInferenceEngine] = None


def get_inference_engine() -> MLXInferenceEngine:
    """Get the global MLX inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = MLXInferenceEngine()
    return _inference_engine


def validate_mlx_installation():
    """Validate that MLX is properly installed and working."""
    try:
        # Test basic MLX functionality
        x = mx.array([1, 2, 3])
        y = mx.array([4, 5, 6])
        z = x + y
        result = z.tolist()

        if result == [5, 7, 9]:
            logger.info("MLX installation validated successfully")
            return True
        else:
            logger.error("MLX computation test failed")
            return False

    except Exception as e:
        logger.error(f"MLX validation failed: {e}")
        return False


def get_mlx_device_info() -> Dict[str, Any]:
    """Get information about the MLX device."""
    try:
        return {
            "device": "GPU" if mx.metal.is_available() else "CPU",
            "metal_available": mx.metal.is_available(),
            "memory_limit": mx.metal.get_memory_limit() if mx.metal.is_available() else None,
            "peak_memory": mx.metal.get_peak_memory() if mx.metal.is_available() else None,
            "cache_size": mx.metal.get_cache_memory() if mx.metal.is_available() else None
        }
    except Exception as e:
        logger.error(f"Error getting MLX device info: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------
#  Hot-patch for Gemma VLM checkpoints missing conv_stem bias
#  We monkey-patch mlx_vlm.utils.sanitize_weights so the patch applies
#  both in the dev virtual-env and inside PyInstaller (runtime hook
#  already patches in the bundled app). Idempotent.
# ------------------------------------------------------------------
def _patch_mlx_vlm_bias():
    """Ensure sanitize_weights injects a zero bias if Gemma conv stem bias is absent (robust)."""
    try:
        import mlx_vlm.utils as vlm_utils
        if getattr(vlm_utils, "__bias_patch_applied", False):
            return

        orig_sanitize = vlm_utils.sanitize_weights

        def _ensure_bias(weights):
            """Inject zero bias if missing and weight present."""
            bias_key = "vision_tower.timm_model.conv_stem.conv.bias"
            weight_key = "vision_tower.timm_model.conv_stem.conv.weight"
            if bias_key not in weights and weight_key in weights:
                try:
                    import mlx.core as mx
                    w = weights[weight_key]
                    out_channels = w.shape[0] if hasattr(w, "shape") else len(w)
                    dtype = getattr(w, "dtype", mx.float32)
                    weights[bias_key] = mx.zeros((out_channels,), dtype=dtype)
                    logger.debug("Injected zero conv_stem bias")
                except Exception as e:
                    logger.warning(f"Bias injection failed: {e}")

        def _patched_sanitize(model_obj, weights, config=None):
            # Pre-patch to avoid upstream missing-param error
            _ensure_bias(weights)
            try:
                weights = orig_sanitize(model_obj, weights, config)
            except Exception as first_err:
                # Retry once after ensuring bias (if not already)
                _ensure_bias(weights)
                try:
                    weights = orig_sanitize(model_obj, weights, config)
                except Exception:
                    # Give up – re-raise original
                    raise first_err
            return weights

        vlm_utils.sanitize_weights = _patched_sanitize
        vlm_utils.__bias_patch_applied = True
        logger.info("Applied Gemma VLM bias patch (development mode, robust)")
    except Exception as e:
        # mlx_vLM may be missing or partially importable; don't crash app startup
        logger.debug(f"VLM bias patch skipped: {e}")

# Apply patch immediately on module import
_patch_mlx_vlm_bias()
