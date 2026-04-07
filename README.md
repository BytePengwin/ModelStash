# ModelStash

A lightweight Python library for managing and invoking multiple AI models with built-in cost tracking and token counting.


Status: Maintenance Mode - No new features planned.

## Features

- **Multi-model management** - Register and switch between multiple AI models via `ModelContainer`
- **Sync & async support** - Use `invoke()` for synchronous calls or `ainvoke()` for async
- **Vision support** - Pass images to models that support multimodal inputs
- **Cost tracking** - Automatic token counting and cost calculation per request
- **OpenRouter compatible** - Works with any OpenAI-compatible API endpoint

## Installation

```bash
pip install ModelStash
```

## Quick Start

```python
from ModelStash import ModelContainer, ImageType

container = ModelContainer(api_key="your-api-key")

container.add(
    name="flash",
    model_name="google/gemini-2.0-flash-001",
    input_cost=0.0,
    output_cost=0.0,
)

result = container.flash.invoke("Hello, world!")
print(result.content)
print(f"Cost: ${result.metadata.cost:.6f}")
```

## API Reference

### ModelContainer

Manages a collection of models and their HTTP clients.

```python
container = ModelContainer(api_key="...", base_url="https://openrouter.ai/api/v1")
```

| Method | Description |
|--------|-------------|
| `add(name, model_name, input_cost, output_cost, temperature=0)` | Register a new model |
| `get(model_name)` | Get a model by name (via `__getattr__`) |

### Model

Represents a single model configuration.

```python
model = container.add("name", "model-id", input_cost=0.0, output_cost=0.0)
```

| Method | Description |
|--------|-------------|
| `invoke(prompt, image_bytes=None, mime_type=ImageType.PNG)` | Synchronous call |
| `ainvoke(prompt, image_bytes=None, mime_type=ImageType.PNG)` | Async call |
| `calculate_cost(input_tokens, output_tokens)` | Calculate cost for tokens |

### Message

Returned by model invocations.

```python
@dataclass
class Message:
    content: str        # Model's response text
    metadata: Metadata  # Token usage and cost info
```

### Metadata

Token usage and cost data.

```python
@dataclass
class Metadata:
    input_tokens: int   # Prompt tokens used
    output_tokens: int  # Completion tokens used
    cost: float         # Total cost in USD
```

### ImageType

Supported image MIME types:

- `ImageType.PNG`
- `ImageType.JPEG`
- `ImageType.JPG`
- `ImageType.WEBP`
- `ImageType.GIF`

## Examples

### Async Usage

```python
import asyncio
from ModelStash import ModelContainer

async def main():
    container = ModelContainer(api_key="...")
    container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)
    
    result = await container.flash.ainvoke("What is this?")
    print(result.content)
    
    container.close()

asyncio.run(main())
```

### With Image Input

```python
from ModelStash import ModelContainer, ImageType

container = ModelContainer(api_key="...")
container.add("vision", "google/gemini-2.0-flash-001", 0.0, 0.0)

with open("image.png", "rb") as f:
    image_bytes = f.read()

result = container.vision.invoke(
    "Describe this image",
    image_bytes=image_bytes,
    mime_type=ImageType.PNG,
)
```

## License

MIT
