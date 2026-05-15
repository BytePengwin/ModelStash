# ModelStash

A lightweight Python library for managing and invoking multiple AI models with built-in cost tracking, token counting, and full chat history support.

Status: Maintenance Mode - No new features planned.

## Features

- **Multi-model management** - Register and switch between multiple AI models via `ModelContainer`
- **Sync & async support** - Use `invoke()` for synchronous calls or `ainvoke()` for async
- **Chat history** - Send multi-turn conversations with system prompts and message history
- **Stateful sessions** - Use context manager sessions (`chat()` / `achat()`) for automatic history tracking
- **Vision support** - Pass images to models that support multimodal inputs, with per-message image support
- **Cost tracking** - Automatic token counting and cost calculation per request
- **OpenRouter compatible** - Works with any OpenAI-compatible API endpoint

## Installation

```bash
pip install ModelStash
```

## Quick Start

```python
from ModelStash import ModelContainer

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
| `invoke(prompt)` | Synchronous call with a string or list of messages |
| `ainvoke(prompt)` | Async call with a string or list of messages |
| `calculate_cost(input_tokens, output_tokens)` | Calculate cost for tokens |
| `chat(initial_messages=None)` | Start a synchronous chat session (context manager) |
| `achat(initial_messages=None)` | Start an async chat session (context manager) |

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

### Role

Enum for message roles when using raw dicts.

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
```

### Typed Message Classes

Convenient dataclasses for building messages with IDE autocomplete.

> **Note:** The `images` type hint looks complex because it encodes both orderings: `(bytes, mime_type)` or `(mime_type, bytes)`. In practice, just pass `(image_bytes, ImageType.PNG)` or `("image/png", image_bytes)` — both work.

```python
@dataclass
class SystemMessage:
    content: str

@dataclass
class UserMessage:
    content: str
    images: tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes] | list[...] | None = None

@dataclass
class AssistantMessage:
    content: str
    images: tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes] | list[...] | None = None
```

### ImageType

Supported image MIME types:

- `ImageType.PNG`
- `ImageType.JPEG`
- `ImageType.JPG`
- `ImageType.WEBP`
- `ImageType.GIF`

## Examples

### Simple Text Prompt

```python
from ModelStash import ModelContainer

container = ModelContainer(api_key="...")
container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)

result = container.flash.invoke("What is the capital of France?")
print(result.content)
```

### Multi-Message Conversation (Stateless)

Send a full conversation history in a single call. The model sees all messages but no state is retained.

```python
from ModelStash import ModelContainer, Role, SystemMessage, UserMessage, AssistantMessage

container = ModelContainer(api_key="...")
container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)

# Using typed message classes
result = container.flash.invoke([
    SystemMessage("You are a helpful coding assistant."),
    UserMessage("How do I reverse a list in Python?"),
    AssistantMessage("You can use list[::-1] or the reversed() function."),
    UserMessage("Which is faster?"),
])
print(result.content)

# Using raw dicts with Role enum
result = container.flash.invoke([
    {"role": Role.SYSTEM, "content": "You are a helpful coding assistant."},
    {"role": Role.USER, "content": "How do I reverse a list in Python?"},
    {"role": Role.ASSISTANT, "content": "You can use list[::-1] or the reversed() function."},
    {"role": Role.USER, "content": "Which is faster?"},
])
```

### With Image Input

Images are passed as `(bytes, mime_type)` or `(mime_type, bytes)` tuples. Order doesn't matter — the library detects which element is the bytes and which is the MIME type.

> **Note:** The type hint looks complex because it encodes both orderings. In practice, just pass `(image_bytes, "image/png")` or `("image/png", image_bytes)` — both work.

```python
from ModelStash import ModelContainer, ImageType

container = ModelContainer(api_key="...")
container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)

with open("image.png", "rb") as f:
    image_bytes = f.read()

# Single image
result = container.flash.invoke([
    {"role": Role.USER, "content": "Describe this image", "images": [(image_bytes, ImageType.PNG)]},
])

# Or using typed classes
from ModelStash import UserMessage

result = container.flash.invoke([
    UserMessage("Describe this image", images=(image_bytes, ImageType.PNG)),
])

# Multiple images with mixed types
with open("photo.jpg", "rb") as f:
    jpg_bytes = f.read()

result = container.flash.invoke([
    UserMessage("Compare these images", images=[
        (image_bytes, ImageType.PNG),
        (jpg_bytes, "image/jpeg"),  # Can use raw strings too
    ]),
])
```

### Async Usage

```python
import asyncio
from ModelStash import ModelContainer

async def main():
    container = ModelContainer(api_key="...")
    container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)
    
    result = await container.flash.ainvoke("What is this?")
    print(result.content)

asyncio.run(main())
```

### Stateful Chat Session

Use `chat()` (sync) or `achat()` (async) for automatic history tracking. The session maintains the full conversation and appends messages only after successful API calls.

```python
from ModelStash import ModelContainer, SystemMessage

container = ModelContainer(api_key="...")
container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)

with container.flash.chat([SystemMessage("You are a pirate.")]) as session:
    msg1 = session.send("Hello!")
    print(msg1.content)  # "Ahoy matey!"
    
    msg2 = session.send("What's the weather like?")
    print(msg2.content)
    
    # Access full conversation history
    print(session.history)
    
    # History is fully editable
    session.history.pop()  # Remove last assistant message
```

```python
import asyncio
from ModelStash import ModelContainer

async def main():
    container = ModelContainer(api_key="...")
    container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)
    
    async with container.flash.achat() as session:
        msg = await session.send("Tell me a joke")
        print(msg.content)

asyncio.run(main())
```

### Session with Images

```python
from ModelStash import ModelContainer, ImageType

container = ModelContainer(api_key="...")
container.add("flash", "google/gemini-2.0-flash-001", 0.0, 0.0)

with open("image.png", "rb") as f:
    image_bytes = f.read()

with container.flash.chat() as session:
    msg = session.send("What's in this image?", images=(image_bytes, ImageType.PNG))
    print(msg.content)
    
    msg = session.send("What color is the main object?")
    print(msg.content)
```

## License

GPL-3.0-only
