# ModelContainer

A Python wrapper for OpenAI-compatible APIs with cost tracking and async/sync support.

**Status:** Maintenance Mode - No new features planned.

## Features

- **Async & Sync**: Both `ainvoke()` and `invoke()` methods
- **Cost Tracking**: Built-in cost calculation based on token usage
- **Model Management**: Container class to manage multiple models
- **OpenAI Compatible**: Works with OpenAI, OpenRouter, and other compatible APIs

## Installation

```python
pip install modelcontainer
```

## Quick Start

### Single Model

```python
from __init__ import Model

model = Model(
    model="gpt-4",
    api_key="your-api-key",
    base_url="https://openrouter.io/api/v1",
    input_cost_per_1m=0.03,
    output_cost_per_1m=0.06
)

response = model.invoke("What is 2+2?")
print(response.content)
```

### Multiple Models

```python
from __init__ import ModelContainer

container = ModelContainer(api_key="your-api-key")

container.add("gpt4", "gpt-4", input_cost=0.03, output_cost=0.06)
container.add("gpt35", "gpt-3.5-turbo", input_cost=0.0005, output_cost=0.0015)

for model in container:
    response = model.invoke("Hello!")
    print(response.content)
```

## API Reference

### Model

**Methods:**
- `invoke(prompt: str) -> Message` - Synchronous API call
- `ainvoke(prompt: str) -> Message` - Asynchronous API call
- `calculate_cost(input_tokens: int, output_tokens: int) -> float` - Calculate request cost

### ModelContainer

**Methods:**
- `add(name, model_name, input_cost, output_cost, temperature=0)` - Add a model
- `__iter__()` - Iterate over all models

### Message

**Attributes:**
- `content: str` - Response text
- `usage_metadata: dict` - Token counts and usage info
