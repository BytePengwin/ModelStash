import httpx
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Message:
    """Response message from API"""
    content: str
    usage_metadata: dict


@dataclass
class Model:
    """OpenAI-compatible API model"""
    model: str
    api_key: str
    base_url: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    temperature: float = 0

    ENDPOINT = "/chat/completions"

    async def ainvoke(self, prompt: str) -> Message:
        """Async invoke"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            data = response.json()

        return Message(
            content=data["choices"][0]["message"]["content"],
            usage_metadata={
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"]
            }
        )

    def invoke(self, prompt: str) -> Message:
        """Sync invoke"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}{self.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            data = response.json()

        return Message(
            content=data["choices"][0]["message"]["content"],
            usage_metadata={
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"]
            }
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        return (input_tokens / 1_000_000 * self.input_cost_per_1m +
                output_tokens / 1_000_000 * self.output_cost_per_1m)


class ModelContainer:
    """Container for managing multiple Model instances"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def __iter__(self) -> Iterator[Model]:
        for name, value in self.__dict__.items():
            if isinstance(value, Model):
                yield value

    def add(self, name: str, model_name: str, input_cost: float, output_cost: float,
            temperature: float = 0) -> None:
        client = Model(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            input_cost_per_1m=input_cost,
            output_cost_per_1m=output_cost,
            temperature=temperature
        )
        setattr(self, name, client)