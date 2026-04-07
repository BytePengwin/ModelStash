import niquests
import base64
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Union


class ImageType(str, Enum):
    PNG = "image/png"
    JPEG = "image/jpeg"
    JPG = "image/jpg"
    WEBP = "image/webp"
    GIF = "image/gif"


@dataclass
class Metadata:
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class Message:
    content: str
    metadata: Metadata

@dataclass
class Model:
    model: str
    api_key: str
    base_url: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    temperature: float = 0

    ENDPOINT = "/chat/completions"

    def _build_message(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        mime_type: Union[str, ImageType] = ImageType.PNG,
    ) -> dict:
        if image_bytes is None:
            return {"role": "user", "content": prompt}
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_str = mime_type.value if isinstance(mime_type, ImageType) else mime_type
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_str};base64,{b64}"},
                },
            ],
        }

    def _parse_response(self, data: dict) -> Message:
        return Message(
            content=data["choices"][0]["message"]["content"],
            metadata=Metadata(
                input_tokens=data["usage"]["prompt_tokens"],
                output_tokens=data["usage"]["completion_tokens"],
                cost=self.calculate_cost(
                    data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]
                ),
            ),
        )

    async def ainvoke(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        mime_type: Union[str, ImageType] = ImageType.PNG,
    ) -> Message:
        payload = {
            "model": self.model,
            "messages": [self._build_message(prompt, image_bytes, mime_type=mime_type)],
            "temperature": self.temperature,
        }
        async with niquests.AsyncSession() as session:
            r = await session.post(
                f"{self.base_url}{self.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            return self._parse_response(r.json())

    def invoke(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        mime_type: Union[str, ImageType] = ImageType.PNG,
    ) -> Message:
        payload = {
            "model": self.model,
            "messages": [self._build_message(prompt, image_bytes, mime_type=mime_type)],
            "temperature": self.temperature,
        }
        with niquests.Session() as session:
            r = session.post(
                f"{self.base_url}{self.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            return self._parse_response(r.json())

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_cost_per_1m
            + output_tokens / 1_000_000 * self.output_cost_per_1m
        )


class ModelContainer:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self._models: dict[str, Model] = {}

    def __getattr__(self, name: str) -> Model:
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name in self._models:
            return self._models[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no model named '{name}'"
        )

    def __iter__(self) -> Iterator[Model]:
        return iter(self._models.values())

    def add(
        self,
        name: str,
        model_name: str,
        input_cost: float,
        output_cost: float,
        temperature: float = 0,
    ) -> Model:
        model = Model(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            input_cost_per_1m=input_cost,
            output_cost_per_1m=output_cost,
            temperature=temperature,
        )
        self._models[name] = model
        return model
