import base64
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import niquests


class ImageType(str, Enum):
    PNG = "image/png"
    JPEG = "image/jpeg"
    JPG = "image/jpg"
    WEBP = "image/webp"
    GIF = "image/gif"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


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
class SystemMessage:
    content: str


@dataclass
class UserMessage:
    content: str
    images: (
        tuple[bytes, str | ImageType]
        | tuple[str | ImageType, bytes]
        | list[tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes]]
        | None
    ) = None


@dataclass
class AssistantMessage:
    content: str
    images: (
        tuple[bytes, str | ImageType]
        | tuple[str | ImageType, bytes]
        | list[tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes]]
        | None
    ) = None


@dataclass
class Model:
    model: str
    api_key: str
    base_url: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    temperature: float = 0

    ENDPOINT = "/chat/completions"

    @staticmethod
    def _build_image_content(
        images: tuple[bytes, str | ImageType]
        | tuple[str | ImageType, bytes]
        | list[tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes]],
    ) -> list[dict]:
        if isinstance(images, tuple):
            images = [images]

        result = []
        for item in images:
            data, mime = item if isinstance(item[0], bytes) else (item[1], item[0])
            mime_str = mime.value if isinstance(mime, ImageType) else mime
            b64 = base64.b64encode(data).decode("utf-8")
            result.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_str};base64,{b64}"},
                }
            )
        return result

    @staticmethod
    def _normalize_messages(
        messages: list[dict | SystemMessage | UserMessage | AssistantMessage],
    ) -> list[dict]:
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
                continue

            if isinstance(msg, SystemMessage):
                result.append({"role": Role.SYSTEM, "content": msg.content})
                continue

            result.append(
                {
                    "role": Role.USER
                    if isinstance(msg, UserMessage)
                    else Role.ASSISTANT,
                    "content": [
                        {"type": "text", "text": msg.content},
                        *Model._build_image_content(msg.images),
                    ]
                    if msg.images
                    else msg.content,
                }
            )
        return result

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

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

    def invoke(
        self,
        prompt: str | list[dict | SystemMessage | UserMessage | AssistantMessage],
    ) -> Message:
        messages = (
            [{"role": Role.USER, "content": prompt}]
            if isinstance(prompt, str)
            else self._normalize_messages(prompt)
        )

        payload = self._build_payload(messages)
        with niquests.Session() as session:
            r = session.post(
                f"{self.base_url}{self.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            return self._parse_response(r.json())

    async def ainvoke(
        self,
        prompt: str | list[dict | SystemMessage | UserMessage | AssistantMessage],
    ) -> Message:
        messages = (
            [{"role": Role.USER, "content": prompt}]
            if isinstance(prompt, str)
            else self._normalize_messages(prompt)
        )

        payload = self._build_payload(messages)
        async with niquests.AsyncSession() as session:
            r = await session.post(
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

    def chat(
        self,
        initial_messages: list[dict | SystemMessage | UserMessage | AssistantMessage]
        | None = None,
    ) -> "ChatSession":
        return ChatSession(self, initial_messages)

    def achat(
        self,
        initial_messages: list[dict | SystemMessage | UserMessage | AssistantMessage]
        | None = None,
    ) -> "AsyncChatSession":
        return AsyncChatSession(self, initial_messages)


class ChatSession:
    def __init__(
        self,
        model: Model,
        initial_messages: list[dict | SystemMessage | UserMessage | AssistantMessage]
        | None = None,
    ):
        self._model = model
        self.history: list[dict] = (
            Model._normalize_messages(initial_messages) if initial_messages else []
        )

    def send(
        self,
        prompt: str,
        images: (
            tuple[bytes, str | ImageType]
            | tuple[str | ImageType, bytes]
            | list[tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes]]
            | None
        ) = None,
    ) -> Message:
        content = (
            [
                {"type": "text", "text": prompt},
                *Model._build_image_content(images),
            ]
            if images
            else prompt
        )

        payload = self._model._build_payload(
            [*self.history, {"role": Role.USER, "content": content}]
        )

        with niquests.Session() as session:
            r = session.post(
                f"{self._model.base_url}{self._model.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self._model.api_key}"},
            )
            r.raise_for_status()
            response = self._model._parse_response(r.json())

        self.history.extend(
            [
                {"role": Role.USER, "content": content},
                {"role": Role.ASSISTANT, "content": response.content},
            ]
        )
        return response

    def __enter__(self) -> "ChatSession":
        return self

    def __exit__(self, *_) -> None:
        pass


class AsyncChatSession:
    def __init__(
        self,
        model: Model,
        initial_messages: list[dict | SystemMessage | UserMessage | AssistantMessage]
        | None = None,
    ):
        self._model = model
        self.history: list[dict] = (
            Model._normalize_messages(initial_messages) if initial_messages else []
        )

    async def send(
        self,
        prompt: str,
        images: (
            tuple[bytes, str | ImageType]
            | tuple[str | ImageType, bytes]
            | list[tuple[bytes, str | ImageType] | tuple[str | ImageType, bytes]]
            | None
        ) = None,
    ) -> Message:
        content = (
            [
                {"type": "text", "text": prompt},
                *Model._build_image_content(images),
            ]
            if images
            else prompt
        )

        payload = self._model._build_payload(
            [*self.history, {"role": Role.USER, "content": content}]
        )

        async with niquests.AsyncSession() as session:
            r = await session.post(
                f"{self._model.base_url}{self._model.ENDPOINT}",
                json=payload,
                headers={"Authorization": f"Bearer {self._model.api_key}"},
            )
            r.raise_for_status()
            response = self._model._parse_response(r.json())

        self.history.extend(
            [
                {"role": Role.USER, "content": content},
                {"role": Role.ASSISTANT, "content": response.content},
            ]
        )
        return response

    async def __aenter__(self) -> "AsyncChatSession":
        return self

    async def __aexit__(self, *_) -> None:
        pass


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
        self._models[name] = Model(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            input_cost_per_1m=input_cost,
            output_cost_per_1m=output_cost,
            temperature=temperature,
        )
        return self._models[name]
