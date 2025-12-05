from langchain_openai import ChatOpenAI
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    client: ChatOpenAI
    input_cost_per_1m: float
    output_cost_per_1m: float

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        print(f"Input token cost: {input_tokens / 1_000_000 * self.input_cost_per_1m:.10f}")
        print(f"Output token cost: {output_tokens / 1_000_000 * self.output_cost_per_1m:.10f}")
        return (input_tokens / 1_000_000 * self.input_cost_per_1m +
                output_tokens / 1_000_000 * self.output_cost_per_1m)


class ModelContainer:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def __iter__(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Model):
                yield value

    def add(self, name: str, model_name: str, input_cost: float, output_cost: float,
            temperature: float = 0) -> None:
        client = ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature
        )
        model = Model(
            name=model_name,
            client=client,
            input_cost_per_1m=input_cost,
            output_cost_per_1m=output_cost
        )
        setattr(self, name, model)