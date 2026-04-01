from typing import Protocol

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel


class AsyncClient(Protocol):
    async def structured_output[T: BaseModel](self, prompt, output_format: type[T]) -> T | None: ...


class ClaudeClient:
    def __init__(self, client: AsyncAnthropic, model: str, max_output_tokens: int = 16384):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    async def structured_output[T: BaseModel](self, prompt: str, output_format: type[T]) -> T | None:
        response = await self.client.messages.parse(
            max_tokens=self.max_output_tokens,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
        return response.parsed_output


class OpenAIClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_output_tokens: int = 16384,
        extra_body: dict | None = None,
    ):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.extra_body = extra_body

    async def structured_output[T: BaseModel](self, prompt: str, output_format: type[T]) -> T | None:
        parse_kwargs = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "input": [{"role": "user", "content": prompt}],
            "text_format": output_format,
        }
        if self.extra_body is not None:
            parse_kwargs["extra_body"] = self.extra_body

        response = await self.client.responses.parse(
            **parse_kwargs,
        )
        return response.output_parsed
