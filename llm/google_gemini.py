from typing import Any, List, Mapping, Optional

import google.generativeai as genai
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult


class GoogleGeminiChat(BaseChatModel):
    model_name: str = "gemini-pro"
    temperature: float = 0.1
    google_api_key: Optional[str] = None

    def __init__(self, google_api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.google_api_key = google_api_key
        self.model_name = kwargs.get("model_name", self.model_name)
        self.temperature = kwargs.get("temperature", self.temperature)
        genai.configure(api_key=google_api_key)

    @property
    def _llm_type(self) -> str:
        return "google-gemini-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        model = genai.GenerativeModel(self.model_name)
        prompt = self._messages_to_prompt(messages)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            ),
        )
        text = self._extract_text(response)
        if stop:
            for stop_token in stop:
                if stop_token in text:
                    text = text.split(stop_token)[0]
                    break

        llm_output = {"model_name": self.model_name}
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            llm_output["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))],
            llm_output=llm_output,
        )

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt_parts = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__.lower())
            prompt_parts.append(f"{role}: {message.content}")
        return "\n".join(prompt_parts)

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            collected = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    collected.append(part_text)
            if collected:
                return "".join(collected)

        raise ValueError("Empty response from Google Gemini")

