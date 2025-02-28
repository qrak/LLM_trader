import configparser
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, APIConnectionError
from openai.types.chat import ChatCompletionChunk

from logger.logger import Logger
from utils.dataclass import ResponseBuffer


@dataclass
class ModelSettings:
    name: str
    base_url: str
    api_key: str


class ModelManager:
    def __init__(self, logger: Logger, config_path: str = "config/config.ini"):
        self.logger = logger
        self.config = self._load_config(config_path)
        self.model_config = self._load_model_config()
        self.primary_settings = self._get_primary_settings()
        self.fallback_settings = self._get_fallback_settings()
        self.current_settings = self.primary_settings
        self.client = self._init_client()

    async def close(self) -> None:
        if hasattr(self, 'client'):
            await self.client.close()

    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _load_model_config(self) -> configparser.ConfigParser:
        model_config = configparser.ConfigParser()
        model_config_path = "config/model_config.ini"

        if not os.path.exists(model_config_path):
            template_path = f"{model_config_path}.template"
            if os.path.exists(template_path):
                import shutil
                shutil.copy2(template_path, model_config_path)
                self.logger.warning(
                    f"Created {model_config_path} from template. Please update with your actual API key.")
            else:
                raise FileNotFoundError(f"Neither {model_config_path} nor template file exists")

        model_config.read(model_config_path)
        return model_config

    def _get_primary_settings(self) -> ModelSettings:
        return ModelSettings(
            name=self.model_config.get("model", "name"),
            base_url=self.model_config.get("model", "base_url"),
            api_key=self.model_config.get("model", "api_key")
        )

    def _get_fallback_settings(self) -> ModelSettings:
        return ModelSettings(
            name=self.config.get("model_fallback_settings", "name"),
            base_url=self.config.get("model_fallback_settings", "base_url"),
            api_key=self.config.get("model_fallback_settings", "api_key")
        )

    def _init_client(self) -> AsyncOpenAI:
        try:
            return AsyncOpenAI(
                base_url=self.current_settings.base_url,
                api_key=self.current_settings.api_key
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize client with current settings: {e}")
            return self._init_fallback_client()

    def _init_fallback_client(self) -> AsyncOpenAI:
        self.current_settings = self.fallback_settings
        self.logger.info("Switching to fallback model configuration")
        return AsyncOpenAI(
            base_url=self.current_settings.base_url,
            api_key=self.current_settings.api_key
        )

    async def send_prompt(self, prompt: str, buffer: ResponseBuffer) -> str:
        try:
            stream = await self.client.chat.completions.create(
                model=self.current_settings.name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that uses Chain of Thought reasoning to solve problems."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=4000
            )
            return await self._process_stream(stream, buffer)

        except (APIConnectionError, TimeoutError):
            self.logger.warning("Primary model unavailable, switching to fallback")
            self.client = self._init_fallback_client()
            try:
                return await self.send_prompt(prompt, buffer)
            except Exception as e:
                self.logger.exception(f"Both primary and fallback models failed: {str(e)}")
                raise RuntimeError("All available models failed")

    async def _process_stream(self, stream: Any, buffer: ResponseBuffer) -> str:
        current_content = {"reasoning": "", "response": ""}
        paragraph_break = "\n\n"
        min_content_length = 120
        
        in_thinking_mode = False
        full_response = buffer.full_response if buffer.full_response else ""

        try:
            async for chunk in stream:
                if not isinstance(chunk, ChatCompletionChunk) or not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue

                if hasattr(delta, 'reasoning') or hasattr(delta, 'reasoning_content'):
                    reasoning = getattr(delta, 'reasoning', '') or getattr(delta, 'reasoning_content', '')
                    if reasoning:
                        if not in_thinking_mode:
                            in_thinking_mode = True
                            header = f"=== Thinking Process ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==="
                            self.logger.stream_info(header)
                            full_response += f"<think>{header}\n"
                            
                        current_content["reasoning"] += reasoning
                        # Only flush content if it contains a paragraph break or is long enough
                        if paragraph_break in current_content["reasoning"] or len(current_content["reasoning"]) >= min_content_length:
                            formatted_reasoning = current_content["reasoning"].strip()
                            self.logger.stream_info(f"  {formatted_reasoning}")
                            full_response += f"  {formatted_reasoning}\n"
                            current_content["reasoning"] = ""

                if hasattr(delta, 'content') and delta.content:
                    if in_thinking_mode:
                        in_thinking_mode = False
                        footer = f"=== Analysis Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==="
                        self.logger.stream_info(footer)
                        full_response += f"</think>\n{footer}\n"
                        
                    current_content["response"] += delta.content
                    # Only flush content if it contains a paragraph break or is long enough
                    if paragraph_break in current_content["response"] or len(current_content["response"]) >= min_content_length:
                        formatted_response = current_content["response"].strip()
                        self.logger.stream_info(f"  {formatted_response}")
                        full_response += f"  {formatted_response}\n"
                        current_content["response"] = ""

        except Exception as e:
            self.logger.error(f"Error processing stream: {str(e)}")
            raise

        # Process any remaining content
        if current_content["reasoning"].strip():
            formatted_reasoning = current_content["reasoning"].strip()
            self.logger.stream_info(f"  {formatted_reasoning}")
            full_response += f"  {formatted_reasoning}\n"
            
        if current_content["response"].strip():
            formatted_response = current_content["response"].strip()
            self.logger.stream_info(f"  {formatted_response}")
            full_response += f"  {formatted_response}\n"
            
        if in_thinking_mode:
            footer = f"=== Analysis Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==="
            self.logger.stream_info(footer)
            full_response += f"</think>\n{footer}\n"
            
        buffer.full_response = full_response
        return full_response