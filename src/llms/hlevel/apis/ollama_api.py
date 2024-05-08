from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from src.llms.hlevel.base import LLMBase
from typing import Dict, Any, Union, Optional, Sequence, Literal
from ollama import Client
from ollama._types import Message, Options, RequestError, ResponseError

MAX_RETRY_ATTEMPTS = 5
MIN_WAIT = 30  # Seconds
MAX_WAIT = 300  # Seconds


def custom_retry_error_callback(retry_state):
    print("OpenAi Exception:", retry_state.outcome.exception())
    return {"error": "ERROR_OPENAI", "message": "Open ai exception: " + str(retry_state.outcome.exception())}


class OllamaLLM(LLMBase):
    def __init__(
            self,
            model: str,
            ollama_host: str,
            ollama_port: int,
            api_key: str = "ollama_tools",
            temperature: float = 0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
    ):
        """
        Args:
            api_key (str): The OpenAI API key.
            model (str): The model.
            temperature (float): The temperature.
            max_tokens (int): The maximum number of tokens.
            top_p (float): The top p.
            frequency_penalty (float): The frequency penalty.
            presence_penalty (float): The presence penalty.
            number_of_results (int): The number of results.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_key = api_key
        self.client = Client(host=f'{ollama_host}:{ollama_port}')

    def get_source(self):
        return "ollama_api"

    @retry(
        retry=(retry_if_exception_type(ConnectionError) | retry_if_exception_type(RequestError)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: print(
            f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion_text(
            self,
            max_tokens: int = 2048,
            messages: Optional[Sequence[Message]] = None,
            format: Literal['', 'json'] = '',
            options: Optional[Options] = None,
            keep_alive: Optional[Union[float, str]] = None,
            **kwargs: Any
    ) -> Dict:
        """
        Call the OpenAI chat completion API.

        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            dict: The response.
            :param keep_alive:
            :param options:
            :param max_tokens:
            :param messages:
            :param format:
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                format=format,
                keep_alive=keep_alive
            )
            content = response.get('message')
            return {"response": response, "content": content}
        except Exception as exception:
            return {"error": "OLLAMA_CALLING_ERROR", "message": "OLLAMA_CALLING_EXCEPTION: " + str(exception)}

    def chat_completion_stream(
            self,
            messages: Optional[Sequence[Message]] = None,
            format: Literal['', 'json'] = '',
            options: Optional[Options] = None,
            keep_alive: Optional[Union[float, str]] = None,
            **kwargs: Any
    ):
        """
        Call the OpenAI chat completion API.
        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            Stream response
            :param messages:
            :param format:
        """
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                format=format,
                keep_alive=keep_alive,
                stream=True
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
        except Exception as exception:
            return {"error": "OLLAMA_CALLING_ERROR", "message": "OLLAMA_CALLING_EXCEPTION: " + str(exception)}

    @retry(
        retry=(retry_if_exception_type(ConnectionError) | retry_if_exception_type(RequestError)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: print(
            f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion_json(
            self,
            messages: Optional[Sequence[Message]] = None,
            format: Literal['', 'json'] = 'json',
            options: Optional[Options] = None,
            keep_alive: Optional[Union[float, str]] = None,
            **kwargs: Any
    ) -> Dict:
        """
        :param format:
        :param function_format:
        :param function_call:
        :param max_tokens:
        :param messages:
        :param prompt:
        :return:
        """
        return self.chat_completion_text(
            messages=messages,
            format='json',
            options=options,
            keep_alive=keep_alive)

    def get_api_key(self):
        """
        Returns:
            str: The API key.
        """
        return self.api_key

    def get_model(self):
        """
        Returns:
            str: The model.
        """
        return self.model

    def get_models(self):
        """
        Returns:
        list: The models.
        """
        return self.model

    def verify_access_key(self):
        return True


if __name__ == "__main__":
    import pprint

    llm = OllamaLLM(
        model='llama2',
        ollama_host="http://gcn008.csns.ihep.ac.cn",
        ollama_port=60000,
    )
    pprint.pprint(llm.chat_completion_json(
        messages=[{
            'role': 'user',
            'content': 'Why is the sky blue?'}]
    ))
