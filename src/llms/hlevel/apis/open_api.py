import openai
from openai import APIError, BadRequestError, RateLimitError, AuthenticationError, APITimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from scholarAI.llms.hlevel.base import LLMBase
from openai import OpenAI
from typing import Dict, List, AnyStr, Any
import json

MAX_RETRY_ATTEMPTS = 5
MIN_WAIT = 30  # Seconds
MAX_WAIT = 300  # Seconds


def custom_retry_error_callback(retry_state):
    print("OpenAi Exception:", retry_state.outcome.exception())
    return {"error": "ERROR_OPENAI", "message": "Open ai exception: " + str(retry_state.outcome.exception())}


class OpenAiLLM(LLMBase):
    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            temperature: float = 0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            number_of_results=1
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
        self.number_of_results = number_of_results
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://api.openai.com/v1")'
        # openai.api_base = "https://api.openai.com/v1"

    def get_source(self):
        return "openai"

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

    @retry(
        retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APITimeoutError)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),  # Maximum number of retry attempts
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: print(
            f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion_text(
            self,
            messages: List,
            max_tokens: int = 2048,
            **kwargs: Any
    ) -> Dict:
        """
        Call the OpenAI chat completion API.

        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            dict: The response.
        """
        try:
            # openai.api_key = get_config("OPENAI_API_KEY")
            response = self.client.chat.completions.create(
                n=self.number_of_results,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            content = response.choices[0].message.content
            return {"response": response, "content": content}
        except RateLimitError as api_error:
            print("OpenAi RateLimitError:", api_error)
        except APITimeoutError as timeout_error:
            print("OpenAi Timeout:", timeout_error)
        except AuthenticationError as auth_error:
            print("OpenAi AuthenticationError:", auth_error)
            return {"error": "ERROR_AUTHENTICATION",
                    "message": "Authentication error please check the api keys: " + str(auth_error)}
        except BadRequestError as invalid_request_error:
            print("OpenAi InvalidRequestError:", invalid_request_error)
            return {"error": "ERROR_INVALID_REQUEST",
                    "message": "Openai invalid request error: " + str(invalid_request_error)}
        except Exception as exception:
            print("OpenAi Exception:", exception)
            return {"error": "ERROR_OPENAI", "message": "Open ai exception: " + str(exception)}

    def chat_completion_stream(self):
        """
        Call the OpenAI chat completion API.
        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            Stream response
        """
        pass

    @retry(
        retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APITimeoutError)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),  # Maximum number of retry attempts
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: print(
            f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion_json(
            self,
            messages: List,
            function_format: List,
            function_call: AnyStr = 'auto',
            max_tokens: int = 2048,
            **kwargs: Any
    ) -> Dict:
        """
        :param function_format:
        :param function_call:
        :param max_tokens:
        :param messages:
        :param prompt:
        :return:
        """
        try:
            # openai.api_key = get_config("OPENAI_API_KEY")
            response = openai.chat.completions.create(
                n=self.number_of_results,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                functions=function_format,
                function_call=function_call,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            if response.choices[0].finish_reason == 'function_call':
                content = json.loads(response.choices[0].message.function_call.arguments)
            elif response.choices[0].finish_reason == 'stop':
                content = json.loads(response.choices[0].message.content.replace("```json\n", "").replace("`", ""))
            return {"response": response, "content": content}
        except RateLimitError as api_error:
            print("OpenAi RateLimitError:", api_error)
        except APITimeoutError as timeout_error:
            print("OpenAi Timeout:", timeout_error)
        except AuthenticationError as auth_error:
            print("OpenAi AuthenticationError:", auth_error)
            return {"error": "ERROR_AUTHENTICATION",
                    "message": "Authentication error please check the api keys: " + str(auth_error)}
        except BadRequestError as invalid_request_error:
            print("OpenAi InvalidRequestError:", invalid_request_error)
            return {"error": "ERROR_INVALID_REQUEST",
                    "message": "Openai invalid request error: " + str(invalid_request_error)}
        except Exception as exception:
            print("OpenAi Exception:", exception)
            return {"error": "ERROR_OPENAI", "message": "Open ai exception: " + str(exception)}

    def verify_access_key(self):
        """
        Verify the access key is valid.

        Returns:
            bool: True if the access key is valid, False otherwise.
        """
        try:
            models = self.client.models.list()
            return True
        except Exception as exception:
            print("OpenAi Exception:", exception)
            return False

    def get_models(self):
        """
        Get the models.
        Returns: list: The models.
        """
        try:
            models = self.client.models.list()
            models = [model.id for model in models.data]
            # models_supported = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-32k']
            # models = [model for model in models if model in models_supported]
            return models
        except Exception as exception:
            print("OpenAi Exception:", exception)
            return []


if __name__ == "__main__":
    open = OpenAiLLM(api_key='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')
    # print(open.get_models())
    import pprint
    pprint.pprint(open.chat_completion_text(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "why the sky is blue?"}
        ]))

