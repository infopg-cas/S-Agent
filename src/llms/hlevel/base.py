from abc import ABC, abstractmethod


class LLMBase(ABC):
    @abstractmethod
    def chat_completion_text(self, prompt):
        pass

    # @abstractmethod
    # def chat_completion_stream(self, prompt):
    #     pass
    #
    # @abstractmethod
    # def chat_completion_json(self, prompt):
    #     pass

    # @abstractmethod
    # def chat_completion_function_call(self, prompt):
    #     pass

    @abstractmethod
    def get_source(self):
        pass

    @abstractmethod
    def get_api_key(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_models(self):
        pass

    @abstractmethod
    def verify_access_key(self):
        pass