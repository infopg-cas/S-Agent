from llama_cpp import Llama
from llama_cpp import LlamaGrammar

GPU_LAYERS = 4
GRAMMAR_PATH = "scholarAI/llms/hlevel/local/json.gbnf"

class LLMLoader:
    _instance = None
    _model = None
    _grammar = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, context_length):
        self.context_length = context_length

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = Llama(model_path="/app/local_model_path", n_ctx=self.context_length, n_gpu_layers=int(GPU_LAYERS))
            except Exception as e:
                print(str(e))
        return self._model

    @property
    def grammar(self):
        if self._grammar is None:
            try:
                self._grammar = LlamaGrammar.from_file(GRAMMAR_PATH)
            except Exception as e:
                print(str(e))
        return self._grammar