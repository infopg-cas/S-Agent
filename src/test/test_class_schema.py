from pydantic import BaseModel, create_model, Field, PydanticUserError
from typing import Callable, Dict, Type, Any

# Your Tool class
class Tool:
    def __init__(self, name: str, description: str, func: Callable) -> None:
        self.name = name
        self.description = description
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)

# Custom type definition for the Tool class using a simpler approach
class ToolType:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> 'Tool':
        if not isinstance(value, Tool):
            raise TypeError('ToolType expected')
        return value

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Dict[str, Any], handler):
        # field_schema = handler(schema)
        # field_schema.update()
        field_schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'description': {'type': 'string'},
                'func': {'type': 'string'}
            },
            'required': ['name', 'description']
        }
        return field_schema

# Example LLMBase class
class LLMBase:
    pass

# Your field data
fields = {
    'agent_name': (str, None),
    'llm': (Type[LLMBase], None),
    'actions': (Dict[str, ToolType], None),
    'prompt': (str, None)
}

# Define a custom model configuration
class Config:
    arbitrary_types_allowed = True

# Create the dynamic model with custom ToolType
DynamicClass = create_model(
    'DynamicClass',
    **fields,
    __config__=Config
)

# Generate the JSON schema
try:
    schema = DynamicClass.model_json_schema()
    print(schema)
except PydanticUserError as exc_info:
    assert exc_info.code == 'invalid-for-json-schema'
