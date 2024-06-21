from typing import Callable, Any, Dict

class Tool:
    def __init__(
            self,
            name: str,
            description: str,
            func: Callable
    ) -> None:
        self.name = name
        self.description = description
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)

# Custom type definition for the Tool class
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
    def __modify_schema__(cls, field_schema: Dict[str, Any]):
        field_schema.update(
            type='object',
            properties={
                'name': {'type': 'string'},
                'description': {'type': 'string'}
            },
            required=['name', 'description']
        )
