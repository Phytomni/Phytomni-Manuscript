from llm import FunctionModel


class PureModel:
    def __init__(self, model: FunctionModel, prxoy: dict[str]):
        self.model_name = model.name
        self.model_func = model.value

    def infer(self, query: str):
        self.model

