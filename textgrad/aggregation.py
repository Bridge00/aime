from .engine import get_engine
from openai import OpenAI
import os
class Aggregator:

    def __init__(self, eval_sys_prompt = None, method = 'concatenate', engines = None) -> None:
        super().__init__()
        if engines is not None:
            self.engines = [get_engine(e) for e in engines]
        self.method = method
        self.evaluation_system_prompt = eval_sys_prompt

    def aggregate(self, responses, input_variable = None):


        return ''.join(responses)

