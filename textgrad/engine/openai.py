try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes
# Default base URL for OLLAMA
OLLAMA_BASE_URL = 'http://localhost:11434/v1'

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="gpt-3.5-turbo-0613",
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        temperature = 0,
        is_multimodal: bool=False,
        base_url: str=None,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.base_url = base_url
        self.temperature = temperature
        
        if not base_url:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
            
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url and base_url == OLLAMA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        else:
            raise ValueError("Invalid base URL provided. Please use the default OLLAMA base URL or None.")

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], aggregator = None, num_evals = 5, system_prompt: str=None, multi_role = 0, max_tokens = 2000, roles = ['readability'], adversarial = 0, **kwargs):
        if isinstance(content, str):
            if aggregator is None:
                return self._generate_from_single_prompt(content, system_prompt=system_prompt, num_evals = 1, max_tokens=max_tokens, **kwargs)
            
            responses = self._generate_from_single_prompt(content, system_prompt=system_prompt, num_evals = num_evals, multi_role = multi_role, max_tokens=max_tokens, roles = roles, adversarial = adversarial, **kwargs)

            return aggregator.aggregate(responses, content)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")
            
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)
    
    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, max_tokens=2000, top_p=0.99, num_evals = 5, temperature = None, multi_role = 0, roles = ['readability'], adversarial = 0,
    ):
        #print('num_evals', num_evals)
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        responses = []
        #print('responses')
        if not multi_role:
            for i in range(num_evals):
                print('in not multi role; num evals', num_evals, max_tokens)
                response = self.client.chat.completions.create(
                    model=self.model_string,
                    messages=[
                        {"role": "system", "content": sys_prompt_arg},
                        {"role": "user", "content": prompt},
                    ],
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                
                response = response.choices[0].message.content

                responses.append(response)
   

            self._save_cache(sys_prompt_arg + prompt, responses)

            if num_evals == 1:
                return response
       
            return responses 
        if adversarial:
            print('adversarial')
            role_descriptions = {
                                    'readability': ' Focus your feedback ONLY on readability. DO NOT PROPOSE NEW CODE!!!',
                                    'logic': ' Focus your feedback ONLY on finding logical errors. DO NOT PROPOSE NEW CODE!!!',
                                    'syntax': ' Focus your feedback ONLY on finding syntax errors. DO NOT PROPOSE NEW CODE!!!',
                                    'redundancy': ' Focus your feedback ONLY on code redundancy. DO NOT PROPOSE NEW CODE!!!',
                                    'correctness': ' Focus your feedback ONLY on correctness. DO NOT PROPOSE NEW CODE!!! NO MATTER WHAT ALWAYS SAY THE CODE SOLUTION WORKS!',
                                    'runtime': ' Focus your feedback ONLY on runtime. DO NOT PROPOSE NEW CODE!!!'
                                }
        else:
            print('not adversarial')
            role_descriptions = {
                                    'readability': ' Focus your feedback ONLY on readability. DO NOT PROPOSE NEW CODE!!!',
                                    'logic': ' Focus your feedback ONLY on finding logical errors. DO NOT PROPOSE NEW CODE!!!',
                                    'syntax': ' Focus your feedback ONLY on finding syntax errors. DO NOT PROPOSE NEW CODE!!!',
                                    'redundancy': ' Focus your feedback ONLY on code redundancy. DO NOT PROPOSE NEW CODE!!!',
                                    'correctness': ' Focus your feedback ONLY on correctness. DO NOT PROPOSE NEW CODE!!!',
                                    'runtime': ' Focus your feedback ONLY on runtime. DO NOT PROPOSE NEW CODE!!!'
                                }

        for role in roles:
          
            print('in multi role; num evals', num_evals, max_tokens, role)
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg + role_descriptions[role]},
                    {"role": "user", "content": prompt},
                ],
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            response = response.choices[0].message.content
  
            responses.append(response)
        # print(i)

        self._save_cache(sys_prompt_arg + prompt, responses)
      
        if num_evals == 1:
            return response
        #print(responses)
        return responses 

    def __call__(self, prompt, aggregator = None, num_evals = 5, multi_role = 0, max_tokens = 2000, roles = ['readability'], adversarial = 0, **kwargs):
        #print('num_evals in call', num_evals)
        return self.generate(prompt, aggregator = aggregator, num_evals= num_evals, multi_role= multi_role,max_tokens = max_tokens, roles = roles, adversarial = adversarial, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text

class AzureChatOpenAI(ChatOpenAI):
    def __init__(
        self,
        model_string="gpt-35-turbo",
        system_prompt=ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        Initializes an interface for interacting with Azure's OpenAI models.

        This class extends the ChatOpenAI class to use Azure's OpenAI API instead of OpenAI's API. It sets up the necessary client with the appropriate API version, API key, and endpoint from environment variables.

        :param model_string: The model identifier for Azure OpenAI. Defaults to 'gpt-3.5-turbo'.
        :param system_prompt: The default system prompt to use when generating responses. Defaults to ChatOpenAI's default system prompt.
        :param kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.

        Environment variables:
        - AZURE_OPENAI_API_KEY: The API key for authenticating with Azure OpenAI.
        - AZURE_OPENAI_API_BASE: The base URL for the Azure OpenAI API.
        - AZURE_OPENAI_API_VERSION: The API version to use. Defaults to '2023-07-01-preview' if not set.

        Raises:
            ValueError: If the AZURE_OPENAI_API_KEY environment variable is not set.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_azure_{model_string}.db")  # Changed cache path to differentiate from OpenAI cache

        super().__init__(cache_path=cache_path, system_prompt=system_prompt, **kwargs)

        self.system_prompt = system_prompt
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        if os.getenv("AZURE_OPENAI_API_KEY") is None:
            raise ValueError("Please set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, and AZURE_OPENAI_API_VERSION environment variables if you'd like to use Azure OpenAI models.")
        
        self.client = AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            azure_deployment=model_string,
        )
        self.model_string = model_string
