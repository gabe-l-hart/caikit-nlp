# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implementation of the TextGeneration task backed by the llama.cpp library
"""
# Standard
from typing import Optional, Union
import os
import shutil

# Third Party
try:
    from llama_cpp import Llama
    HAVE_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAVE_LLAMA_CPP = False

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.nlp.data_model import FinishReason, GeneratedTextResult
from caikit.interfaces.nlp.tasks import  TextGenerationTask
import alog

log = alog.use_channel("LMA_CPP")
error = error_handler.get(log)


@module(
    id="2006dfdc-bf52-4109-b3ed-5a1365b60753",
    name="Text Generation llama.cpp",
    version="0.1.0",
    task=TextGenerationTask,
)
class TextGenerationLlamaCpp(ModuleBase):
    __doc__ = __doc__

    # Mapping of known stop strings to finish reason enum
    _FINISH_REASON_MAP = {
        "length": FinishReason.MAX_TOKENS,
        "stop": FinishReason.STOP_SEQUENCE,
    }

    def __init__(
        self,
        model: Llama,
        system_prompt: Optional[str] = None,
        system_prompt_label: str = "system",
        user_prompt_label: str = "user",
        assistant_prompt_label: str = "assistant",
        prompt_label_template: str = "<|{}>",
    ):
        """Initialize with an in-memory model

        Args:
            model (Llama): The in-memory llama.cpp model
            system_prompt (Optional[str]): The static prompt string for all
                inference requests
            system_prompt_label (str): The label used to delineate the system
                prompt text from other parts of the prompt
            user_prompt_label (str): The label used to delineate the user prompt
                text from other parts of the prompt
            assistant_prompt_label (str): The label used to delineate the
                assistant's response text from other parts of the prompt
            prompt_label_template (str): The string template for encoding the
                prompt labels into a single prompt string
        """
        self._verify_llama_cpp()
        self._model = model
        self._system_prompt = system_prompt
        self._system_prompt_label = prompt_label_template.format(system_prompt_label)
        self._user_prompt_label = prompt_label_template.format(user_prompt_label)
        self._assistant_prompt_label = prompt_label_template.format(assistant_prompt_label)
        self._prompt_label_template = prompt_label_template

        # Pre-construct the prompt prefix so that this is only done once
        # NOTE: Retain input args for saving
        prompt_pfx_parts = []
        if self._system_prompt:
            log.debug3("Using system prompt: %s", self._system_prompt)
            prompt_pfx_parts.append(self._system_prompt_label + self._system_prompt)
        prompt_pfx_parts.append(self._user_prompt_label)
        self._prompt_prefix = "\n".join(prompt_pfx_parts)

    ###############
    ## Save/Load ##
    ###############

    def save(self, model_path: str):
        """Save as a caikit model to the given path

        Args:
            model_path (str): Folder to save text-generation caikit model
        """
        error.value_check(
            "<NLP64227258E>",
            os.path.exists(self._model.model_path),
            "Cannot save llama.cpp model without the original model file",
        )
        model_file_name = os.path.basename(self._model.model_path)
        saver = ModuleSaver(self, model_path=model_path)
        with saver:
            artifacts_dir, artifacts_dir_abs = saver.add_dir("artifacts")
            target_model_file = os.path.join(artifacts_dir, model_file_name)
            saver.update_config({
                "model_file": target_model_file,
                "system_prompt": self._system_prompt,
                "system_prompt_label": self._system_prompt_label,
                "user_prompt_label": self._user_prompt_label,
                "assistant_prompt_label": self._assistant_prompt_label,
                "prompt_label_template": self._prompt_label_template,
            })
            shutil.copyfile(
                self._model.model_path,
                os.path.join(artifacts_dir_abs, model_file_name),
            )

    @classmethod
    def load(
        cls, model_path: Union[str, ModuleConfig], **kwargs
    ) -> "TextGenerationLlamaCpp":
        """Load a caikit model from the given path

        Args:
            model_path (Union[str, ModuleConfig]): The path on disk, or
                pre-parsed config for the model to load
            **kwargs: Additional kwargs to pass to the Llama model init

        Returns:
            model (TextGenerationLlamaCpp): The loaded caikit model instance
        """
        cls._verify_llama_cpp()
        config = ModuleConfig.load(model_path)
        model_file_path = os.path.join(config.model_path, config.model_file)
        with alog.ContextTimer(
            log.debug, "Done loading model %s in: ", model_file_path,
        ):
            kwargs.setdefault("verbose", False)
            model = Llama(model_file_path, **kwargs)
        return cls(
            model=model,
            system_prompt=config.system_prompt,
            system_prompt_label=config.system_prompt_label,
            user_prompt_label=config.user_prompt_label,
            assistant_prompt_label=config.assistant_prompt_label,
            prompt_label_template=config.prompt_label_template,
        )

    ###############
    ## Inference ##
    ###############

    @TextGenerationTask.taskmethod(input_streaming=False, output_streaming=False)
    def run(
        self,
        text: str,
        raw: bool = False,
        max_new_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        preserve_input_text: bool = True,
        **kwargs,
    ) -> GeneratedTextResult:
        """Run the model as a unary-unary inference

        Args:
            text (str): The user prompt text
            raw (bool): If True, run without the configured system prompt or
                labels
            max_new_tokens (Optional[int]): Max tokens to produce
            top_k (Optional[int]): The number of highest probability vocabulary
                tokens to keep for top-k-filtering.
                Default: 0 - means disabled
            top_p (Optional[int]): If set to float < 1, only the smallest set of
                most probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
                Default: 1.0 - means disabled - 0.0 equivalent to 1.0
            typical_p (Optional[float]): Local typicality measures how similar
                the conditional probability of predicting a target token next is
                to the expected conditional probability of predicting a random
                token next, given the partial text already generated. If set to
                float < 1, the smallest set of the most locally typical tokens
                with probabilities that add up to typical_p or higher are kept
                for generation.
                Default: 1.0 - means disabled - 0.0 equivalent to 1.0
            temperature (Optional[float]): The value used to modulate the next
                token probabilities.
                Default: 1.0 - means disabled - equivalent to 1.0
            repetition_penalty (Optional[float]): The more a token is used
                within generation the more it is penalized to not be picked in
                successive generation passes.
                Default: 1.0 - means no penalty - 0.0 equivalent to 1.0
            preserve_input_text (bool): Echo the input user text as part of the
                response
            **kwargs: Additional keyword args to pass to Llama.__call__

        Returns:
            generated_text (GeneratedTextResult): The resulting generated text
                and generation information
        """
        # Create the full prompt
        prompt = self._make_prompt(text, raw)
        log.debug4("Full prompt: %s", prompt)

        # Map these keywords to the kwargs of the model
        run_kwargs = kwargs
        if max_new_tokens is not None:
            run_kwargs["max_tokens"] = max_new_tokens
        if top_k is not None:
            run_kwargs["top_k"] = top_k
        if top_p is not None:
            run_kwargs["top_p"] = top_p
        if typical_p is not None:
            run_kwargs["typical_p"] = typical_p
        if temperature is not None:
            run_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            run_kwargs["repeat_penalty"] = repetition_penalty
        run_kwargs["echo"] = preserve_input_text

        # Run the inference
        result = self._model(prompt, stream=False, **run_kwargs)

        # Convert the output to the data model
        result_text_choices = result.get("choices")
        error.value_check(
            "<NLP14950347E>",
            (
                result_text_choices and
                len(result_text_choices) == 1 and
                "text" in result_text_choices[0]
            ),
            "Got unexpected output choices shape",
        )
        generated_text = result_text_choices[0]["text"]
        usage = result.get("usage", {})
        error.type_check("<NLP00719552E>", dict, usage=usage)
        generated_tokens = usage.get("completion_tokens")
        input_token_count = usage.get("prompt_tokens")
        finish_reason = self._FINISH_REASON_MAP.get(
            result_text_choices[0].get("finish_reason")
        )
        seed = kwargs.get("seed")
        return GeneratedTextResult(
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
            seed=seed,
        )

    ##########
    ## Impl ##
    ##########

    def _make_prompt(self, user_prompt: str, raw: bool = False) -> str:
        """Shared logic for making the final prompt string"""
        if raw:
            return user_prompt
        return f"{self._prompt_prefix}{user_prompt}\n{self._assistant_prompt_label}"

    @classmethod
    def _verify_llama_cpp(cls):
        error.value_check(
            "<NLP70441781E>",
            HAVE_LLAMA_CPP,
            "Please install caikit-nlp[llama-cpp] to use %s",
            cls.__name__,
        )
