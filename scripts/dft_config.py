from typing import Any, Dict, List, NewType, Optional, Tuple
from transformers import TrainingArguments
from dataclasses import dataclass

@dataclass
class DFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`DFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the
            default data collator.
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_prompt_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the target. This argument is required if you want to use the default data collator and
            your model is an encoder-decoder.
        is_encoder_decoder(`Optional[int]`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        ref_model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
            from a string.
        model_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        u_init (`float`, *optional*, defaults to `0`):
            Initial value of u estimator (stored in log u).
        tau (`float`, *optional*, defaults to `1.0`):
            Temperature.
        samples_per_prompt (`int`, *optional*, defaults to `1`):
            Number of samples per prompt, concatenated batch size will be batch_size * (samples_per_prompt + 1).
        gamma (`float`, *optional*, defaults to `0.95`):
            Parameter in updating the u estimater.
        ref_free (`bool`, *optional*, defaults to `False`):
            Whether to use DFT2.
        sample (`bool`, *optional*, defaults to `True`):
            Whether to sample different set of negative data at each epoch.
        precompute_offline_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Precompute the logp for the reference model. A `logps.pt` file will be saved to output_dir.
        probs_dir (`Optional[str]`, *optional*, defaults to `None`):
            The path to the logps.pt file, if you want to use precomputed reference model log probabilities for training.
            This is useful when training without the reference model to reduce the total GPU memory needed.

    """
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict[str, Any]] = None
    ref_model_init_kwargs: Optional[Dict[str, Any]] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    u_init: float = 0
    tau: float = 1.0
    samples_per_prompt : int = 1
    gamma: float = 0.95
    ref_free: bool = False
    sample: bool = True
    precompute_offline_ref_log_probs : bool = False
    probs_dir: Optional[str] = None
    sft : bool = False