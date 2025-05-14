import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
import math
import re

from trl.models import PreTrainedModelWrapper, create_reference_model
from dft_config import DFTConfig
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    add_bos_token_if_needed,
    add_eos_token_if_needed,
    cap_exp,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


def _tokenize(
    features: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    args: DFTConfig,
    processor: Optional[Callable] = None,
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, List]:
    """
    Tokenizes and processes a batch of input features using the provided tokenizer and processor.
    """
    batch = defaultdict(list)

    if model is None:
        prompt = features["prompt"]
        images = features.get("images", [None] * len(features["prompt"]))

        prompt_tokens = _process_prompt(prompt, processor, tokenizer, images)
        chosen_tokens = _process_answer(prompt, features["chosen"], processor, tokenizer, images)
        rejected_tokens_ls = []
        if args.sample:
            num_gen = int(args.samples_per_prompt * args.num_train_epochs)
        else:
            num_gen = int(args.samples_per_prompt)

        for i in range(num_gen):
            rejected_tokens = _process_answer(prompt, features[f"rejected_{i}"], processor, tokenizer, images)
            rejected_tokens_ls.append(rejected_tokens)
            prompt_len_input_ids = _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)
            
        for i in range(num_gen):
            prompt_tokens, chosen_tokens, rejected_tokens_ls[i] = _add_special_tokens(
                tokenizer, prompt_len_input_ids, prompt_tokens, chosen_tokens, rejected_tokens_ls[i]
            )

        _truncate_tokens(chosen_tokens, rejected_tokens_ls, prompt_tokens, args)

        _build_sequence_tokens(batch, chosen_tokens, args, "chosen")
        for i in range(num_gen):
            _build_sequence_tokens(batch, rejected_tokens_ls[i], args, f"rejected_{i}")

        _append_prompt_tokens_to_batch(batch, prompt_tokens)
        batch["ids"] = features["ids"]
        if "logps" in features:
            for logps in features["logps"]:
                batch["reference_chosen_ps"].append(logps[-1])
                batch["reference_rejected_ps"].append(logps[:-1])

    else:
        raise ValueError("_tokenize_encoder_decoder is not implemented for 1-to-n datasets.")

    return dict(batch)


def _process_prompt(
    prompts: List[str], processor: Optional[Callable], tokenizer: PreTrainedTokenizerBase, images: List[Optional[Any]]
) -> List[Dict[str, List[int]]]:
    """
    Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
    """
    if processor:
        processor_kwargs = (
            {"add_special_tokens": False} if "add_special_tokens" in inspect.signature(processor).parameters else {}
        )
        prompt_tokens = []
        for prompt, image in zip(prompts, images):
            tokens = processor(images=image, text=prompt, **processor_kwargs)
            tokens = {k: v[0] for k, v in tokens.items()}
            if not isinstance(tokens["input_ids"], list):
                tokens["input_ids"] = tokens["input_ids"].tolist()
                tokens["attention_mask"] = tokens["attention_mask"].tolist()
            prompt_tokens.append(tokens)
    else:
        prompt_tokens = [tokenizer(prompt, add_special_tokens=False) for prompt in prompts]
    return [{f"prompt_{k}": v for k, v in tokens.items()} for tokens in prompt_tokens]


def _process_answer(
    prompts: List[str],
    answers: List[str],
    processor: Optional[Callable],
    tokenizer: PreTrainedTokenizerBase,
    images: List[Optional[Any]],
) -> List[Dict[str, Any]]:
    return [
        _build_tokenized_answer(prompt, answer, image, processor=processor, tokenizer=tokenizer)
        for prompt, answer, image in zip(prompts, answers, images)
    ]


def _adjust_prompt_length(
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> List[int]:
    prompt_len_input_ids = []
    for p_tokens, c_tokens, r_tokens in zip(prompt_tokens, chosen_tokens, rejected_tokens):
        c_len = len(c_tokens["prompt_input_ids"])
        r_len = len(r_tokens["prompt_input_ids"])
        min_len = min(c_len, r_len)

        for k, v in p_tokens.items():
            p_tokens[k] = v[:min_len]

        num_diff_tokens = sum([a != b for a, b in zip(c_tokens["prompt_input_ids"], r_tokens["prompt_input_ids"])])
        num_diff_len = abs(c_len - r_len)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )
        prompt_len_input_ids.append(len(p_tokens["prompt_input_ids"]))
    
    return prompt_len_input_ids


def _add_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt_len_input_ids: List[int],
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    for i in range(len(prompt_tokens)):
        prompt_tokens[i], chosen_tokens[i], rejected_tokens[i] = add_bos_token_if_needed(
            tokenizer.bos_token_id,
            prompt_len_input_ids[i],
            prompt_tokens[i],
            len(chosen_tokens[i]["prompt_input_ids"]),
            chosen_tokens[i],
            len(rejected_tokens[i]["prompt_input_ids"]),
            rejected_tokens[i],
        )

        chosen_tokens[i], rejected_tokens[i] = add_eos_token_if_needed(
            tokenizer.eos_token_id, chosen_tokens[i], rejected_tokens[i]
        )
    return prompt_tokens, chosen_tokens, rejected_tokens


def _truncate_tokens(
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens_ls: List[List[Dict[str, List[int]]]],
    prompt_tokens: List[Dict[str, List[int]]],
    args: DFTConfig,
) -> None:
    """
    Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
    """
    if args.truncation_mode not in ["keep_start", "keep_end"]:
        raise ValueError(f"Invalid truncation mode: {args.truncation_mode}")

    for c_tokens, r_tokens_ls, p_tokens in zip(chosen_tokens, zip(*rejected_tokens_ls) , prompt_tokens):
        longer_response_length = len(c_tokens["input_ids"])
        for r_tokens in r_tokens_ls:
            longer_response_length = max(longer_response_length, len(r_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [c_tokens, *r_tokens_ls, p_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                if args.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: args.max_prompt_length]
                elif args.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-args.max_prompt_length :]

        # if that's still too long, truncate the response from the end
        for answer_tokens in [c_tokens, *r_tokens_ls]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: args.max_length - args.max_prompt_length]


def _build_sequence_tokens(
    batch: Dict[str, List[int]], tokens: List[Dict[str, List[int]]], args: DFTConfig, prefix: str
) -> None:
    for token in tokens:
        sequence_tokens = {f"{prefix}_{k}": token[f"prompt_{k}"] + token[k] for k in ["input_ids", "attention_mask"]}
        sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
        sequence_tokens[f"{prefix}_labels"][: len(token["prompt_input_ids"])] = [args.label_pad_token_id] * len(
            token["prompt_input_ids"]
        )
        sequence_tokens[f"{prefix}_len"] = len(sequence_tokens[f"{prefix}_input_ids"]) - len(token["prompt_input_ids"])
        for k, v in sequence_tokens.items():
            batch[k].append(v)


def _append_prompt_tokens_to_batch(batch: Dict[str, List[int]], prompt_tokens: List[Dict[str, List[int]]]) -> None:
    for p_tokens in prompt_tokens:
        for k, v in p_tokens.items():
            batch[k].append(v)


def _build_tokenized_answer(
    prompt: str,
    answer: str,
    images: Optional[List[Any]] = None,
    processor: Optional[Callable] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, Any]:
    """
    Build tokenized response, handling vision models and different tokenizers.
    """

    def tokenize(text, images=None):
        if processor:
            processor_kwargs = (
                {"add_special_tokens": False}
                if "add_special_tokens" in inspect.signature(processor).parameters
                else {}
            )
            tokenized = processor(images=images, text=text, **processor_kwargs)
            tokenized = {k: v[0] for k, v in tokenized.items()}
            if not isinstance(tokenized["input_ids"], list):
                tokenized["input_ids"] = tokenized["input_ids"].tolist()
                tokenized["attention_mask"] = tokenized["attention_mask"].tolist()
        else:
            tokenized = tokenizer(text, add_special_tokens=False)
        return tokenized

    full_tokenized = tokenize(prompt + answer, images)
    prompt_tokenized = tokenize(prompt, images)

    prompt_input_ids = prompt_tokenized["input_ids"]
    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    return_dict = {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "input_ids": answer_input_ids,
        "attention_mask": answer_attention_mask,
    }
    if "pixel_values" in full_tokenized:
        return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
    if "pixel_attention_mask" in full_tokenized:
        return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

    return return_dict


class DFTTrainer(Trainer):
    r"""
    Initialize DFTTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DFTConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name"
        ],
        custom_message="Deprecated positional argument(s) used in DFTTrainer, please use the DFTConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DFTConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
    ):
        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DFTTrainer/DFTConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DFTTrainer/DFTConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DFTTrainer/DFTConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if model is not None:
            self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        else:
            warnings.warn(
                "No model provided, cannot determine if it is a vision model. Setting is_vision_model to False."
            )
            self.is_vision_model = False

        if self.is_vision_model:
            self.processor = tokenizer
            self.tokenizer = tokenizer.tokenizer  # tokenizer is actually a processor at this point
        else:
            self.tokenizer = tokenizer

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_offline_ref_log_probs or args.probs_dir or args.ref_free:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DFTConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DFTConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_completion_length` in the DFTConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_completion_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = args.max_completion_length
        self.precompute_offline_ref_log_probs = args.precompute_offline_ref_log_probs
        self.ref_free = args.ref_free

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the DFTTrainer, the value you passed will override the one in the `DFTConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc
        self.tau = args.tau
        self.u_init = args.u_init
        self.gamma = args.gamma
        self.samples_per_prompt = args.samples_per_prompt
        self.eps = 1e-8
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            fn_kwargs = {
                "tokenizer": self.tokenizer,
                "args": args,
                "processor": self.processor if self.is_vision_model else None,
                "model": model if self.is_encoder_decoder else None,
            }

            from datasets.fingerprint import format_kwargs_for_fingerprint, format_transform_for_fingerprint, update_fingerprint
            from dataclasses import asdict
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), {
                "shard": train_dataset,
                "fn_kwargs": {**fn_kwargs, "args" : {k : v for k, v in asdict(args).items() if k != "local_rank"} },
                "batched" : True,
                "writer_batch_size" : 10,
            })
            new_fingerprint = update_fingerprint(train_dataset._fingerprint, format_transform_for_fingerprint(Dataset._map_single), kwargs_for_fingerprint)

            train_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=self.dataset_num_proc,
                writer_batch_size=10,
                desc="Tokenizing train dataset",
                new_fingerprint=new_fingerprint
            )
            
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    desc="Tokenizing eval dataset",
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # sogclr parameters
        self.u = torch.ones(len(train_dataset), device=self.accelerator.device) * self.u_init
        self.u_mask = torch.zeros(len(train_dataset), device=self.accelerator.device)

        # only compute reference logps
        if self.precompute_offline_ref_log_probs:
            self.chosen_logps = []
            self.reject_logps = []
            self.ids = []

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_offline_ref_log_probs or self.ref_free or args.probs_dir):
                raise ValueError(
                    "No reference model and model is not a Peft model."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        batch_rejected: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = batch["chosen_labels"].shape[1]
            for k in batch_rejected:
                if k.startswith("rejected") and k.endswith("_labels"):
                    max_length = max(max_length, batch_rejected[k].shape[1]) 
        else:
            max_length = batch["chosen_input_ids"].shape[1]
            for k in batch_rejected:
                if k.startswith("rejected") and k.endswith("_input_ids"):
                    max_length = max(max_length, batch_rejected[k].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch_rejected:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = re.sub(r"rejected_\d+", "concatenated", k)
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        if is_vision_model:
            concatenated_batch["pixel_values"] = torch.cat(
                [batch["prompt_pixel_values"], batch["prompt_pixel_values"]], dim=0
            )
            if "prompt_pixel_attention_mask" in batch:
                concatenated_batch["pixel_attention_mask"] = torch.cat(
                    [batch["prompt_pixel_attention_mask"], batch["prompt_pixel_attention_mask"]], dim=0
                )
        return concatenated_batch
    
    def dft_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        ids : torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        diff_generated = policy_rejected_logps / self.tau - reference_rejected_logps
        
        b = diff_generated.max(dim = 1).values
        weight = torch.exp(diff_generated - b[:, None]).detach()
        w = math.log(self.gamma) + torch.log(weight.mean(dim = 1)) + b
        if self.gamma == 1:
            u = w
        else:
            u = math.log(1.0 - self.gamma) + self.u[ids]
            u = torch.maximum(u, w) - F.logsigmoid(torch.abs(u - w))
        self.u[ids] = u.detach().clone()
        self.u_mask[ids] = 1

        custom_metrics = {}

        losses_ref = (torch.exp(diff_generated - u[:, None]).detach() * policy_rejected_logps).mean(dim = 1)
        losses = -policy_chosen_logps + losses_ref

        custom_metrics["u"] = u.detach().cpu().mean()
        custom_metrics["diff_generated"] = diff_generated.detach().mean().cpu()
        custom_metrics["losses_ref"] = losses_ref.detach().mean().cpu()
        custom_metrics["weight"] = torch.exp(diff_generated - u[:, None]).detach().mean().cpu()
        custom_metrics["ref_logps/chosen"] = reference_chosen_logps.detach().mean().cpu()
        custom_metrics["ref_logps/rejected"] = reference_rejected_logps.detach().mean().cpu()

        return losses, custom_metrics
    
    def dft_loss_ref_free(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ids : torch.LongTensor,
        policy_chosen_size : torch.LongTensor,
        policy_rejected_size : torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        policy_chosen_logps_avg = policy_chosen_logps / (policy_chosen_size + self.eps)
        policy_rejected_logps_avg = policy_rejected_logps / (policy_rejected_size + self.eps)

        diff_generated = policy_rejected_logps_avg / self.tau

        b = diff_generated.max(dim = 1).values
        weight = torch.exp(diff_generated - b[:, None]).detach()
        w = math.log(self.gamma) + torch.log(weight.mean(dim = 1)) + b
        if self.gamma == 1:
            u = w
        else:
            u = math.log(1.0 - self.gamma) + self.u[ids]
            u = torch.maximum(u, w) - F.logsigmoid(torch.abs(u - w))
        self.u[ids] = u.detach().clone()
        self.u_mask[ids] = 1

        custom_metrics = {}

        losses_ref = (torch.exp(diff_generated - u[:, None]).detach() * policy_rejected_logps_avg).mean(dim = 1)
        losses = -policy_chosen_logps_avg + losses_ref

        custom_metrics["logps_avg/chosen"] = policy_chosen_logps_avg.detach().mean().cpu()
        custom_metrics["logps_avg/rejected"] = policy_rejected_logps_avg.detach().mean().cpu()
        custom_metrics["u"] = u.detach().cpu().mean()
        custom_metrics["diff_generated"] = diff_generated.detach().mean().cpu()
        custom_metrics["losses_ref"] = losses_ref.detach().mean().cpu()
        custom_metrics["weight"] = torch.exp(diff_generated - u[:, None]).detach().mean().cpu()
        
        return losses, custom_metrics

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], batch_rejected: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            batch_rejected, 
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]


        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]


        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, size_completion[:len_chosen], size_completion[len_chosen:])

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        batch_rejected = {}
        if self.args.sample:
            e = int(self.state.epoch)
            st = e * self.args.samples_per_prompt
            ed = (e + 1) * self.args.samples_per_prompt
            for i in range(st, ed):
                for k in batch:
                    if k.startswith(f"rejected_{i}_"):
                        batch_rejected[k] = batch[k]
        else:
            st = 0
            ed = self.args.samples_per_prompt
            for i in range(ed):
                for k in batch:
                    if k.startswith(f"rejected_{i}_"):
                        batch_rejected[k] = batch[k]

        if self.precompute_offline_ref_log_probs:
            with torch.no_grad():
                with self.null_ref_context():
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                        self.model, batch, batch_rejected
                    )[:2]
            for x, id in enumerate(batch["ids"]):
                self.chosen_logps.append(reference_chosen_logps[x])
                self.reject_logps.append(reference_rejected_logps.view(self.samples_per_prompt, -1).transpose(0,1)[x])
            self.ids.append(torch.tensor(batch["ids"], device = self.accelerator.device))
            losses = torch.tensor([0.0], requires_grad=True) # no backwards
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}ref_logps/rejected"] = reference_rejected_logps.detach().mean().cpu()
            metrics[f"{prefix}ref_logps/chosen"] = reference_chosen_logps.detach().mean().cpu()
            return losses.mean(), metrics

        forward_output = self.concatenated_forward(model, batch, batch_rejected)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            policy_chosen_size,
            policy_rejected_size
        ) = forward_output[:7]

        if self.args.sft:
            return policy_nll_loss.mean(), metrics
        
        policy_rejected_logps = policy_rejected_logps.view(self.samples_per_prompt, -1).transpose(0,1)
        policy_rejected_size = policy_rejected_size.view(self.samples_per_prompt, -1).transpose(0,1)

        if not self.args.ref_free:
            # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
            if (
                "reference_chosen_ps" in batch
                and "reference_rejected_ps" in batch
                and (self.args.probs_dir)
            ):
                reference_chosen_logps = torch.tensor(batch["reference_chosen_ps"], device = self.accelerator.device)
                reference_rejected_logps = torch.tensor([seq[st : ed] for seq in batch["reference_rejected_ps"]], device = self.accelerator.device)
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                                self.model, batch, batch_rejected
                            )[:2]
                    else:
                        reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                            self.ref_model, batch, batch_rejected
                        )[:2]
                reference_rejected_logps = reference_rejected_logps.view(self.samples_per_prompt, -1).transpose(0,1)

            losses, custom_metrics = self.dft_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                batch["ids"]
            )
        else:
            losses, custom_metrics = self.dft_loss_ref_free(
                policy_chosen_logps,
                policy_rejected_logps,
                batch["ids"],
                policy_chosen_size,
                policy_rejected_size
            )

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        for k, v in custom_metrics.items():
            metrics[f"{prefix}{k}"] = v

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        Unlike the parent class, we don't use the `token` argument to mitigate security risks.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
    
    def _save_optimizer_and_scheduler(self, output_dir):
        res = super()._save_optimizer_and_scheduler(output_dir)
        import os
        torch.save(self.u.detach().cpu(), os.path.join(output_dir, "u.pt"))
        # it is bad to save model in the middle of an epoch
        # since u are not reduced yet
        return res
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        res = super()._load_optimizer_and_scheduler(checkpoint)
        import os
        if checkpoint is not None:
            self.u = torch.load(os.path.join(checkpoint, "u.pt")).to(self.accelerator.device)
        return res
