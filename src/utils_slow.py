
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import (
Cache,
DynamicCache,
EncoderDecoderCache,
OffloadedCache,
QuantizedCacheConfig,
StaticCache,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
# from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.pytorch_utils import isin_mps_friendly
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils import (
ModelOutput,
is_accelerate_available,
is_hqq_available,
is_quanto_available,
is_torchdynamo_compiling,
logging,
)
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
AssistedCandidateGenerator,
# AssistedCandidateGeneratorDifferentTokenizers,
CandidateGenerator,
PromptLookupCandidateGenerator,
_crop_past_key_values,
_prepare_attention_mask,
_prepare_token_type_ids,
)
from transformers.generation.configuration_utils import (
NEED_SETUP_CACHE_CLASSES_MAPPING,
QUANT_BACKEND_CLASSES_MAPPING,
GenerationConfig,
GenerationMode,
)
from transformers.generation.logits_process import (
EncoderNoRepeatNGramLogitsProcessor,
EncoderRepetitionPenaltyLogitsProcessor,
EpsilonLogitsWarper,
EtaLogitsWarper,
ExponentialDecayLengthPenalty,
ForcedBOSTokenLogitsProcessor,
ForcedEOSTokenLogitsProcessor,
HammingDiversityLogitsProcessor,
InfNanRemoveLogitsProcessor,
LogitNormalization,
LogitsProcessorList,
MinLengthLogitsProcessor,
MinNewTokensLengthLogitsProcessor,
MinPLogitsWarper,
NoBadWordsLogitsProcessor,
NoRepeatNGramLogitsProcessor,
PrefixConstrainedLogitsProcessor,
RepetitionPenaltyLogitsProcessor,
SequenceBiasLogitsProcessor,
SuppressTokensAtBeginLogitsProcessor,
SuppressTokensLogitsProcessor,
TemperatureLogitsWarper,
TopKLogitsWarper,
TopPLogitsWarper,
TypicalLogitsWarper,
UnbatchedClassifierFreeGuidanceLogitsProcessor,
WatermarkLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
ConfidenceCriteria,
EosTokenCriteria,
MaxLengthCriteria,
MaxTimeCriteria,
StoppingCriteria,
StoppingCriteriaList,
StopStringCriteria,
)
from transformers.generation.utils import (
GenerateOutput,
GenerateNonBeamOutput,
GenerateEncoderDecoderOutput,
GenerateDecoderOnlyOutput,
_split_model_outputs

)


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

from .generator import LongAssistedCandidateGenerator
from transformers.generation import GenerationMixin


# class LongGenerationMixin(GenerationMixin):

def _get_candidate_generator(
    self,
    generation_config: GenerationConfig,
    input_ids: torch.LongTensor,
    inputs_tensor: torch.Tensor,
    assistant_model: "PreTrainedModel",
    logits_processor: LogitsProcessorList,
    target_tokenizer: "PreTrainedTokenizerBase",
    assistant_tokenizer: "PreTrainedTokenizerBase",
    assistant_input_ids: torch.LongTensor,
    model_kwargs: Dict,
) -> CandidateGenerator:
    """
    Returns the candidate generator to be used in `assisted_generation`
    """
    different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))

    if generation_config.prompt_lookup_num_tokens is not None:
        candidate_generator = PromptLookupCandidateGenerator(
            eos_token_id=generation_config._eos_token_tensor,
            num_output_tokens=generation_config.prompt_lookup_num_tokens,
            max_matching_ngram_size=generation_config.max_matching_ngram_size,
            max_length=generation_config.max_length,
        )
    elif assistant_input_ids is not None:
        candidate_generator = LongAssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_input_ids=assistant_input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )
    else:
        candidate_generator = AssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
        )
    return candidate_generator




@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    # speculative_margin=1,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    print("****************Using long speculative decoding**************")
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation
    assistant_input_ids = kwargs.pop("assistant_input_ids", None)  # only used for assisted generation
    speculative_margin = kwargs.pop("speculative_margin", 1)
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False
            
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    accepted_tokens = 0
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
            raise ValueError("assisted generate is not supported with Static cache classes`")
        if self._is_stateful:
            # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
            # which is not possible with stateful models (they can't reset to a previous subset of generated text)
            raise ValueError(
                f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            target_tokenizer=tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            assistant_input_ids=assistant_input_ids,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result, accepted_tokens = self._assisted_decoding(
            input_ids,
            assistant_input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            speculative_margin=speculative_margin,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.DOLA_GENERATION:
        if self._is_stateful:
            # DoLa decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )
        result = self._dola_decoding(
            input_ids,
            dola_layers=generation_config.dola_layers,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and hasattr(result.past_key_values, "to_legacy_cache")
        and result.past_key_values.to_legacy_cache is not None
    ):
        # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
        should_convert_cache = generation_config.return_legacy_cache
        is_user_defined_cache = user_defined_cache is not None
        is_default_cache_type = (
            type(result.past_key_values) == DynamicCache  # noqa E721
            or (
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
            )
        )
        if not is_user_defined_cache and is_default_cache_type:
            logger.warning_once(
                "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                "keep returning the legacy format, please set `return_legacy_cache=True`."
            )
            should_convert_cache = True
        if should_convert_cache:
            result.past_key_values = result.past_key_values.to_legacy_cache()
    return result, accepted_tokens




import concurrent.futures

def _assisted_decoding(
    self,
    input_ids: torch.LongTensor,
    assistant_input_ids: torch.LongTensor,
    candidate_generator: CandidateGenerator,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    speculative_margin: float,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
    **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
    candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
    models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        candidate_generator (`CandidateGenerator`):
            A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
            more information, the documentation of [`CandidateGenerator`] should be read.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    print("***********Using Cuton assistant decoding**********")
    # init values
    do_sample = generation_config.do_sample
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    accepted_tokens = 0
    this_peer_finished = False
    is_first_iteration = True  # to preserve the same API in the output as other generation methods
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        cur_len = input_ids.shape[-1]
        import time
        start_time = time.time()
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids, assistant_input_ids)
        end_time = time.time()
        #  1. Fetch candidate sequences from a `CandidateGenerator`
        # Mark: Modify Here
        

        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        print(f"Candidate Length: {candidate_length}, Time: {end_time - start_time}")
        # print()
        is_done_candidate = stopping_criteria(candidate_input_ids, None)

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
        )
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if "cache_position" in candidate_kwargs:
            candidate_kwargs["cache_position"] = torch.cat(
                (
                    candidate_kwargs["cache_position"],
                    torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                ),
                dim=0,
            )

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
        if "num_logits_to_keep" in model_inputs:
            model_inputs["num_logits_to_keep"] = candidate_length + 1

        # 2.2. Run a forward pass on the candidate sequence
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        start_time = time.time()
        outputs = self(**model_inputs)
        end_time = time.time()
        print(f"Target Model Forward Time: {end_time - start_time}")

        # 2.3. Process the new logits
        # .float() is needed to retain precision for later logits manipulations
        new_logits = outputs.logits[:, -candidate_length - 1 :].float()  # excludes the input prompt if present
        new_logits = new_logits.to(input_ids.device)
        next_token_logits = new_logits.clone()
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        # 3. Select the accepted tokens. There are two possible cases:
        # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
        # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
        if candidate_logits is not None: 
            print("Start Speculative Sampling")
            valid_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate,
                m=speculative_margin,
            )
        # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
        # original model logits with the candidate tokens. We can keep the candidate tokens until the first
        # mismatch, or until the max length is reached.
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            # Ensure we don't generate beyond max_len or an EOS token
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, : n_matches + 1]
        print("Current iteration matches: ", n_matches)
        accepted_tokens += n_matches
        # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.

        # 4.1. Get the valid continuation, after the matching tokens
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        assistant_input_ids = torch.cat((assistant_input_ids, valid_tokens.to(assistant_input_ids.device)), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]

        # 4.2. Discard past key values relative to unused assistant tokens
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        # 5. Update the candidate generation strategy if needed
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
            num_new_tokens=n_matches + 1,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.
        if return_dict_in_generate:
            newly_added_length = n_matches + 1
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(newly_added_length))
            if output_logits:
                raw_logits += tuple(next_token_logits[:, i, :] for i in range(newly_added_length))

            newly_added_length = new_cur_len if is_first_iteration else newly_added_length
            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, newly_added_length
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        newly_added_length,
                        is_decoder_attention=True,
                    )
                # some (V)LLMs have hard requirement on SDPA and thus never return attn
                elif outputs.attentions[0] is not None:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        newly_added_length,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.decoder_hidden_states, cur_len, newly_added_length
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, newly_added_length
                    )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        is_first_iteration = False

    if streamer is not None:
        streamer.end()

    if (
        hasattr(candidate_generator, "assistant_model")
        and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
    ):
        candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
            candidate_generator.num_assistant_tokens
        )
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids, accepted_tokens



def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
    m=1,
    ):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    # dual rejection sampling
    # is_accepted_lc = r_i <= probability_ratio 
    # # n_matches_naive = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    # print("M=", m)
    # rag_probability_ratio = q_i / (m*p_i)
    # r_i_rag = torch.rand_like(rag_probability_ratio)
    # is_accepted_rag = r_i_rag <= rag_probability_ratio
    # is_accepted = torch.logical_or(is_accepted_lc, is_accepted_rag)
    



    # log_q_i = q_i.log()
    # log_p_i = p_i.log()
    
    # for i in range(candidate_length)
    
    
    
    # span_size = 4
    # num_complete_spans = candidate_length // span_size
    # n_matches = 0

    # for span_idx in range(num_complete_spans):
    #     start = span_idx * span_size
    #     end = start + span_size
    #     q_i_span = q_i[start:end].prod(dim=-1)
    #     p_i_span = p_i[start:end].prod(dim=-1)
    #     span_probability_ratio = p_i_span / (m * q_i_span)  # Product over the spans
        
    #     r_i_span = torch.rand_like(span_probability_ratio)
    #     if r_i_span <= span_probability_ratio:
    #         n_matches += q_i[start:end].size(0)
    #     else:
    #         break  # Stop at the first rejection span
    # # print(n_matches)
    # is_accepted[:n_matches] = True
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    
    # def update_spans(is_accepted, span_size):
    #     is_accepted = is_accepted.clone()  # Clone to avoid modifying the original tensor
    #     for i in range(0, is_accepted.size(0), span_size):
    #         span = is_accepted[i:i + span_size]
    #         if span.sum().item() > 2:  # Count of True (as True is 1 and False is 0)
    #             is_accepted[i:i + span_size] = torch.ones_like(span, dtype=bool)
    #     return is_accepted

    # span_size = 4
    # is_accepted = update_spans(is_accepted, span_size)
    
    # print(is_accepted)
    # n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # print(updated_matches)
    # Output: [True, True, True, True, False, True, False, False]

    # real_n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    # tolerance_tokens = candidate_length//4
    # is_accepted[:tolerance_tokens] = True  # always accept the first tokens
    # tol_n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()
    
    # if tol_n_matches >= 3*candidate_length // 4:
    #     n_matches = tol_n_matches
    # else:
    #     n_matches = real_n_matches

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches




from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch





def get_max_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        return gpu_properties.total_memory
    else:
        return None

def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
    device_list="0"
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()

    max_gpu_memory = get_max_gpu_memory() / (1024 ** 3)
    device_list = [int(x) for x in device_list.split(",")]
    max_memory = {i: f"{max_gpu_memory}GiB" for i in device_list}
    llm =  AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        use_flash_attention_2=True, 
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True)
    # llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.95)#, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore