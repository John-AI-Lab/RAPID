import transformers




def monkey_patch_longspec():
    # Monkey patching the LongGenerationMixin into the GPT2LMHeadModel class
    from .utils_dual_rej import generate, _get_candidate_generator, _assisted_decoding
    transformers.generation.utils.GenerationMixin.generate = generate
    transformers.generation.utils.GenerationMixin._get_candidate_generator = _get_candidate_generator
    transformers.generation.utils.GenerationMixin._assisted_decoding = _assisted_decoding
    
def monkey_patch_streamingllm():
    from .streamingllm import generate, _get_candidate_generator, _assisted_decoding
    transformers.generation.utils.GenerationMixin.generate = generate
    transformers.generation.utils.GenerationMixin._get_candidate_generator = _get_candidate_generator
    transformers.generation.utils.GenerationMixin._assisted_decoding = _assisted_decoding