from .utils import generate, _get_candidate_generator, _assisted_decoding
import transformers




def monkey_patch():
    # Monkey patching the LongGenerationMixin into the GPT2LMHeadModel class
    transformers.generation.utils.GenerationMixin.generate = generate
    transformers.generation.utils.GenerationMixin._get_candidate_generator = _get_candidate_generator
    transformers.generation.utils.GenerationMixin._assisted_decoding = _assisted_decoding