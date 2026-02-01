
import outlines
from transformers import AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Trying to reproduce the issue potentially caused by outlines + transformers
# The user reports: ValueError: You need to specify either `text` or `text_target`.

try:
    print("Loading model with outlines...")
    model = outlines.models.transformers(MODEL_ID)
    
    prompt = "Test prompt"
    
    # Simulating what the notebook does
    generator = outlines.generate.regex(
        model,
        r"([a-zA-Z ]+|NULL) \| ([a-zA-Z ]+|NULL) \| ([1-2][0-9][0-9][0-9]|NULL)",
        sampler=outlines.samplers.greedy(),
    )
    
    print("Generating...")
    # This matches the call structure in the notebook
    result = generator(prompt, max_tokens=30)
    print("Result:", result)

except Exception as e:
    print("\nCAUGHT EXCEPTION:")
    print(e)
    import traceback
    traceback.print_exc()
