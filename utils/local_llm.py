from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class LocalLLM:
    def __init__(self, model_id="gpt2", max_new_tokens=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, prompt):
        result = self.generator(prompt)
        return result[0]['generated_text'].strip()
