from __future__ import annotations

from config import ServeConfig
from generation import strip_stop_strings


class TransformersChatBackend:
    def __init__(self, config: ServeConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers serving fallback requires `transformers` and `torch` to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, trust_remote_code=config.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            trust_remote_code=config.trust_remote_code,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and not getattr(generation_config, "do_sample", False):
            # Avoid noisy warnings from sampling-only fields when we are doing greedy decode.
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = None
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = None
        self.backend_name = "transformers"
        self.seed = config.seed

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        stop_strings: list[str] | None = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        effective_stop_strings = list(stop_strings or [])
        stop_token_ids: list[int] = []
        for stop in effective_stop_strings:
            token_id = self.tokenizer.convert_tokens_to_ids(stop)
            if token_id is None or token_id == self.tokenizer.unk_token_id or token_id < 0:
                continue
            stop_token_ids.append(int(token_id))
        eos_token_id = stop_token_ids or self.tokenizer.eos_token_id
        generate_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            repetition_penalty=max(repetition_penalty, 1.0),
            no_repeat_ngram_size=max(no_repeat_ngram_size, 0),
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            tokenizer=self.tokenizer,
        )
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_stop_strings(text, effective_stop_strings)
