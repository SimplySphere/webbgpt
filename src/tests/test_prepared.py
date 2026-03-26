from data.prepared import encode_sft_messages


class FakeTokenizer:
    def __init__(self):
        self._special_ids = {
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "<|assistant|>": 4,
            "<|user|>": 5,
            "<|system|>": 6,
            "<|tool|>": 7,
        }
        self._next_id = 100
        self._vocab: dict[str, int] = {}

    def token_to_id(self, token: str) -> int:
        return self._special_ids[token]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens: list[int] = [self._special_ids["<s>"]] if add_bos else []
        index = 0
        specials = sorted(self._special_ids.keys(), key=len, reverse=True)
        while index < len(text):
            matched = False
            for special in specials:
                if text.startswith(special, index):
                    tokens.append(self._special_ids[special])
                    index += len(special)
                    matched = True
                    break
            if matched:
                continue
            char = text[index]
            if char.isspace():
                tokens.append(9)
                index += 1
                continue
            token = []
            while index < len(text) and not text[index].isspace():
                if any(text.startswith(special, index) for special in specials):
                    break
                token.append(text[index])
                index += 1
            piece = "".join(token)
            if piece not in self._vocab:
                self._vocab[piece] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[piece])
        if add_eos:
            tokens.append(self._special_ids["</s>"])
        return tokens


def test_encode_sft_messages_masks_non_assistant_tokens():
    tokenizer = FakeTokenizer()
    input_ids, labels = encode_sft_messages(
        [
            {"role": "system", "content": "You are WebbGPT."},
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning, Harry."},
        ],
        tokenizer,
        sequence_length=64,
    )
    first_labeled_index = next(index for index, value in enumerate(labels) if value != -100)
    assistant_prefix = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
    assistant_token_positions = [index for index, value in enumerate(input_ids) if value == tokenizer.token_to_id("<|assistant|>")]
    assert assistant_token_positions
    assert first_labeled_index >= assistant_token_positions[0] + len(assistant_prefix)
    assert any(value != -100 for value in labels[first_labeled_index:])
    assert all(value == -100 for value in labels[:assistant_token_positions[0] + len(assistant_prefix)])
