from __future__ import annotations

from collections.abc import Iterable


class PackedSequencePacker:
    def __init__(self, sequence_length: int, pad_token_id: int, eos_token_id: int, current: list[int] | None = None):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.current = list(current or [])

    def state_dict(self) -> dict[str, list[int]]:
        return {"current": list(self.current)}

    def load_state_dict(self, payload: dict[str, list[int]] | None) -> None:
        self.current = list((payload or {}).get("current", []))

    def push(self, tokens: list[int]):
        chunk = list(tokens)
        if not chunk:
            return
        if chunk[-1] != self.eos_token_id:
            chunk.append(self.eos_token_id)
        if len(chunk) > self.sequence_length:
            for start in range(0, len(chunk), self.sequence_length):
                window = chunk[start : start + self.sequence_length]
                if len(window) < self.sequence_length:
                    window = window + [self.pad_token_id] * (self.sequence_length - len(window))
                yield window
            return
        if len(self.current) + len(chunk) > self.sequence_length:
            current = self.current + [self.pad_token_id] * (self.sequence_length - len(self.current))
            self.current = []
            yield current
        self.current.extend(chunk)

    def finish(self):
        if self.current:
            current = self.current + [self.pad_token_id] * (self.sequence_length - len(self.current))
            self.current = []
            yield current


def iter_packed_token_sequences(
    token_sequences: Iterable[list[int]],
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
):
    packer = PackedSequencePacker(
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    for tokens in token_sequences:
        yield from packer.push(tokens)
    yield from packer.finish()


def pack_token_sequences(
    token_sequences: Iterable[list[int]],
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> list[list[int]]:
    return list(
        iter_packed_token_sequences(
            token_sequences,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    )
