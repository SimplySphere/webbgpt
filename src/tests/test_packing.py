from data.packing import pack_token_sequences


def test_pack_token_sequences_pads_to_length():
    sequences = [[1, 2, 3], [4, 5]]
    packed = pack_token_sequences(sequences, sequence_length=6, pad_token_id=0, eos_token_id=9)
    assert all(len(row) == 6 for row in packed)
    assert packed[0][-1] == 0


def test_pack_token_sequences_splits_long_documents():
    sequences = [[1, 2, 3, 4, 5, 6, 7]]
    packed = pack_token_sequences(sequences, sequence_length=4, pad_token_id=0, eos_token_id=9)
    assert len(packed) == 2

