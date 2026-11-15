"""Read the official GPT-2 FineWeb shards without loading them all into RAM."""

from pathlib import Path

import numpy as np
import torch

DATASET_REPO = "kjj0/fineweb10B-gpt2"
HEADER_BYTES = 256 * 4


def shard_info(path):
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=256)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"Invalid FineWeb shard header: {path}")
    count = int(header[2])
    if count <= 0 or path.stat().st_size != HEADER_BYTES + 2 * count:
        raise ValueError(f"FineWeb shard size does not match token count: {path}")
    return count


class TokenStream:
    """Sequential shard traversal matching upstream's boundary/tail behavior.

    A step is a contiguous global batch. Microbatching is applied afterward,
    so changing device memory or microbatch size cannot change the token stream.
    Exhaustion is an error, never an implicit wrap back to earlier training data.
    """
    def __init__(self, pattern, batch_tokens, seq_len, vocab_size=50304):
        import glob

        self.files = [Path(p) for p in sorted(glob.glob(str(pattern)))]
        if not self.files:
            raise FileNotFoundError(f"No FineWeb shards match {pattern}; use --download-shards N")
        self.counts = [shard_info(p) for p in self.files]
        if batch_tokens <= 0 or seq_len <= 0 or batch_tokens % seq_len:
            raise ValueError("batch_tokens must be a positive multiple of seq_len")
        self.batch_tokens, self.seq_len = batch_tokens, seq_len
        self.vocab_size = vocab_size
        # Upstream advances shards when pos + batch + 1 >= len(tokens).
        self.capacity = sum(max(0, (n - 2) // batch_tokens) for n in self.counts)
        self.reset()

    def reset(self):
        self.file_index, self.position = 0, 0
        self.tokens = None

    def next_batch(self):
        while self.file_index < len(self.files):
            if self.position + self.batch_tokens + 1 >= self.counts[self.file_index]:
                self.file_index += 1
                self.position, self.tokens = 0, None
                continue
            if self.tokens is None:
                self.tokens = np.memmap(self.files[self.file_index], mode="r", dtype="<u2",
                                        offset=HEADER_BYTES)
            buf = np.array(self.tokens[self.position:self.position + self.batch_tokens + 1],
                           dtype=np.int64)
            self.position += self.batch_tokens
            if buf.max() >= self.vocab_size:
                raise ValueError("Shard token ID exceeds configured vocabulary")
            data = torch.from_numpy(buf)
            return (data[:-1].view(-1, self.seq_len), data[1:].view(-1, self.seq_len))
        raise RuntimeError("FineWeb training shards exhausted; download more shards")


def download_shards(directory, count):
    """Explicit opt-in download of N training shards plus the validation shard."""
    if not 1 <= count <= 103:
        raise ValueError("download-shards must be between 1 and 103")
    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, count + 1)]
    for name in names:
        path = directory / name
        if not path.exists():
            hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=name,
                            local_dir=str(directory))
        shard_info(path)


class SyntheticStream:
    """Deterministic offline plumbing check; never a benchmark data source."""
    def __init__(self, batch_tokens, seq_len, vocab_size, seed):
        self.batch_tokens, self.seq_len, self.vocab_size = batch_tokens, seq_len, vocab_size
        self.seed = seed
        self.capacity = float("inf")
        self.reset()

    def reset(self):
        self.generator = torch.Generator().manual_seed(self.seed)

    def next_batch(self):
        data = torch.randint(self.vocab_size, (self.batch_tokens + 1,), generator=self.generator)
        return data[:-1].view(-1, self.seq_len), data[1:].view(-1, self.seq_len)
