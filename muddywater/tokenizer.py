from __future__ import annotations

import base64
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable


try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    spm = None

try:
    from tokenizers import Tokenizer as HFTokenizerImpl
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    HFTokenizerImpl = None


def split_text_by_utf8_bytes(text: str, max_bytes: int) -> list[str]:
    """Split text on Python-character boundaries without exceeding *max_bytes*.

    SentencePiece's ``max_sentence_length`` is measured in UTF-8 bytes, not
    Python characters.  Keeping this conversion explicit prevents a 4096
    character Chinese sample (roughly 12 KiB) from being silently skipped by
    a legacy 4192-byte limit.
    """

    max_bytes = int(max_bytes)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for char in text:
        char_bytes = len(char.encode("utf-8", errors="surrogatepass"))
        if char_bytes > max_bytes:
            raise ValueError(
                f"A single character needs {char_bytes} UTF-8 bytes, which exceeds "
                f"max_sentence_length={max_bytes}. Increase the byte limit."
            )
        if current and current_bytes + char_bytes > max_bytes:
            chunks.append("".join(current))
            current = []
            current_bytes = 0
        current.append(char)
        current_bytes += char_bytes
    if current:
        chunks.append("".join(current))
    return chunks


def prepare_sentencepiece_training_texts(
    texts: Iterable[str],
    max_sentence_length: int,
) -> tuple[list[str], dict[str, int]]:
    """Make every SentencePiece training sentence byte-safe and report it.

    Overlong source documents are chunked rather than entrusted to
    SentencePiece's silent skip path.  Empty inputs are the only rejected
    records and are counted explicitly.
    """

    max_sentence_length = int(max_sentence_length)
    if max_sentence_length <= 0:
        raise ValueError("max_sentence_length must be positive UTF-8 bytes")

    prepared: list[str] = []
    input_documents = 0
    accepted_documents = 0
    chunked_documents = 0
    rejected_documents = 0
    max_observed_bytes = 0
    input_utf8_bytes = 0
    for raw_text in texts:
        input_documents += 1
        text = str(raw_text)
        if not text or not text.strip():
            rejected_documents += 1
            continue
        byte_length = len(text.encode("utf-8", errors="surrogatepass"))
        input_utf8_bytes += byte_length
        max_observed_bytes = max(max_observed_bytes, byte_length)
        if byte_length <= max_sentence_length:
            prepared.append(text)
            accepted_documents += 1
            continue
        chunks = split_text_by_utf8_bytes(text, max_sentence_length)
        if "".join(chunks) != text:  # Defensive invariant: no corpus bytes disappear.
            raise AssertionError("UTF-8 sentence chunking failed to preserve the source text")
        prepared.extend(chunk for chunk in chunks if chunk)
        chunked_documents += 1

    output_utf8_bytes = sum(
        len(text.encode("utf-8", errors="surrogatepass")) for text in prepared
    )
    if output_utf8_bytes != input_utf8_bytes:
        raise AssertionError(
            "SentencePiece preprocessing changed the number of accepted UTF-8 bytes"
        )
    stats = {
        "max_sentence_length_utf8_bytes": max_sentence_length,
        "input_documents": input_documents,
        "accepted_documents": accepted_documents,
        "chunked_documents": chunked_documents,
        "rejected_documents": rejected_documents,
        "training_sentences": len(prepared),
        "accepted_training_sentences": len(prepared),
        "rejected_training_sentences": 0,
        "input_utf8_bytes": input_utf8_bytes,
        "training_utf8_bytes": output_utf8_bytes,
        "max_observed_document_utf8_bytes": max_observed_bytes,
    }
    return prepared, stats


def _byte_fallback_value(piece: str) -> int | None:
    if len(piece) != 6 or not piece.startswith("<0x") or not piece.endswith(">"):
        return None
    try:
        return int(piece[3:5], 16)
    except ValueError:
        return None


def _offsets_are_utf8_bytes(text: str, spans: list[tuple[int, int]]) -> bool:
    byte_length = len(text.encode("utf-8", errors="surrogatepass"))
    if byte_length == len(text) or not spans:
        return False
    final_end = max((end for _, end in spans), default=0)
    return final_end == byte_length or final_end > len(text)


def _byte_offset_to_char_span(text: str, start: int, end: int) -> tuple[int, int]:
    encoded = text.encode("utf-8", errors="surrogatepass")
    start = max(0, min(int(start), len(encoded)))
    end = max(start, min(int(end), len(encoded)))
    byte_owners: list[int] = []
    for char_index, char in enumerate(text):
        byte_owners.extend(
            [char_index] * len(char.encode("utf-8", errors="surrogatepass"))
        )
    if start == end:
        if start >= len(byte_owners):
            return len(text), len(text)
        owner = byte_owners[start]
        return owner, owner
    return byte_owners[start], byte_owners[end - 1] + 1


def normalize_token_char_spans(
    text: str,
    pieces: Iterable[str],
    spans: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Normalize backend offsets and expand every byte-fallback token span.

    SentencePiece represents the first bytes of a fallback Unicode character
    with zero-width offsets on some releases.  We decode consecutive
    ``<0xHH>`` pieces and assign *every* byte token to its owning Python
    character, so assistant-only masks supervise the complete UTF-8 sequence.
    """

    piece_list = [str(piece) for piece in pieces]
    raw_spans = [(int(start), int(end)) for start, end in spans]
    if len(piece_list) != len(raw_spans):
        raise ValueError("pieces and spans must have identical lengths")

    if _offsets_are_utf8_bytes(text, raw_spans):
        normalized = [
            _byte_offset_to_char_span(text, start, end)
            for start, end in raw_spans
        ]
    else:
        normalized = [
            (max(0, min(start, len(text))), max(0, min(end, len(text))))
            for start, end in raw_spans
        ]

    cursor = 0
    index = 0
    while index < len(piece_list):
        first_value = _byte_fallback_value(piece_list[index])
        if first_value is None:
            start, end = normalized[index]
            if end > start:
                cursor = max(cursor, end)
            index += 1
            continue

        group_end = index
        values: list[int] = []
        while group_end < len(piece_list):
            value = _byte_fallback_value(piece_list[group_end])
            if value is None:
                break
            values.append(value)
            group_end += 1

        try:
            decoded = bytes(values).decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        nonempty_starts = [
            start
            for start, end in normalized[index:group_end]
            if end > start
        ]
        candidate = min(nonempty_starts) if nonempty_starts else cursor
        if decoded and text[candidate : candidate + len(decoded)] != decoded:
            found = text.find(decoded, cursor)
            if found < 0:
                found = text.find(decoded)
            candidate = found if found >= 0 else candidate

        if decoded and text[candidate : candidate + len(decoded)] == decoded:
            byte_owner: list[int] = []
            for char_index, char in enumerate(decoded):
                byte_owner.extend(
                    [char_index] * len(char.encode("utf-8", errors="surrogatepass"))
                )
            if len(byte_owner) == len(values):
                for offset, owner in enumerate(byte_owner):
                    normalized[index + offset] = (
                        candidate + owner,
                        candidate + owner + 1,
                    )
                cursor = max(cursor, candidate + len(decoded))

        # Conservative fallback for unusual invalid byte groups: propagate the
        # closest concrete span instead of silently producing an empty mask.
        for offset in range(index, group_end):
            start, end = normalized[offset]
            if end > start:
                continue
            replacement: tuple[int, int] | None = None
            for neighbor in range(offset + 1, group_end):
                if normalized[neighbor][1] > normalized[neighbor][0]:
                    replacement = normalized[neighbor]
                    break
            if replacement is None:
                for neighbor in range(offset - 1, index - 1, -1):
                    if normalized[neighbor][1] > normalized[neighbor][0]:
                        replacement = normalized[neighbor]
                        break
            if replacement is not None:
                normalized[offset] = replacement
        index = group_end

    return normalized


class LegacyCharacterTokenizer:
    """Read-only compatibility tokenizer for old character-level vocab JSON files."""

    DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        special_tokens: Iterable[str] | None = None,
    ) -> None:
        self.special_tokens = tuple(special_tokens or self.DEFAULT_SPECIAL_TOKENS)
        if token_to_id is None:
            token_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}

        missing = [token for token in self.special_tokens if token not in token_to_id]
        if missing:
            raise ValueError(f"Missing special tokens in vocabulary: {missing}")

        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        if len(self.token_to_id) != len(self.id_to_token):
            raise ValueError("Vocabulary contains duplicated token ids.")

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        special_tokens: Iterable[str] | None = None,
    ) -> "LegacyCharacterTokenizer":
        special_tokens = tuple(special_tokens or cls.DEFAULT_SPECIAL_TOKENS)
        token_to_id = {token: idx for idx, token in enumerate(special_tokens)}

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)

        sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for token, freq in sorted_items:
            if freq < min_freq or token in token_to_id:
                continue
            if max_vocab_size is not None and len(token_to_id) >= max_vocab_size:
                break
            token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token(self) -> str:
        return self.special_tokens[0]

    @property
    def unk_token(self) -> str:
        return self.special_tokens[1]

    @property
    def bos_token(self) -> str:
        return self.special_tokens[2]

    @property
    def eos_token(self) -> str:
        return self.special_tokens[3]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return self._postprocess_ids(
            ids,
            add_eos=add_eos,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            truncation=truncation,
        )

    def encode_with_char_spans(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[list[int], list[tuple[int, int] | None]]:
        ids: list[int] = []
        spans: list[tuple[int, int] | None] = []
        if add_bos:
            ids.append(self.bos_id)
            spans.append(None)
        for index, ch in enumerate(text):
            ids.append(self.token_to_id.get(ch, self.unk_id))
            spans.append((index, index + 1))
        if add_eos:
            ids.append(self.eos_id)
            spans.append(None)
        return ids, spans

    def _postprocess_ids(
        self,
        ids: list[int],
        add_eos: bool,
        max_length: int | None,
        pad_to_max_length: bool,
        truncation: bool,
    ) -> list[int]:
        if max_length is not None:
            if len(ids) > max_length:
                if not truncation:
                    raise ValueError(
                        f"Encoded sequence length {len(ids)} exceeds max_length {max_length}."
                    )
                ids = ids[:max_length]
                if add_eos and max_length > 0:
                    ids[-1] = self.eos_id
            if pad_to_max_length and len(ids) < max_length:
                ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        special_set = set(self.special_tokens)
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and token in special_set:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": list(self.special_tokens),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LegacyCharacterTokenizer":
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls(
            token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
            special_tokens=payload.get("special_tokens", cls.DEFAULT_SPECIAL_TOKENS),
        )


class HuggingFaceBPETokenizer:
    """HuggingFace tokenizers BPE wrapper with the same interface used by training."""

    DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
    DEFAULT_USER_DEFINED_SYMBOLS = ("<|im_start|>", "<|im_end|>")

    def __init__(
        self,
        tokenizer_json: str | bytes,
        special_tokens: Iterable[str] | None = None,
        user_defined_symbols: Iterable[str] | None = None,
    ) -> None:
        self._require_tokenizers()
        if isinstance(tokenizer_json, bytes):
            tokenizer_json = tokenizer_json.decode("utf-8")
        self.tokenizer_json = tokenizer_json
        self.special_tokens = tuple(special_tokens or self.DEFAULT_SPECIAL_TOKENS)
        inferred_symbols = self._infer_user_defined_symbols(tokenizer_json, self.special_tokens)
        self.user_defined_symbols = tuple(
            user_defined_symbols or inferred_symbols or self.DEFAULT_USER_DEFINED_SYMBOLS
        )
        self.tokenizer = HFTokenizerImpl.from_str(self.tokenizer_json)
        self.token_to_id = self.tokenizer.get_vocab()
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @staticmethod
    def _require_tokenizers() -> None:
        if HFTokenizerImpl is None:
            raise ImportError(
                "HuggingFace tokenizers is required for this tokenizer. "
                "Install dependencies with: pip install -r requirements.txt"
            )

    @staticmethod
    def _infer_user_defined_symbols(
        tokenizer_json: str,
        base_special_tokens: Iterable[str],
    ) -> tuple[str, ...]:
        try:
            payload = json.loads(tokenizer_json)
        except json.JSONDecodeError:
            return ()
        base = set(base_special_tokens)
        symbols: list[str] = []
        for token in payload.get("added_tokens", []):
            if not isinstance(token, dict) or not token.get("special"):
                continue
            content = token.get("content")
            if isinstance(content, str) and content not in base and content not in symbols:
                symbols.append(content)
        return tuple(symbols)

    @classmethod
    def load(cls, path: str | Path) -> "HuggingFaceBPETokenizer":
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {input_path}")
        return cls(input_path.read_bytes())

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def pad_token(self) -> str:
        return self.special_tokens[0]

    @property
    def unk_token(self) -> str:
        return self.special_tokens[1]

    @property
    def bos_token(self) -> str:
        return self.special_tokens[2]

    @property
    def eos_token(self) -> str:
        return self.special_tokens[3]

    @property
    def pad_id(self) -> int:
        return self._token_id(self.pad_token)

    @property
    def unk_id(self) -> int:
        return self._token_id(self.unk_token)

    @property
    def bos_id(self) -> int:
        return self._token_id(self.bos_token)

    @property
    def eos_id(self) -> int:
        return self._token_id(self.eos_token)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token: {token}")
        return int(token_id)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(int(token_id) for token_id in encoded.ids)
        if add_eos:
            ids.append(self.eos_id)
        return self._postprocess_ids(ids, add_eos, max_length, pad_to_max_length, truncation)

    def encode_with_char_spans(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[list[int], list[tuple[int, int] | None]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = []
        spans: list[tuple[int, int] | None] = []
        if add_bos:
            ids.append(self.bos_id)
            spans.append(None)
        ids.extend(int(token_id) for token_id in encoded.ids)
        spans.extend(
            normalize_token_char_spans(
                text,
                encoded.tokens,
                [(int(start), int(end)) for start, end in encoded.offsets],
            )
        )
        if add_eos:
            ids.append(self.eos_id)
            spans.append(None)
        return ids, spans

    def _postprocess_ids(
        self,
        ids: list[int],
        add_eos: bool,
        max_length: int | None,
        pad_to_max_length: bool,
        truncation: bool,
    ) -> list[int]:
        if max_length is not None:
            if len(ids) > max_length:
                if not truncation:
                    raise ValueError(
                        f"Encoded sequence length {len(ids)} exceeds max_length {max_length}."
                    )
                ids = ids[:max_length]
                if add_eos and max_length > 0:
                    ids[-1] = self.eos_id
            if pad_to_max_length and len(ids) < max_length:
                ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode([int(idx) for idx in ids], skip_special_tokens=skip_special_tokens)

    def count_unknown_tokens(self, text: str) -> int:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return sum(1 for token_id in encoded.ids if int(token_id) == self.unk_id)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.tokenizer_json, encoding="utf-8")


class BPETokenizer:
    """SentencePiece tokenizer used by pretraining, finetuning, and generation."""

    DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
    DEFAULT_USER_DEFINED_SYMBOLS = ("<|im_start|>", "<|im_end|>")

    def __init__(
        self,
        model_proto: bytes,
        special_tokens: Iterable[str] | None = None,
        user_defined_symbols: Iterable[str] | None = None,
        model_type: str = "bpe",
    ) -> None:
        self._require_sentencepiece()
        self.model_proto = bytes(model_proto)
        self.special_tokens = tuple(special_tokens or self.DEFAULT_SPECIAL_TOKENS)
        self.user_defined_symbols = tuple(user_defined_symbols or self.DEFAULT_USER_DEFINED_SYMBOLS)
        self.model_type = str(model_type or "bpe").lower()
        self.processor = spm.SentencePieceProcessor()
        self.processor.LoadFromSerializedProto(self.model_proto)
        self.token_to_id = {
            self.processor.IdToPiece(idx): idx for idx in range(self.processor.GetPieceSize())
        }
        self.id_to_token = {
            idx: self.processor.IdToPiece(idx) for idx in range(self.processor.GetPieceSize())
        }

    @staticmethod
    def _require_sentencepiece() -> None:
        if spm is None:
            raise ImportError(
                "SentencePiece is required for BPE tokenization. "
                "Install dependencies with: pip install -r requirements.txt"
            )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        byte_fallback: bool = True,
        split_digits: bool = True,
        max_sentencepiece_length: int = 16,
        max_sentence_length: int = 16384,
        hard_vocab_limit: bool = False,
        normalization_rule_name: str = "identity",
        remove_extra_whitespaces: bool = False,
        user_defined_symbols: Iterable[str] | None = None,
        num_threads: int = 16,
    ) -> "BPETokenizer":
        del min_freq  # SentencePiece BPE does not expose a direct min_freq knob.
        target_vocab_size = int(max_vocab_size or vocab_size)
        return cls.train(
            texts=texts,
            vocab_size=target_vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            split_digits=split_digits,
            max_sentencepiece_length=max_sentencepiece_length,
            max_sentence_length=max_sentence_length,
            hard_vocab_limit=hard_vocab_limit,
            normalization_rule_name=normalization_rule_name,
            remove_extra_whitespaces=remove_extra_whitespaces,
            user_defined_symbols=user_defined_symbols,
            num_threads=num_threads,
        )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        byte_fallback: bool = True,
        split_digits: bool = True,
        max_sentencepiece_length: int = 16,
        max_sentence_length: int = 16384,
        hard_vocab_limit: bool = False,
        normalization_rule_name: str = "identity",
        remove_extra_whitespaces: bool = False,
        user_defined_symbols: Iterable[str] | None = None,
        num_threads: int = 16,
    ) -> "BPETokenizer":
        cls._require_sentencepiece()
        user_defined_symbols = tuple(user_defined_symbols or cls.DEFAULT_USER_DEFINED_SYMBOLS)
        prepared_texts, _length_stats = prepare_sentencepiece_training_texts(
            texts,
            max_sentence_length=max_sentence_length,
        )
        if not prepared_texts:
            raise ValueError("No non-empty texts remain for SentencePiece training")
        writer = BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(prepared_texts),
            model_writer=writer,
            vocab_size=int(vocab_size),
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=cls.DEFAULT_SPECIAL_TOKENS[0],
            unk_piece=cls.DEFAULT_SPECIAL_TOKENS[1],
            bos_piece=cls.DEFAULT_SPECIAL_TOKENS[2],
            eos_piece=cls.DEFAULT_SPECIAL_TOKENS[3],
            user_defined_symbols=list(user_defined_symbols),
            character_coverage=float(character_coverage),
            byte_fallback=bool(byte_fallback),
            split_digits=bool(split_digits),
            max_sentencepiece_length=int(max_sentencepiece_length),
            max_sentence_length=int(max_sentence_length),
            hard_vocab_limit=bool(hard_vocab_limit),
            normalization_rule_name=normalization_rule_name,
            remove_extra_whitespaces=bool(remove_extra_whitespaces),
            num_threads=int(num_threads),
        )
        return cls(
            model_proto=writer.getvalue(),
            special_tokens=cls.DEFAULT_SPECIAL_TOKENS,
            user_defined_symbols=user_defined_symbols,
            model_type=model_type,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.processor.GetPieceSize())

    @property
    def pad_token(self) -> str:
        return self.special_tokens[0]

    @property
    def unk_token(self) -> str:
        return self.special_tokens[1]

    @property
    def bos_token(self) -> str:
        return self.special_tokens[2]

    @property
    def eos_token(self) -> str:
        return self.special_tokens[3]

    @property
    def pad_id(self) -> int:
        return int(self.processor.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self.processor.unk_id())

    @property
    def bos_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.processor.eos_id())

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(int(token_id) for token_id in self.processor.EncodeAsIds(text))
        if add_eos:
            ids.append(self.eos_id)
        return self._postprocess_ids(
            ids,
            add_eos=add_eos,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            truncation=truncation,
        )

    def encode_with_char_spans(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[list[int], list[tuple[int, int] | None]]:
        ids: list[int] = []
        spans: list[tuple[int, int] | None] = []
        if add_bos:
            ids.append(self.bos_id)
            spans.append(None)

        proto = self._encode_as_proto(text)
        proto_pieces = list(proto.pieces)
        ids.extend(int(piece.id) for piece in proto_pieces)
        spans.extend(
            normalize_token_char_spans(
                text,
                [str(piece.piece) for piece in proto_pieces],
                [(int(piece.begin), int(piece.end)) for piece in proto_pieces],
            )
        )

        if add_eos:
            ids.append(self.eos_id)
            spans.append(None)
        return ids, spans

    def _encode_as_proto(self, text: str):
        """Return token pieces with offsets across SentencePiece 0.2.x APIs."""

        try:
            return self.processor.Encode(text, return_type="proto")
        except TypeError:
            # SentencePiece <= 0.2.1 exposes the same protobuf through the
            # legacy method and does not accept ``return_type``.
            return self.processor.EncodeAsImmutableProto(text)

    def _postprocess_ids(
        self,
        ids: list[int],
        add_eos: bool,
        max_length: int | None,
        pad_to_max_length: bool,
        truncation: bool,
    ) -> list[int]:
        if max_length is not None:
            if len(ids) > max_length:
                if not truncation:
                    raise ValueError(
                        f"Encoded sequence length {len(ids)} exceeds max_length {max_length}."
                    )
                ids = ids[:max_length]
                if add_eos and max_length > 0:
                    ids[-1] = self.eos_id
            if pad_to_max_length and len(ids) < max_length:
                ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        int_ids = [int(idx) for idx in ids]
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            int_ids = [idx for idx in int_ids if idx not in special_ids]
        return self.processor.DecodeIds(int_ids)

    def count_unknown_tokens(self, text: str) -> int:
        return sum(1 for token_id in self.processor.EncodeAsIds(text) if int(token_id) == self.unk_id)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            payload = {
                "type": "sentencepiece",
                "model_type": self.model_type,
                "model_proto_b64": base64.b64encode(self.model_proto).decode("ascii"),
                "special_tokens": list(self.special_tokens),
                "user_defined_symbols": list(self.user_defined_symbols),
                "vocab_size": self.vocab_size,
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            output_path.write_bytes(self.model_proto)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer | LegacyCharacterTokenizer":
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {input_path}")

        raw = input_path.read_bytes()
        if input_path.suffix.lower() == ".json":
            try:
                payload = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                payload = None
            if isinstance(payload, dict):
                if "token_to_id" in payload:
                    return LegacyCharacterTokenizer(
                        token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
                        special_tokens=payload.get(
                            "special_tokens", LegacyCharacterTokenizer.DEFAULT_SPECIAL_TOKENS
                        ),
                    )
                if payload.get("type") in {"sentencepiece", "sentencepiece_bpe"} and "model_proto_b64" in payload:
                    return cls(
                        model_proto=base64.b64decode(payload["model_proto_b64"]),
                        special_tokens=payload.get("special_tokens", cls.DEFAULT_SPECIAL_TOKENS),
                        user_defined_symbols=payload.get(
                            "user_defined_symbols", cls.DEFAULT_USER_DEFINED_SYMBOLS
                        ),
                        model_type=payload.get("model_type", "bpe"),
                    )
                if "model" in payload and payload.get("model", {}).get("type") == "BPE":
                    return HuggingFaceBPETokenizer(raw)
        return cls(model_proto=raw)


CharacterTokenizer = BPETokenizer
