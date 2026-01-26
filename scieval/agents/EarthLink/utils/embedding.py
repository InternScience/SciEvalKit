import io
import os
import contextlib
from typing import Any, Union, List
from collections.abc import Iterable
from pydantic import Field
from langchain_openai import OpenAIEmbeddings


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI Embeddings with tokenizer initialization."""

    tokenizer: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        tokenizer_name = self.tiktoken_model_name or self.model
        if not self.tiktoken_enabled:
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO())
                ):
                    from transformers import AutoTokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed for OpenAIEmbeddings to work without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_name
            )
        else:
            try:
                import tiktoken
                tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            except KeyError:
                tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

    def _tokenize(
        self, texts: list[str], chunk_size: int
    ) -> tuple[Iterable[int], list[Union[list[int], str]], list[int]]:
        """
        Take the input `texts` and `chunk_size` and return 3 iterables as a tuple:

        We have `batches`, where batches are sets of individual texts
        we want responses from the openai api. The length of a single batch is
        `chunk_size` texts.

        Each individual text is also split into multiple texts based on the
        `embedding_ctx_length` parameter (based on number of tokens).

        This function returns a 3-tuple of the following:

        _iter: An iterable of the starting index in `tokens` for each *batch*
        tokens: A list of tokenized texts, where each text has already been split
            into sub-texts based on the `embedding_ctx_length` parameter. In the
            case of tiktoken, this is a list of token arrays. In the case of
            HuggingFace transformers, this is a list of strings.
        indices: An iterable of the same length as `tokens` that maps each token-array
            to the index of the original text in `texts`.
        """
        tokens: list[Union[list[int], str]] = []
        indices: list[int] = []

        tokenizer = self.tokenizer
        
        # If tiktoken flag set to False
        if not self.tiktoken_enabled:

            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized: list[int] = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk: list[int] = tokenized[
                        j : j + self.embedding_ctx_length
                    ]

                    # Convert token IDs back to a string
                    chunk_text: str = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            encoder_kwargs: dict[str, Any] = {
                k: v
                for k, v in {
                    "allowed_special": self.allowed_special,
                    "disallowed_special": self.disallowed_special,
                }.items()
                if v is not None
            }
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                if encoder_kwargs:
                    token = tokenizer.encode(text, **encoder_kwargs)
                else:
                    token = tokenizer.encode_ordinary(text)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), chunk_size)
        else:
            _iter = range(0, len(tokens), chunk_size)
        return _iter, tokens, indices


EMBEDDING_FUNC = None

def get_embedding_func():
    global EMBEDDING_FUNC

    if EMBEDDING_FUNC is None:
        from .. import config as CFG
        EMBEDDING_FUNC = CustomOpenAIEmbeddings(
            model=CFG.EMBEDDING_MODEL,
            api_key=CFG.EMBEDDING_API_KEY,
            base_url=CFG.EMBEDDING_BASE_URL,
            tiktoken_enabled=False,
            tiktoken_model_name="Qwen/Qwen3-Embedding-8B",
        )

    return EMBEDDING_FUNC


def embed_query(text: str) -> List[float]:
    embedding_func = get_embedding_func()
    return embedding_func.embed_query(text)


async def aembed_query(text: str) -> List[float]:
    embedding_func = get_embedding_func()
    return await embedding_func.aembed_query(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    embedding_func = get_embedding_func()
    return embedding_func.embed_documents(texts)


async def aembed_documents(texts: List[str]) -> List[List[float]]:
    embedding_func = get_embedding_func()
    return await embedding_func.aembed_documents(texts)