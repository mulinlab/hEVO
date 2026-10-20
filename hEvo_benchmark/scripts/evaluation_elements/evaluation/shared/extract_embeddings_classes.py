from typing import Dict, List, Optional, Union

import numpy as np
import torch
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from modelgenerator.tasks import Embed
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class ExtractEmbeddingBase(object):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError


class ExtractEmbeddingForDNABert(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForhEvo(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForhEvo_rare(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            self.model = self.model.to(torch.bfloat16)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]
                print(last_hidden.shape)

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForhEvo_rand(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            self.model = self.model.to(torch.bfloat16)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForEvo2_1B(object):
    def __init__(self, model_path, model_name, **kwargs):
        from evo2 import Evo2

        self.model_name = "evo2-1b"
        self.model = Evo2(
            "evo2_1b_base",
            model_path,
        )
        self.tokenizer = self.model.tokenizer.tokenize_batch
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # 1. Convert to ASCII bytes while preserving the existing behavior.
        byte_texts = [text.encode("ascii") for text in texts]

        # 2. Retrieve the original token IDs as List[List[int]].
        raw_batch_ids = self.tokenizer(byte_texts)

        processed_ids = []
        for seq_ids in raw_batch_ids:
            # 3. Apply truncation manually.
            if truncation and len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]

            # 4. Apply padding manually.
            # If the Dataset requests padding (usually "max_length"), pad with zeros.
            if padding == "max_length" or padding is True:
                if len(seq_ids) < max_length:
                    # Assume 0 is the padding token; this is generally safe for DNA models because it is masked.
                    pad_len = max_length - len(seq_ids)
                    seq_ids = seq_ids + [0] * pad_len

            processed_ids.append(seq_ids)

        # 5. Convert to a tensor.
        # Note: embedding layers usually require the Long dtype.
        input_ids = torch.tensor(processed_ids, dtype=torch.long)

        encoded = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
        }
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        layer_name = "blocks.20.mlp.l3"

        # Ensure the input is on the correct device.
        input_ids = input_ids.to(self.device)

        # Evo2 forward
        outputs, embeddings = self.model(
            input_ids,
            return_embeddings=True,
            layer_names=[layer_name],
        )
        last_hidden = embeddings[layer_name]

        if pooling == "cls":
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                emb = pooler
            else:
                emb = last_hidden[:, 0, :]
        elif pooling == "mean":
            # Use attention_mask for correct mean pooling.
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device).unsqueeze(-1)
                # Avoid division by zero.
                sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                emb = sum_embeddings / sum_mask
            else:
                emb = last_hidden.mean(dim=1)
        elif pooling == "max":
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device).unsqueeze(-1)
                # Set padded positions to a very small value so they do not affect the maximum.
                masked_hidden = last_hidden.masked_fill(attention_mask == 0, -1e9)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb, _ = torch.max(last_hidden, dim=1)
        else:
            emb = last_hidden

        if return_numpy:
            emb = emb.cpu().numpy()

        return emb


class ExtractEmbeddingForEvo2_7B(object):
    def __init__(self, model_path, model_name, **kwargs):
        from evo2 import Evo2

        self.model_name = "evo2-7b"
        self.model = Evo2(
            "evo2_7b_base",
            model_path,
        )
        self.tokenizer = self.model.tokenizer.tokenize_batch
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # 1. Convert to ASCII bytes while preserving the existing behavior.
        byte_texts = [text.encode("ascii") for text in texts]

        # 2. Retrieve the original token IDs as List[List[int]].
        raw_batch_ids = self.tokenizer(byte_texts)

        processed_ids = []
        for seq_ids in raw_batch_ids:
            # 3. Apply truncation manually.
            if truncation and len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]

            # 4. Apply padding manually.
            # If the Dataset requests padding (usually "max_length"), pad with zeros.
            if padding == "max_length" or padding is True:
                if len(seq_ids) < max_length:
                    # Assume 0 is the padding token; this is generally safe for DNA models because it is masked.
                    pad_len = max_length - len(seq_ids)
                    seq_ids = seq_ids + [0] * pad_len

            processed_ids.append(seq_ids)

        # 5. Convert to a tensor.
        # Note: embedding layers usually require the Long dtype.
        input_ids = torch.tensor(processed_ids, dtype=torch.long)

        encoded = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
        }
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        layer_name = "blocks.24.mlp.l3"

        # Ensure the input is on the correct device.
        input_ids = input_ids.to(self.device)

        # Evo2 forward
        outputs, embeddings = self.model(
            input_ids,
            return_embeddings=True,
            layer_names=[layer_name],
        )
        last_hidden = embeddings[layer_name]

        if pooling == "cls":
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                emb = pooler
            else:
                emb = last_hidden[:, 0, :]
        elif pooling == "mean":
            # Use attention_mask for correct mean pooling.
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device).unsqueeze(-1)
                # Avoid division by zero.
                sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                emb = sum_embeddings / sum_mask
            else:
                emb = last_hidden.mean(dim=1)
        elif pooling == "max":
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device).unsqueeze(-1)
                # Set padded positions to a very small value so they do not affect the maximum.
                masked_hidden = last_hidden.masked_fill(attention_mask == 0, -1e9)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb, _ = torch.max(last_hidden, dim=1)
        else:
            emb = last_hidden

        if return_numpy:
            emb = emb.cpu().numpy()

        return emb


class ExtractEmbeddingForNT(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForHyena(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGPN(ExtractEmbeddingBase):
    import gpn.model  # noqa: F401
    from gpn.data import GenomeMSA, Tokenizer  # noqa: F401

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states
                # print(last_hidden.shape)

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForSpeciesLM(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # Split each text into 6-base segments.
        encoded = self.tokenizer(
            [" ".join([text[i : i + 6] for i in range(len(text))]) for text in texts],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        # print(encoded["input_ids"].shape)
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGenos_1B(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGenos_10B(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForAIDO(object):
    """
    适配 AIDO 框架 (modelgenerator) 的 Embedding 提取器
    """

    def __init__(self, model_path: str, model_name: str):
        # Note: AIDO usually specifies the backbone in its config; here, model_name is assumed to be the backbone name.
        # For example: "aido_dna_300m" or "aido_dna_7b".
        # model_path may not be needed here unless AIDO supports loading weights from a local path.
        # To load local weights, specify 'pretrained_weights': model_path in the config.
        self.model_path = model_path
        self.model_name = model_name
        print(f"Loading AIDO model with backbone: {model_name}...")

        # Initialize the AIDO Embed task.
        # The config dictionary may need to be updated to load local weights.
        model_dict = {
            "AIDO.DNA-300M": "aido_dna_300m",
            "AIDO.DNA-7B": "aido_dna_7b",
        }
        self.model = Embed.from_config(
            {
                "model.backbone": model_dict.get(model_name, self.model_name),
                "model.batch_size": 1,
                # "train.pretrained_weights": model_path  # Add this entry when specifying a local path.
            }
        )

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # AIDO wraps its tokenizer in the transform and does not expose a pad_token_id attribute.
        # Set a fallback value; AIDO usually handles padding internally.
        self.pad_token_id = 0

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        使用 AIDO 的 model.transform 进行分词，并强制统一长度
        """
        try:
            # The AIDO transform usually returns a Dict[str, torch.Tensor].
            encoded = self.model.transform({"sequences": texts})
        except Exception as e:
            raise RuntimeError(f"AIDO transformation failed: {e}")

        out = {}
        input_ids = encoded.get("input_ids")

        # Core fix: handle the length of each sequence manually.
        # Case A: AIDO returned a batch with shape (Batch_size, Seq_len).
        if input_ids.ndim == 2:
            # Truncate the sequence.
            if truncation and input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]

            # Pad the sequence if it is too short and padding is enabled.
            if padding and input_ids.shape[1] < max_length:
                pad_size = max_length - input_ids.shape[1]
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=self.pad_token_id
                )

        # Case B: A single-item Dataset call returned a tensor with shape (Seq_len,).
        elif input_ids.ndim == 1:
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            if padding and len(input_ids) < max_length:
                pad_size = max_length - len(input_ids)
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_size), value=self.pad_token_id
                )

        out["input_ids"] = input_ids

        # Process attention_mask so it matches the length of input_ids.
        if "attention_mask" in encoded:
            mask = encoded["attention_mask"]
            if truncation:
                mask = mask[..., :max_length]
            if padding and mask.shape[-1] < max_length:
                pad_size = max_length - mask.shape[-1]
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=0)
            out["attention_mask"] = mask
        else:
            out["attention_mask"] = (out["input_ids"] != self.pad_token_id).long()

        return out

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()  # Assume 0 is the padding value.
        attention_mask = attention_mask.to(self.device)

        # Rebuild the batch dictionary expected by AIDO.
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.inference_mode():
            # The AIDO Embed task forward pass usually returns embeddings directly.
            outputs = self.model(batch)

            # Check the output type.
            # Case 1: outputs is an embedding tensor (B, Hidden); AIDO has already pooled it.
            # Case 2: outputs is a sequence tensor (B, Seq, Hidden); pool it here.

            emb = outputs

            # The output may be a dictionary in some AIDO versions.
            if isinstance(outputs, dict):
                if "embedding" in outputs:
                    emb = outputs["embedding"]
                elif "last_hidden_state" in outputs:
                    emb = outputs["last_hidden_state"]
                else:
                    # Try to find the first tensor value.
                    for v in outputs.values():
                        if isinstance(v, torch.Tensor):
                            emb = v
                            break

            # Apply pooling.
            if len(emb.shape) == 3:  # (Batch, Seq, Hidden)
                last_hidden = emb
                if pooling == "cls":
                    # Assume the first token is CLS; if AIDO has no CLS token, use the first token directly.
                    emb = last_hidden[:, 0, :]
                elif pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                    summed = torch.sum(last_hidden * mask, dim=1)
                    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                    emb = summed / denom
                elif pooling == "max":
                    neg_inf = -1e9
                    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                    masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                    emb, _ = torch.max(masked_hidden, dim=1)
                else:
                    emb = last_hidden

            # If the shape is (Batch, Hidden), the model has already pooled the output; use it directly.
            elif len(emb.shape) == 2:
                pass

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGENALM(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        # Use super().__init__ to avoid duplicating initialization logic.
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # Note: AutoModel (the base model) is generally recommended for extracting embeddings.
        # Use AutoModelForSequenceClassification only when outputs from a specific classification head are needed.
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # GENA-LM usually accepts raw DNA sequences (ACGT) and tokenizes them internally.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Prefer the final-layer hidden states.
            last_hidden = getattr(outputs, "last_hidden_state", None)

            if last_hidden is None:
                # Fallback: use the last layer from outputs.hidden_states when available.
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    last_hidden = hidden_states[-1]
                else:
                    raise RuntimeError("无法从 GENA-LM 输出中提取隐藏状态")

            # Pooling strategy.
            if pooling == "cls":
                # GENA-LM usually places [CLS] or <s> at position 0.
                emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                # Fill padded positions with a very small value so they are ignored.
                masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = (
                    last_hidden  # Return embeddings for all sequence positions with shape (batch, seq_len, hidden_dim).
                )

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGROVER(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        # Use super().__init__ to avoid duplicating initialization logic.
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # Note: AutoModel (the base model) is generally recommended for extracting embeddings.
        # Use AutoModelForSequenceClassification only when outputs from a specific classification head are needed.
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # GENA-LM usually accepts raw DNA sequences (ACGT) and tokenizes them internally.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Prefer the final-layer hidden states.
            last_hidden = getattr(outputs, "last_hidden_state", None)

            if last_hidden is None:
                # Fallback: use the last layer from outputs.hidden_states when available.
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    last_hidden = hidden_states[-1]
                else:
                    raise RuntimeError("无法从 GENA-LM 输出中提取隐藏状态")

            # Pooling strategy.
            if pooling == "cls":
                # GENA-LM usually places [CLS] or <s> at position 0.
                emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                # Fill padded positions with a very small value so they are ignored.
                masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = (
                    last_hidden  # Return embeddings for all sequence positions with shape (batch, seq_len, hidden_dim).
                )

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForMistralDNA(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        # Use super().__init__ to avoid duplicating initialization logic.
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # Note: AutoModel (the base model) is generally recommended for extracting embeddings.
        # Use AutoModelForSequenceClassification only when outputs from a specific classification head are needed.
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # GENA-LM usually accepts raw DNA sequences (ACGT) and tokenizes them internally.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Prefer the final-layer hidden states.
            last_hidden = getattr(outputs, "last_hidden_state", None)

            if last_hidden is None:
                # Fallback: use the last layer from outputs.hidden_states when available.
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    last_hidden = hidden_states[-1]
                else:
                    raise RuntimeError("无法从 GENA-LM 输出中提取隐藏状态")

            # Pooling strategy.
            if pooling == "cls":
                # GENA-LM usually places [CLS] or <s> at position 0.
                emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                # Fill padded positions with a very small value so they are ignored.
                masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = (
                    last_hidden  # Return embeddings for all sequence positions with shape (batch, seq_len, hidden_dim).
                )

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForNTv3(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=128,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForGENERator(ExtractEmbeddingBase):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=128,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "模型输出中既不存在 last_hidden_state 也不存在 hidden_states"
                    )
                last_hidden = hidden_states[-1]

            if pooling == "cls":
                pooler = getattr(outputs, "pooler_output", None)
                if pooler is not None:
                    emb = pooler
                else:
                    emb = last_hidden[:, 0, :]
            elif pooling == "mean":
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = torch.sum(last_hidden * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / denom
            elif pooling == "max":
                neg_inf = -1e9
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                masked_hidden = last_hidden.masked_fill(mask == 0, neg_inf)
                emb, _ = torch.max(masked_hidden, dim=1)
            else:
                emb = last_hidden

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForEnformer:
    def __init__(self, model_path: str, model_name: str = "enformer"):
        self.model_path = model_path
        self.model_name = model_name

        # Load the model from pretrained weights (assuming the official weights are used).
        # For a local checkpoint, Enformer.from_hparams(...) can also be used.
        self.model = Enformer.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Follow the documented mapping exactly: 0:A, 1:C, 2:G, 3:T, 4:N.
        self.base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 196608,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        all_ids = []
        for text in texts:
            seq = text.upper()
            if truncation:
                seq = seq[:max_length]

            # Convert bases to indices.
            ids = [self.base_map.get(base, 4) for base in seq]

            if padding and len(ids) < max_length:
                ids += [4] * (max_length - len(ids))  # Pad with N.

            all_ids.append(ids)

        input_ids = torch.tensor(all_ids)  # Shape: (batch, seq_len).
        return {"input_ids": input_ids}

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)

        with torch.inference_mode():
            # 1. Convert to one-hot encoding as documented.
            one_hot = seq_indices_to_one_hot(input_ids).to(self.device)

            # 2. Set return_embeddings=True.
            # The documented return value is (prediction dictionary, embeddings).
            _, embeddings = self.model(one_hot, return_embeddings=True)

            # Embedding shape: (batch, 896, 3072).
            if pooling == "mean":
                emb = embeddings.mean(dim=1)
            elif pooling == "max":
                emb, _ = embeddings.max(dim=1)
            else:
                emb = embeddings

            if return_numpy:
                return emb.cpu().numpy()
            return emb


class ExtractEmbeddingForCaduceus:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        # Caduceus relies on remote code for its Mamba implementation.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self, texts: List[str], max_length: int = 512, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)
        # Note: Mamba models may not need a conventional attention_mask, but it is retained for compatibility.
        with torch.inference_mode():
            outputs = self.model(input_ids)
            # Caduceus outputs usually include last_hidden_state directly.
            last_hidden = outputs.last_hidden_state

            if pooling == "mean":
                # Use simple mean pooling; padding can be excluded in a future refinement.
                emb = last_hidden.mean(dim=1)
            elif pooling == "cls":
                # Use the first token in the sequence.
                emb = last_hidden[:, 0, :]
            else:
                emb = last_hidden

            return emb.cpu().numpy() if return_numpy else emb
