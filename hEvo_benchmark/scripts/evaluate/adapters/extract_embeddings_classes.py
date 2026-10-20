import os
from typing import Dict, List, Optional, Tuple, Union

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
    def __init__(self, model_path: str, model_name: str):
        try:
            import gpn.model  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "gpn.model":
                raise
            from gpn import register_auto_classes

            register_auto_classes("ss")
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


class ExtractEmbeddingForGPNMSA:
    """Use GPN's native MSA embedding inference on genomic intervals."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        msa_path: str,
        window_size: int = 128,
        center_window_size: int = 1000,
    ):
        try:
            from gpn.msa.data import GenomeMSA
            from gpn.msa.inference import EmbeddingInference
        except ModuleNotFoundError as exc:
            if exc.name not in {"gpn.msa.data", "gpn.msa.inference"}:
                raise
            from gpn.data import GenomeMSA
            from gpn.msa.embedding import EmbeddingInference
        import zarr

        self.model_path = model_path
        self.model_name = model_name
        self.msa_path = msa_path
        self.window_size = int(window_size)
        self.center_window_size = int(center_window_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_mismatch_rows: List[Dict[str, object]] = []

        if self.window_size < 1:
            raise ValueError("GPN-MSA window_size must be positive")
        if self.center_window_size < 1:
            raise ValueError("GPN-MSA center_window_size must be positive")

        zip_store_type = getattr(zarr, "ZipStore", None)
        if zip_store_type is None:
            zip_store_type = zarr.storage.ZipStore
        self._msa_store = zip_store_type(msa_path, mode="r")
        self.genome_msa = GenomeMSA(self._msa_store, in_memory=False)
        self.inference = EmbeddingInference(
            model_path=model_path,
            genome_msa=self.genome_msa,
            window_size=self.window_size,
            center_window_size=self.center_window_size,
        )
        self.model = self.inference.model.to(self.device).eval()

        model_config = self.model.model.config
        if hasattr(model_config, "n_aux_features") and self.genome_msa.data:
            first_chrom = next(iter(self.genome_msa.data))
            species_count = int(self.genome_msa.data[first_chrom].shape[1])
            expected_aux_features = (species_count - 1) * 5
            if int(model_config.n_aux_features) != expected_aux_features:
                self.close()
                raise ValueError(
                    "GPN-MSA checkpoint and MSA species count do not match: "
                    f"checkpoint expects {model_config.n_aux_features} auxiliary "
                    f"features, but {species_count} MSA species provide "
                    f"{expected_aux_features}"
                )

    def get_embedding_for_intervals(
        self, records: List[Dict[str, object]], verify_sequences: bool = True
    ) -> torch.Tensor:
        if not records:
            return torch.empty((0, 0), dtype=torch.float32)

        # GenomeMSA returns the requested genomic interval without padding, so
        # rows of different lengths cannot be stacked into one model batch.
        # Bucket by interval length, use the full interval as the pooling window
        # (capped by the configured maximum), then restore the input row order.
        length_groups: Dict[int, List[Tuple[int, Dict[str, object]]]] = {}
        for index, record in enumerate(records):
            interval_length = int(record["end"]) - int(record["start"])
            if interval_length <= 0:
                raise ValueError(
                    f"Invalid GPN-MSA interval {record.get('chrom')}:"
                    f"{record.get('start')}-{record.get('end')}"
                )
            length_groups.setdefault(interval_length, []).append((index, record))

        embeddings_by_index: List[Optional[torch.Tensor]] = [None] * len(records)
        configured_center_window = self.center_window_size
        try:
            for interval_length, indexed_records in length_groups.items():
                self.center_window_size = min(configured_center_window, interval_length)
                self.inference.model.center_window_size = self.center_window_size
                indices = [index for index, _ in indexed_records]
                group_records = [record for _, record in indexed_records]
                group_embeddings = self._get_embedding_for_equal_length_intervals(
                    group_records, verify_sequences=verify_sequences
                )
                for index, embedding in zip(indices, group_embeddings):
                    embeddings_by_index[index] = embedding
        finally:
            self.center_window_size = configured_center_window
            self.inference.model.center_window_size = configured_center_window

        if any(embedding is None for embedding in embeddings_by_index):
            raise RuntimeError("GPN-MSA failed to embed one or more input intervals")
        return torch.stack(embeddings_by_index, dim=0)

    def _get_embedding_for_equal_length_intervals(
        self, records: List[Dict[str, object]], verify_sequences: bool = True
    ) -> torch.Tensor:
        if not records:
            return torch.empty((0, 0), dtype=torch.float32)

        chroms = [str(record["chrom"]).removeprefix("chr") for record in records]
        starts = np.asarray(
            [int(record["start"]) for record in records], dtype=np.int64
        )
        ends = np.asarray([int(record["end"]) for record in records], dtype=np.int64)
        missing_chroms = sorted(set(chroms).difference(self.genome_msa.data))
        if missing_chroms:
            raise ValueError(
                "Chromosomes are missing from the GPN-MSA store: "
                + ", ".join(missing_chroms)
            )
        for record, start, end in zip(records, starts, ends):
            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid GPN-MSA interval {record.get('chrom')}:{start}-{end}"
                )
            if end - start < self.center_window_size:
                raise ValueError(
                    f"Interval {record.get('chrom')}:{start}-{end} is shorter than "
                    f"the requested {self.center_window_size} bp pooling window"
                )

        model_inputs = self.inference.tokenize_function(
            {"chrom": chroms, "start": starts, "end": ends}
        )
        input_ids_fwd = model_inputs["input_ids_fwd"]
        if input_ids_fwd.shape[1] < self.center_window_size:
            raise ValueError(
                "GPN-MSA returned an alignment shorter than the requested pooling window: "
                f"{input_ids_fwd.shape[1]} < {self.center_window_size}"
            )

        if verify_sequences:
            expected_rows = []
            for record in records:
                sequence = str(record["seq"]).upper()
                if len(sequence) != int(record["end"]) - int(record["start"]):
                    raise ValueError(
                        f"Sequence length does not match coordinates for "
                        f"{record.get('SeqID', record.get('chrom'))}"
                    )
                expected_rows.append(
                    np.frombuffer(sequence.encode("ascii"), dtype="S1")
                )
            expected = self.genome_msa.tokenizer(np.stack(expected_rows))
            target_species = input_ids_fwd
            if target_species.shape != expected.shape:
                raise ValueError(
                    "GPN-MSA target sequence length does not match the benchmark sequence: "
                    f"{target_species.shape} != {expected.shape}"
                )
            mismatches = np.argwhere(target_species != expected)
            if mismatches.size:
                mismatches_per_row = np.count_nonzero(
                    target_species != expected, axis=1
                )
                target_coverage = np.count_nonzero(target_species != 0, axis=1)
                for row_index in np.flatnonzero(mismatches_per_row):
                    record = records[int(row_index)]
                    self.sequence_mismatch_rows.append(
                        {
                            "sample_id": str(record.get("SeqID", row_index)),
                            "chrom": chroms[int(row_index)],
                            "start": int(starts[int(row_index)]),
                            "end": int(ends[int(row_index)]),
                            "mismatch_bases": int(mismatches_per_row[int(row_index)]),
                            "msa_target_coverage": float(
                                target_coverage[int(row_index)]
                                / target_species.shape[1]
                            ),
                        }
                    )

        tensor_inputs = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            embeddings = self.model(**tensor_inputs)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise RuntimeError(
                "Official GPN-MSA embedding inference returned an unexpected shape: "
                f"{tuple(embeddings.shape)}"
            )
        return embeddings.detach().cpu().to(torch.float32)

    def close(self) -> None:
        store = getattr(self, "_msa_store", None)
        if store is not None:
            store.close()
            self._msa_store = None


class ExtractEmbeddingForGPNStar:
    """Use GPN-Star's native MSA embedding inference on genomic intervals."""

    class _CenterEmbedding(torch.nn.Module):
        """GPN-Star model wrapper with exact-width, configurable center pooling."""

        def __init__(self, encoder: torch.nn.Module, center_window_size: int):
            super().__init__()
            self.encoder = encoder
            self.center_window_size = int(center_window_size)

        def _pool(self, input_ids, source_ids, target_species):
            hidden = self.encoder(
                input_ids=input_ids,
                source_ids=source_ids,
                target_species=target_species,
            ).last_hidden_state
            width = int(self.center_window_size)
            center = hidden.shape[1] // 2
            left = center - width // 2
            right = left + width
            if left < 0 or right > hidden.shape[1]:
                raise ValueError(
                    f"Requested GPN-Star pooling width {width} exceeds model output "
                    f"length {hidden.shape[1]}"
                )
            return hidden[:, left:right].mean(dim=1)

        def forward(
            self,
            input_ids_fwd=None,
            input_ids_rev=None,
            source_ids_fwd=None,
            source_ids_rev=None,
            target_species=None,
        ):
            forward_embedding = self._pool(
                input_ids_fwd, source_ids_fwd, target_species
            )
            reverse_embedding = self._pool(
                input_ids_rev, source_ids_rev, target_species
            )
            return (forward_embedding + reverse_embedding) / 2

    def __init__(
        self,
        model_path: str,
        model_name: str,
        msa_paths: List[str],
        msa_species: List[int],
        window_size: int = 128,
        center_window_size: int = 1000,
    ):
        from gpn.star.data import GenomeMSA
        from gpn.star.embedding import EmbeddingInference

        if not msa_paths:
            raise ValueError("GPN-Star requires at least one MSA store")
        if len(msa_paths) != len(msa_species):
            raise ValueError(
                "GPN-Star MSA store and species-count lists must have equal lengths"
            )
        if any(int(count) < 1 for count in msa_species):
            raise ValueError("GPN-Star MSA species counts must all be positive")

        self.model_path = model_path
        self.model_name = model_name
        self.msa_paths = [str(path) for path in msa_paths]
        self.msa_species = [int(count) for count in msa_species]
        self.window_size = int(window_size)
        self.center_window_size = int(center_window_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_mismatch_rows: List[Dict[str, object]] = []

        if self.window_size < 1:
            raise ValueError("GPN-Star window_size must be positive")
        if self.center_window_size < 1:
            raise ValueError("GPN-Star center_window_size must be positive")

        missing_paths = [path for path in self.msa_paths if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError(
                "GPN-Star MSA store(s) not found: " + ", ".join(missing_paths)
            )

        self.genome_msa_list = [
            GenomeMSA(path, n_species=count, in_memory=False)
            for path, count in zip(self.msa_paths, self.msa_species)
        ]
        for path, count, genome_msa in zip(
            self.msa_paths, self.msa_species, self.genome_msa_list
        ):
            if not genome_msa.data:
                raise ValueError(f"GPN-Star MSA store contains no chromosomes: {path}")
            first_chrom = next(iter(genome_msa.data))
            observed_count = int(genome_msa.data[first_chrom].shape[1])
            if observed_count != count:
                raise ValueError(
                    f"GPN-Star MSA species count mismatch for {path}: "
                    f"configured {count}, found {observed_count}"
                )
        self.inference = EmbeddingInference(
            model_path=model_path,
            genome_msa_list=self.genome_msa_list,
            window_size=self.window_size,
            center_window_size=self.center_window_size,
        )
        # Keep the official GPN-Star checkpoint and MSA tokenizer, while using
        # a local pooling wrapper so intervals shorter than the configured
        # center window can be embedded over their full length. The official
        # wrapper's odd-width slice drops one base, so this wrapper also uses
        # exact-width slicing. DataParallel lets multi-GPU Slurm jobs distribute
        # each inference batch without splitting or reordering the dataset.
        self.model = (
            self._CenterEmbedding(self.inference.model.model, self.center_window_size)
            .to(self.device)
            .eval()
        )
        self.multi_gpu = self.device.type == "cuda" and torch.cuda.device_count() > 1
        if self.multi_gpu:
            print(
                f"GPN-Star embedding inference uses DataParallel across "
                f"{torch.cuda.device_count()} visible GPUs"
            )
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(torch.cuda.device_count()))
            )

    def _resolve_chrom(self, chrom: str) -> str:
        candidates = [chrom]
        without_prefix = chrom.removeprefix("chr")
        if without_prefix not in candidates:
            candidates.append(without_prefix)
        prefixed = "chr" + without_prefix
        if prefixed not in candidates:
            candidates.append(prefixed)
        for candidate in candidates:
            if all(candidate in genome_msa.data for genome_msa in self.genome_msa_list):
                return candidate
        raise ValueError(
            f"Chromosome {chrom!r} is missing from one or more GPN-Star MSA stores"
        )

    def get_embedding_for_intervals(
        self, records: List[Dict[str, object]], verify_sequences: bool = True
    ) -> torch.Tensor:
        if not records:
            return torch.empty((0, 0), dtype=torch.float32)

        chroms = [self._resolve_chrom(str(record["chrom"])) for record in records]
        starts = np.asarray(
            [int(record["start"]) for record in records], dtype=np.int64
        )
        ends = np.asarray([int(record["end"]) for record in records], dtype=np.int64)
        for record, start, end in zip(records, starts, ends):
            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid GPN-Star interval {record.get('chrom')}:{start}-{end}"
                )
            if end - start < 1:
                raise ValueError(
                    f"GPN-Star interval {record.get('chrom')}:{start}-{end} is empty"
                )

        # GenomeMSA batches require equal interval lengths. Keep unlike lengths
        # in separate inference batches, then restore the original row order.
        # This is needed for variable-length URE and GenomicBenchmark tasks.
        indices_by_length: Dict[int, List[int]] = {}
        for index, (start, end) in enumerate(zip(starts, ends)):
            indices_by_length.setdefault(int(end - start), []).append(index)

        ordered_embeddings: Optional[torch.Tensor] = None
        for interval_length, row_indices in indices_by_length.items():
            effective_center_window = min(self.center_window_size, interval_length)
            model_module = self.model.module if self.multi_gpu else self.model
            model_module.center_window_size = effective_center_window
            group_chroms = [chroms[index] for index in row_indices]
            group_starts = starts[row_indices]
            group_ends = ends[row_indices]
            model_inputs = self.inference.tokenize_function(
                {"chrom": group_chroms, "start": group_starts, "end": group_ends}
            )

            if verify_sequences:
                expected_rows = []
                for index in row_indices:
                    record = records[index]
                    sequence = str(record["seq"]).upper()
                    if len(sequence) != interval_length:
                        raise ValueError(
                            "Sequence length does not match coordinates for "
                            f"{record.get('SeqID', record.get('chrom'))}"
                        )
                    expected_rows.append(
                        np.frombuffer(sequence.encode("ascii"), dtype="S1")
                    )
                expected = self.genome_msa_list[0].tokenizer(np.stack(expected_rows))
                target_species = model_inputs["input_ids_fwd"][:, :, 0]
                if target_species.shape != expected.shape:
                    raise ValueError(
                        "GPN-Star target sequence length does not match the benchmark "
                        f"sequence: {target_species.shape} != {expected.shape}"
                    )
                mismatches = target_species != expected
                if np.any(mismatches):
                    mismatch_counts = np.count_nonzero(mismatches, axis=1)
                    target_coverage = np.count_nonzero(target_species != 0, axis=1)
                    for local_index in np.flatnonzero(mismatch_counts):
                        index = row_indices[int(local_index)]
                        record = records[index]
                        self.sequence_mismatch_rows.append(
                            {
                                "sample_id": str(record.get("SeqID", index)),
                                "chrom": chroms[index],
                                "start": int(starts[index]),
                                "end": int(ends[index]),
                                "mismatch_bases": int(
                                    mismatch_counts[int(local_index)]
                                ),
                                "msa_target_coverage": float(
                                    target_coverage[int(local_index)]
                                    / target_species.shape[1]
                                ),
                            }
                        )

            tensor_inputs = {
                key: torch.as_tensor(value, device=self.device)
                for key, value in model_inputs.items()
            }
            with torch.inference_mode():
                group_embeddings = self.model(**tensor_inputs)
            # GPN-Star's official EmbeddingInference preserves a singleton
            # target-species axis: (batch, target_species, hidden_size).
            if group_embeddings.ndim == 3 and group_embeddings.shape[1] == 1:
                group_embeddings = group_embeddings.squeeze(1)
            if group_embeddings.ndim != 2 or group_embeddings.shape[0] != len(
                row_indices
            ):
                raise RuntimeError(
                    "Official GPN-Star embedding inference returned an unexpected shape: "
                    f"{tuple(group_embeddings.shape)}"
                )
            group_embeddings = group_embeddings.detach().cpu().to(torch.float32)
            if ordered_embeddings is None:
                ordered_embeddings = torch.empty(
                    (len(records), group_embeddings.shape[1]), dtype=torch.float32
                )
            elif ordered_embeddings.shape[1] != group_embeddings.shape[1]:
                raise RuntimeError("GPN-Star hidden size changed between length groups")
            ordered_embeddings[row_indices] = group_embeddings

        if ordered_embeddings is None:
            return torch.empty((0, 0), dtype=torch.float32)
        return ordered_embeddings


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
