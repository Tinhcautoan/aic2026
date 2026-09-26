"""The vi->en translator must be driven like its model card says (VinAI mBART: src_lang=vi_VN, decoder starts with
en_XX). Without that short queries degenerate ("biggest biggest biggest ... bird"). Fake model/tokenizer: no download."""
import types

import pytest
import torch

import transformers
from embedding.translate_fast import FastTranslator

EN_ID = 66750


class FakeTok:
    last_kwargs = None

    def __init__(self, **kw):
        FakeTok.last_kwargs = kw
        self.src_lang = kw.get("src_lang", "en_XX")
        self.lang_code_to_id = {"en_XX": EN_ID, "vi_VN": 66751}

    def __call__(self, batch, **kw):
        n = len(batch)
        return {"input_ids": torch.ones((n, 3), dtype=torch.long), "attention_mask": torch.ones((n, 3), dtype=torch.long)}

    def batch_decode(self, gen, skip_special_tokens=True):
        return [f"en-{i}" for i in range(len(gen))]

    def convert_tokens_to_ids(self, t):
        return self.lang_code_to_id[t]


class FakeModel:
    calls = []

    def __init__(self, model_type):
        self.config = types.SimpleNamespace(model_type=model_type)

    def eval(self):
        return self

    def generate(self, **kw):
        FakeModel.calls.append(kw)
        return torch.zeros((kw["input_ids"].shape[0], 2), dtype=torch.long)


@pytest.fixture()
def fake_hf(monkeypatch):
    state = {"model_type": "mbart"}
    monkeypatch.setattr(transformers, "AutoConfig", types.SimpleNamespace(
        from_pretrained=lambda p: types.SimpleNamespace(model_type=state["model_type"])))
    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda p, **kw: FakeTok(**kw)))
    monkeypatch.setattr(transformers, "AutoModelForSeq2SeqLM", types.SimpleNamespace(
        from_pretrained=lambda p: FakeModel(state["model_type"])))
    FakeModel.calls.clear()
    return state


def test_vinai_mbart_uses_vi_source_and_decoder_start(fake_hf):
    t = FastTranslator(model_name="vinai/vinai-translate-vi2en-v2", device="cpu", max_length=96, num_beams=3)
    out = t.translate_many(["chim to nhất vũ trụ", "a red car", "con mèo đen. Nó đang ngủ."])
    assert FakeTok.last_kwargs == {"src_lang": "vi_VN"}
    kw = FakeModel.calls[-1]
    assert kw["decoder_start_token_id"] == EN_ID and "forced_bos_token_id" not in kw
    assert kw["no_repeat_ngram_size"] == 3 and kw["num_beams"] == 3 and kw["early_stopping"] is True
    assert kw["max_new_tokens"] == 96 and kw["max_length"] is None
    assert out[1] == "a red car"                                     # English passes through untouched
    assert len(FakeModel.calls) == 1                                  # every Vietnamese sentence in ONE batched call


def test_single_translate_uses_the_same_settings(fake_hf):
    t = FastTranslator(model_name="vinai/vinai-translate-vi2en-v2", device="cpu", max_length=64, num_beams=1)
    t.translate("xe máy màu đỏ")
    kw = FakeModel.calls[-1]
    assert kw["decoder_start_token_id"] == EN_ID and "num_beams" not in kw and kw["max_new_tokens"] == 64


def test_nllb_keeps_forced_bos(fake_hf):
    fake_hf["model_type"] = "m2m_100"
    t = FastTranslator(model_name="facebook/nllb-200-distilled-600M", device="cpu", max_length=64, num_beams=2)
    t._load()
    assert FakeTok.last_kwargs == {"src_lang": "vie_Latn"}
    t._tokenizer.lang_code_to_id = {}
    t._tokenizer.convert_tokens_to_ids = lambda tok: 7 if tok == "eng_Latn" else -1
    t._forced_bos_token_id = t._detect_forced_bos()
    assert t._forced_bos_token_id == 7
