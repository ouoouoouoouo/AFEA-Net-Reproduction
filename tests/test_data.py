import os

import numpy as np
import pytest
import torch

from afea.data.iemocap import CLASSES, build_iemocap_manifest, check_counts, parse_emo_evaluation
from afea.data.ravdess import actor_fold
from afea.features import compute_fbank

EMO_TXT = """% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

[6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
C-E2:\tNeutral;\t()

[10.0100 - 11.3925]\tSes01F_impro01_F001\texc\t[2.5000, 2.5000, 2.5000]

[14.8872 - 18.0175]\tSes01F_impro01_M002\tfru\t[2.5000, 3.5000, 3.5000]

[19.2900 - 20.7875]\tSes01F_impro01_M003\txxx\t[2.5000, 3.5000, 3.5000]

[21.3257 - 24.7400]\tSes01F_impro01_F004\thap\t[3.0, 3.0, 3.0]
"""


def _fake_iemocap(tmp_path):
    for s in range(1, 6):
        d = tmp_path / f"Session{s}" / "dialog" / "EmoEvaluation"
        d.mkdir(parents=True)
        (d / f"Ses0{s}F_impro01.txt").write_text(EMO_TXT.replace("Ses01", f"Ses0{s}"))
    return str(tmp_path)


def test_parse_emo_evaluation(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(EMO_TXT)
    items = parse_emo_evaluation(str(p))
    assert [i["raw_label"] for i in items] == ["neu", "exc", "fru", "xxx", "hap"]


def test_manifest_merges_excited_into_happy(tmp_path):
    rows = build_iemocap_manifest(_fake_iemocap(tmp_path), strict=False)
    assert len(rows) == 15                          # neu, exc, hap per session; fru/xxx dropped
    assert {r["label"] for r in rows} == {"neu", "hap"}
    exc = [r for r in rows if r["utt_id"].endswith("F001")]
    assert all(r["label"] == "hap" and r["label_id"] == CLASSES.index("hap") for r in exc)
    assert sorted({r["fold"] for r in rows}) == [1, 2, 3, 4, 5]
    assert rows[0]["path"].endswith(os.path.join("Session1", "sentences", "wav", "Ses01F_impro01",
                                                 "Ses01F_impro01_F000.wav"))


def test_strict_count_check_rejects_wrong_subset(tmp_path):
    with pytest.raises(ValueError):
        build_iemocap_manifest(_fake_iemocap(tmp_path), strict=True)


def test_expected_counts_pass():
    rows = ([{"label": "sad"}] * 1084 + [{"label": "hap"}] * 1636 +
            [{"label": "ang"}] * 1103 + [{"label": "neu"}] * 1708)
    assert sum(check_counts(rows).values()) == 5531


def test_ravdess_actor_folds():
    folds = [actor_fold(a) for a in range(1, 25)]
    assert sorted(set(folds)) == [1, 2, 3, 4, 5]
    assert [folds.count(k) for k in range(1, 6)] == [5, 5, 5, 5, 4]


def test_fbank_40d_25ms_10ms():
    wav = torch.randn(16000)                         # 1 s at 16 kHz
    fb = compute_fbank(wav)
    assert fb.shape == (98, 40)                      # 1 + (16000 - 400) // 160 frames
    assert np.allclose(fb.mean(0).numpy(), 0, atol=1e-4)
