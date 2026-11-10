"""联合消融共用的跟踪功能配置。"""

from dataclasses import replace

from .association import OCMConfig, OCRConfig
from .nsa import NSAConfig
from .replay import ORUConfig


def ablation_profiles(
    nsa: NSAConfig, *, ocm: OCMConfig = OCMConfig(), ocr: OCRConfig = OCRConfig(), oru: ORUConfig = ORUConfig(),
) -> dict:
    full = {"ocm": replace(ocm, mode="adaptive"), "ocr": ocr, "oru": oru, "nsa": nsa}
    return {
        "baseline": {},
        "nsa_only": {"nsa": nsa},
        "full": full,
        **{f"without_{feature}": {k: v for k, v in full.items() if k != feature} for feature in full},
        "fixed_ocm": {**full, "ocm": replace(full["ocm"], mode="fixed")},
    }
