# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from loguru import logger

from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.config_reader import (
    get_ocr_hook_enable,
    get_ocr_hook_langs,
    get_ocr_hook_provider,
    get_ocr_hook_vlm_block_types,
)


@dataclass
class OCRHookResult:
    text: str
    score: float


def _normalize_lang(lang: str | None) -> str:
    return (lang or "").strip().lower()


def _should_run_hook(lang: str | None) -> bool:
    if not get_ocr_hook_enable():
        return False
    normalized_lang = _normalize_lang(lang)
    if not normalized_lang:
        return False
    return normalized_lang in get_ocr_hook_langs()


class VietnameseOCRHookProvider:
    """Default hook provider. Uses existing OCR engine with vi language."""

    def __init__(self):
        self.atom_model_manager = AtomModelSingleton()

    def recognize_crops(self, image_crops: list[np.ndarray], lang: str) -> list[OCRHookResult]:
        if not image_crops:
            return []
        ocr_model = self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=lang,
        )
        ocr_res_list = ocr_model.ocr(image_crops, det=False, tqdm_enable=True)[0]
        results: list[OCRHookResult] = []
        for item in ocr_res_list:
            if not item:
                results.append(OCRHookResult(text="", score=0.0))
                continue
            text = item[0] if len(item) > 0 else ""
            score = float(item[1]) if len(item) > 1 else 0.0
            results.append(OCRHookResult(text=text, score=score))
        return results

    def recognize_vlm_blocks(
        self,
        page_blocks: list[dict[str, Any]],
        page_image_rgb: np.ndarray,
        lang: str,
    ) -> None:
        if not page_blocks:
            return
        image_h, image_w = page_image_rgb.shape[:2]
        target_block_types = get_ocr_hook_vlm_block_types()
        target_blocks: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        for block in page_blocks:
            if block.get("type") not in target_block_types:
                continue
            bbox = block.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x0 = int(max(0, min(image_w, round(float(bbox[0]) * image_w))))
            y0 = int(max(0, min(image_h, round(float(bbox[1]) * image_h))))
            x1 = int(max(0, min(image_w, round(float(bbox[2]) * image_w))))
            y1 = int(max(0, min(image_h, round(float(bbox[3]) * image_h))))
            if x1 <= x0 or y1 <= y0:
                continue
            crop_rgb = page_image_rgb[y0:y1, x0:x1]
            if crop_rgb.size == 0:
                continue
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            target_blocks.append(block)
            crops.append(crop_bgr)

        if not crops:
            return

        results = self.recognize_crops(crops, lang=lang)
        for block, result in zip(target_blocks, results):
            if result.text:
                block["content"] = result.text


class VietOCRProvider:
    """Optional VietOCR provider with lazy initialization."""

    def __init__(self):
        self._predictor = None
        self._init_error: str | None = None

    def _ensure_predictor(self) -> bool:
        if self._predictor is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            from vietocr.tool.predictor import Predictor  # type: ignore
            from vietocr.tool.config import Cfg  # type: ignore

            config = Cfg.load_config_from_name("vgg_transformer")
            config["cnn"]["pretrained"] = False
            config["device"] = "cpu"
            self._predictor = Predictor(config)
            return True
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning(f"VietOCR init failed: {exc}")
            return False

    def recognize_crops(self, image_crops: list[np.ndarray], lang: str) -> list[OCRHookResult]:
        if not image_crops:
            return []
        if not self._ensure_predictor():
            raise RuntimeError(f"VietOCR unavailable: {self._init_error}")
        from PIL import Image

        results: list[OCRHookResult] = []
        for crop_bgr in image_crops:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            text = self._predictor.predict(pil_img)
            results.append(OCRHookResult(text=text or "", score=1.0))
        return results

    def recognize_vlm_blocks(
        self,
        page_blocks: list[dict[str, Any]],
        page_image_rgb: np.ndarray,
        lang: str,
    ) -> None:
        if not page_blocks:
            return
        image_h, image_w = page_image_rgb.shape[:2]
        target_block_types = get_ocr_hook_vlm_block_types()
        target_blocks: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        for block in page_blocks:
            if block.get("type") not in target_block_types:
                continue
            bbox = block.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x0 = int(max(0, min(image_w, round(float(bbox[0]) * image_w))))
            y0 = int(max(0, min(image_h, round(float(bbox[1]) * image_h))))
            x1 = int(max(0, min(image_w, round(float(bbox[2]) * image_w))))
            y1 = int(max(0, min(image_h, round(float(bbox[3]) * image_h))))
            if x1 <= x0 or y1 <= y0:
                continue
            crop_rgb = page_image_rgb[y0:y1, x0:x1]
            if crop_rgb.size == 0:
                continue
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            target_blocks.append(block)
            crops.append(crop_bgr)
        results = self.recognize_crops(crops, lang=lang)
        for block, result in zip(target_blocks, results):
            if result.text:
                block["content"] = result.text


class OCRHookDispatcher:
    def __init__(self):
        self.provider_name = get_ocr_hook_provider()
        self.provider = self._build_provider(self.provider_name)
        self._fallback_provider = VietnameseOCRHookProvider()

    @staticmethod
    def _build_provider(provider_name: str):
        if provider_name == "vietnamese_default":
            return VietnameseOCRHookProvider()
        if provider_name == "vietocr":
            return VietOCRProvider()
        logger.warning(f"Unknown OCR hook provider: {provider_name}, fallback to vietnamese_default")
        return VietnameseOCRHookProvider()

    def run_recognition(
        self,
        image_crops: list[np.ndarray],
        lang: str | None,
    ) -> list[OCRHookResult] | None:
        if not _should_run_hook(lang):
            return None
        try:
            return self.provider.recognize_crops(image_crops, lang=_normalize_lang(lang))
        except Exception as exc:
            logger.warning(f"OCR hook provider {self.provider_name} failed: {exc}")
            try:
                return self._fallback_provider.recognize_crops(image_crops, lang=_normalize_lang(lang))
            except Exception as fallback_exc:
                logger.warning(f"OCR hook fallback provider failed, use pipeline default OCR flow: {fallback_exc}")
                return None

    def run_vlm_postprocess(
        self,
        page_blocks: list[dict[str, Any]],
        page_image_rgb: np.ndarray,
        lang: str | None,
    ) -> bool:
        if not _should_run_hook(lang):
            return False
        try:
            self.provider.recognize_vlm_blocks(
                page_blocks=page_blocks,
                page_image_rgb=page_image_rgb,
                lang=_normalize_lang(lang),
            )
            return True
        except Exception as exc:
            logger.warning(f"VLM OCR hook provider {self.provider_name} failed: {exc}")
            try:
                self._fallback_provider.recognize_vlm_blocks(
                    page_blocks=page_blocks,
                    page_image_rgb=page_image_rgb,
                    lang=_normalize_lang(lang),
                )
                return True
            except Exception as fallback_exc:
                logger.warning(f"VLM OCR hook fallback failed, keep original VLM output: {fallback_exc}")
                return False


_DISPATCHER: OCRHookDispatcher | None = None


def get_ocr_hook_dispatcher() -> OCRHookDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = OCRHookDispatcher()
    return _DISPATCHER


def reset_ocr_hook_dispatcher() -> None:
    global _DISPATCHER
    _DISPATCHER = None
