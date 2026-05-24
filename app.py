"""
Azyah FashionCLIP Wardrobe Analysis Worker

FastAPI worker that scores a single wardrobe-item image against
Marqo/marqo-fashionSigLIP label groups and returns structured
garment metadata + a directive prompt_hint for Live Cam.

Run once per uploaded wardrobe item. NOT per live frame.
Failures must return clean JSON. They must NEVER affect Live Cam.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "Marqo/marqo-fashionSigLIP")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.38"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.07"))
PORT = int(os.getenv("PORT", "8000"))
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
ALLOWED_IMAGE_HOSTS_RAW = os.getenv("ALLOWED_IMAGE_HOSTS", "").strip()
IMAGE_FETCH_TIMEOUT_S = float(os.getenv("IMAGE_FETCH_TIMEOUT_S", "10"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))

ALLOWED_HOSTS: List[str] = [
    h.strip().lower() for h in ALLOWED_IMAGE_HOSTS_RAW.split(",") if h.strip()
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fashionclip-worker")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNKNOWN = "unknown"

# -----------------------------------------------------------------------------
# Label groups
# -----------------------------------------------------------------------------

CATEGORY_LABELS: List[str] = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie",
    "jacket", "coat",
    "dress",
    "skirt", "pants", "jeans", "shorts",
    "activewear top", "activewear bottom",
    "swimwear", "underwear",
    "shoes",
    "bag", "accessory",
]

GROUP_LABELS: Dict[str, List[str]] = {
    "sleeve_length": [
        "sleeveless", "short sleeve", "three-quarter sleeve", "long sleeve",
        "not applicable",
    ],
    "garment_length": [
        "cropped", "regular", "tunic", "knee length", "midi", "maxi",
        "not applicable",
    ],
    "pattern_type": [
        "solid", "striped", "checked", "plaid", "floral",
        "graphic print", "logo print", "abstract", "color block",
    ],
    "material_appearance": [
        "cotton", "denim", "knit", "wool", "leather",
        "satin or silk", "linen", "fleece", "synthetic", "lace", "sheer",
    ],
    "fit_or_silhouette": [
        "slim fit", "regular fit", "relaxed fit", "oversized",
        "flared", "bodycon", "a-line", "straight",
    ],
}

HINT_TO_SUBSET: Dict[str, List[str]] = {
    "top": ["t-shirt", "shirt", "blouse", "sweater", "hoodie", "activewear top"],
    "bottom": ["pants", "jeans", "shorts", "skirt", "activewear bottom"],
    "dress": ["dress"],
    "outerwear": ["jacket", "coat"],
    "shoes": ["shoes"],
    "bag": ["bag"],
    "accessory": ["accessory"],
    "swimwear": ["swimwear"],
    "underwear": ["underwear"],
}

CATEGORY_TO_BODY_REGION: Dict[str, Dict[str, object]] = {
    "t-shirt":           {"replace": "torso", "preserve": ["head", "arms_lower", "legs", "feet"]},
    "shirt":             {"replace": "torso", "preserve": ["head", "legs", "feet"]},
    "blouse":            {"replace": "torso", "preserve": ["head", "legs", "feet"]},
    "sweater":           {"replace": "torso", "preserve": ["head", "legs", "feet"]},
    "hoodie":            {"replace": "torso", "preserve": ["head", "legs", "feet"]},
    "jacket":            {"replace": "torso_outer", "preserve": ["head", "legs", "feet"]},
    "coat":              {"replace": "torso_outer", "preserve": ["head", "feet"]},
    "dress":             {"replace": "torso_and_legs", "preserve": ["head", "feet"]},
    "skirt":             {"replace": "legs_upper", "preserve": ["head", "torso", "feet"]},
    "pants":             {"replace": "legs", "preserve": ["head", "torso", "feet"]},
    "jeans":             {"replace": "legs", "preserve": ["head", "torso", "feet"]},
    "shorts":            {"replace": "legs_upper", "preserve": ["head", "torso", "feet"]},
    "activewear top":    {"replace": "torso", "preserve": ["head", "legs", "feet"]},
    "activewear bottom": {"replace": "legs", "preserve": ["head", "torso", "feet"]},
    "swimwear":          {"replace": "torso_and_legs_upper", "preserve": ["head", "feet"]},
    "underwear":         {"replace": "torso_and_legs_upper", "preserve": ["head", "feet"]},
    "shoes":             {"replace": "feet", "preserve": ["head", "torso", "legs"]},
    "bag":               {"replace": "none", "preserve": ["head", "torso", "legs", "feet"]},
    "accessory":         {"replace": "none", "preserve": ["head", "torso", "legs", "feet"]},
}


def garment_class(category: str) -> str:
    if category in ("t-shirt", "shirt", "blouse", "sweater", "hoodie", "activewear top"):
        return "top"
    if category in ("pants", "jeans", "shorts", "skirt", "activewear bottom"):
        return "bottom"
    if category == "dress":
        return "dress"
    if category in ("jacket", "coat"):
        return "outerwear"
    if category in ("swimwear", "underwear"):
        return "fullbody"
    if category == "shoes":
        return "shoes"
    if category in ("bag", "accessory"):
        return "accessory"
    return "unknown"


_model = None
_processor = None
_model_load_error: Optional[str] = None
_model_loaded_at: Optional[float] = None


def get_model() -> Tuple[object, object]:
    global _model, _processor, _model_load_error, _model_loaded_at
    if _model is not None and _processor is not None:
        return _model, _processor
    try:
        t0 = time.time()
        log.info("Loading model %s on %s", MODEL_NAME, DEVICE)
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        _model.eval()
        _model_loaded_at = time.time()
        log.info("Model loaded in %.1fs", _model_loaded_at - t0)
        return _model, _processor
    except Exception as e:  # noqa: BLE001
        _model_load_error = type(e).__name__
        log.exception("Model load failed (%s)", _model_load_error)
        raise


class AnalyzeRequest(BaseModel):
    wardrobe_item_id: str = Field(..., min_length=1, max_length=128)
    image_url: str = Field(..., min_length=1)
    category_hint: Optional[str] = None


def _check_token(provided: Optional[str]) -> None:
    if not WORKER_TOKEN:
        raise HTTPException(status_code=503, detail="worker_token_not_configured")
    if not provided or provided != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def _host_allowed(url: str) -> bool:
    if not ALLOWED_HOSTS:
        return True
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


async def _fetch_image(url: str) -> Tuple[bytes, str]:
    if not _host_allowed(url):
        raise HTTPException(status_code=400, detail="image_host_not_allowed")
    async with httpx.AsyncClient(timeout=IMAGE_FETCH_TIMEOUT_S, follow_redirects=True) as c:
        r = await c.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"image_fetch_status_{r.status_code}")
    data = r.content
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="image_too_large")
    sha = hashlib.sha256(data).hexdigest()
    return data, sha


def _normalize_hint(hint: Optional[str]) -> Optional[str]:
    if not hint:
        return None
    h = hint.strip().lower()
    aliases = {
        "tops": "top", "shirt": "top", "t-shirt": "top", "tshirt": "top",
        "blouse": "top", "sweater": "top", "hoodie": "top",
        "bottoms": "bottom", "pants": "bottom", "jeans": "bottom",
        "shorts": "bottom", "skirt": "bottom",
        "dresses": "dress",
        "jacket": "outerwear", "coat": "outerwear", "outer": "outerwear",
        "shoe": "shoes", "footwear": "shoes",
        "bags": "bag", "purse": "bag", "handbag": "bag",
        "accessories": "accessory",
    }
    if h in HINT_TO_SUBSET:
        return h
    if h in aliases:
        return aliases[h]
    return None


def _score_labels(
    image_inputs: dict,
    labels: List[str],
    template: str,
) -> Tuple[str, float, float, Dict[str, float]]:
    model, processor = get_model()
    prompts = [template.format(label=lbl) for lbl in labels]
    text_inputs = processor(
        text=prompts, padding="max_length", return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        image_features = model.get_image_features(
            image_inputs["pixel_values"], normalize=True
        )
        text_features = model.get_text_features(
            text_inputs["input_ids"], normalize=True
        )
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    sorted_idx = torch.argsort(probs, descending=True)
    top_i = int(sorted_idx[0].item())
    top_conf = float(probs[top_i].item())
    second_conf = float(probs[int(sorted_idx[1].item())].item()) if len(labels) > 1 else 0.0
    margin = top_conf - second_conf
    all_scores = {labels[i]: float(probs[i].item()) for i in range(len(labels))}
    return labels[top_i], top_conf, margin, all_scores


def _gate(label: str, conf: float, margin: float) -> str:
    if conf >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
        return label
    return UNKNOWN


def _build_prompt_hint(
    raw_category: str,
    metadata: Dict[str, object],
) -> str:
    gclass = garment_class(raw_category)
    category = metadata.get("category")
    sleeve = metadata.get("sleeve_length")
    length = metadata.get("garment_length")
    pattern = metadata.get("pattern_type")
    material = metadata.get("material_appearance")
    fit = metadata.get("fit_or_silhouette")

    descriptors: List[str] = []
    if sleeve and sleeve not in (UNKNOWN, "not applicable"):
        descriptors.append(str(sleeve))
    if length and length not in (UNKNOWN, "not applicable"):
        descriptors.append(str(length))
    if fit and fit not in (UNKNOWN,):
        descriptors.append(str(fit))
    if pattern and pattern not in (UNKNOWN,):
        descriptors.append(str(pattern))
    if material and material not in (UNKNOWN,):
        descriptors.append(str(material))

    noun = category if category and category != UNKNOWN else "garment"
    phrase = (" ".join(descriptors) + " " + str(noun)).strip()

    if gclass == "top":
        preserve = ("Replace only the torso and sleeves. Preserve pants, "
                    "bottoms, legs, shoes, face, and hair.")
        invariants = ("sleeve length, neckline, hem, cuffs, pattern placement, "
                      "material appearance, fit, and silhouette")
    elif gclass == "bottom":
        preserve = ("Replace only the lower-body clothing. Preserve the existing "
                    "top, shirt, torso, arms, shoes, face, and hair.")
        invariants = ("garment length, waistline, hem, pattern placement, "
                      "material appearance, fit, and silhouette")
    elif gclass == "dress":
        preserve = ("Replace the full dress region. Preserve face, hair, hands, "
                    "and shoes. Do not alter dress length or sleeve length.")
        invariants = ("sleeve length, full garment length, neckline, hem, "
                      "pattern placement, material appearance, fit, and silhouette")
    elif gclass == "outerwear":
        preserve = ("Replace only the outer layer over the torso. Preserve the "
                    "inner top, pants, bottoms, shoes, face, and hair.")
        invariants = ("sleeve length, garment length, lapels, closures, "
                      "pattern placement, material appearance, fit, and silhouette")
    elif gclass == "shoes":
        preserve = ("Replace only footwear. Preserve pants, legs, bottoms, "
                    "torso, top, face, hair, and the rest of the outfit.")
        invariants = ("shoe shape, sole, color, material appearance, and silhouette")
    elif gclass == "fullbody":
        preserve = ("Replace the torso and upper legs region only. "
                    "Preserve face, hair, hands, and feet.")
        invariants = ("garment length, neckline, pattern placement, material "
                      "appearance, fit, and silhouette")
    elif gclass == "accessory":
        preserve = ("Add the accessory only. Preserve all clothing, body, face, "
                    "hair, and the rest of the outfit unchanged.")
        invariants = ("shape, color, material appearance, and proportions")
    else:
        preserve = ("Replace only the relevant garment region. Preserve the rest "
                    "of the outfit, body, face, and hair.")
        invariants = ("pattern placement, material appearance, fit, and silhouette")

    return (
        "Use the reference image as the source of truth. "
        f"Apply a {phrase}. "
        f"Preserve {invariants}. "
        f"{preserve} "
        "Do not simplify, redesign, or invent a different garment."
    )


app = FastAPI(title="azyah-fashionclip-worker", version="1")


@app.get("/ping")
def ping() -> dict:
    return {
        "ok": True,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": _model is not None,
        "model_load_error": _model_load_error,
        "min_confidence": MIN_CONFIDENCE,
        "min_margin": MIN_MARGIN,
        "allowlist_enabled": bool(ALLOWED_HOSTS),
    }


@app.post("/analyze")
async def analyze(
    req: AnalyzeRequest,
    x_worker_token: Optional[str] = Header(default=None, alias="X-Worker-Token"),
) -> JSONResponse:
    _check_token(x_worker_token)
    try:
        img_bytes, sha = await _fetch_image(req.image_url)
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="image_decode_failed")

        _, processor = get_model()
        image_inputs = processor(images=pil, return_tensors="pt").to(DEVICE)

        TEMPLATE = "a photo of a {label} garment"

        norm_hint = _normalize_hint(req.category_hint)
        hinted_subset = HINT_TO_SUBSET.get(norm_hint) if norm_hint else None

        full_top, full_top_conf, full_margin, full_scores = _score_labels(
            image_inputs, CATEGORY_LABELS, TEMPLATE
        )

        if hinted_subset:
            sub_top, sub_top_conf, sub_margin, sub_scores = _score_labels(
                image_inputs, hinted_subset, TEMPLATE
            )
            override = (
                full_top not in hinted_subset
                and full_top_conf >= max(0.55, sub_top_conf + 0.15)
                and full_margin >= MIN_MARGIN
            )
            if override:
                raw_cat = full_top
                raw_cat_conf = full_top_conf
                raw_cat_margin = full_margin
                cat_scores = full_scores
                hint_used = "overridden"
            else:
                raw_cat = sub_top
                raw_cat_conf = sub_top_conf
                raw_cat_margin = sub_margin
                cat_scores = sub_scores
                hint_used = "applied"
        else:
            raw_cat = full_top
            raw_cat_conf = full_top_conf
            raw_cat_margin = full_margin
            cat_scores = full_scores
            hint_used = "none"

        category = _gate(raw_cat, raw_cat_conf, raw_cat_margin)

        groups_out: Dict[str, dict] = {}
        for group, labels in GROUP_LABELS.items():
            top, conf, margin, scores = _score_labels(image_inputs, labels, TEMPLATE)
            groups_out[group] = {
                "label": _gate(top, conf, margin),
                "raw_label": top,
                "confidence": conf,
                "margin": margin,
            }

        logo_top, logo_conf, logo_margin, _ = _score_labels(
            image_inputs,
            ["with visible logo or text", "without visible logo or text"],
            "a clothing photo {label}",
        )
        logo_gate = _gate(logo_top, logo_conf, logo_margin)
        logo_or_text = (
            True  if logo_gate == "with visible logo or text"
            else False if logo_gate == "without visible logo or text"
            else False
        )

        body_map = CATEGORY_TO_BODY_REGION.get(
            raw_cat, {"replace": "unknown", "preserve": []}
        )

        sleeve   = groups_out["sleeve_length"]["label"]
        length   = groups_out["garment_length"]["label"]
        pattern  = groups_out["pattern_type"]["label"]
        material = groups_out["material_appearance"]["label"]
        fit      = groups_out["fit_or_silhouette"]["label"]

        bits: List[str] = []
        for v in (fit, material, length, sleeve, category):
            if v and v not in (UNKNOWN, "not applicable"):
                bits.append(v)
        item_type_phrase = " ".join(bits).strip()

        overall_conf = round(raw_cat_conf, 4)

        metadata = {
            "category": category,
            "raw_category_label": raw_cat,
            "item_type_phrase": item_type_phrase,
            "sleeve_length": sleeve,
            "garment_length": length,
            "pattern_type": pattern,
            "material_appearance": material,
            "fit_or_silhouette": fit,
            "logo_or_text": logo_or_text,
            "body_region_to_replace": body_map["replace"],
            "body_regions_to_preserve": body_map["preserve"],
            "confidence": overall_conf,
            "category_hint_used": hint_used,
            "scores": {
                "category": {
                    "confidence": round(raw_cat_conf, 4),
                    "margin": round(raw_cat_margin, 4),
                    "top5": {
                        k: round(v, 4)
                        for k, v in sorted(cat_scores.items(), key=lambda kv: -kv[1])[:5]
                    },
                },
                **{
                    g: {
                        "confidence": round(groups_out[g]["confidence"], 4),
                        "margin": round(groups_out[g]["margin"], 4),
                    }
                    for g in groups_out
                },
            },
        }

        prompt_hint = _build_prompt_hint(raw_cat, metadata)

        return JSONResponse(
            {
                "wardrobe_item_id": req.wardrobe_item_id,
                "image_hash": sha,
                "model_name": MODEL_NAME,
                "metadata": metadata,
                "prompt_hint": prompt_hint,
                "confidence": overall_conf,
            }
        )

    except HTTPException as he:
        log.warning("analyze failed: %s", he.detail)
        return JSONResponse(
            status_code=he.status_code,
            content={
                "error": str(he.detail),
                "error_type": "http_exception",
                "wardrobe_item_id": req.wardrobe_item_id,
            },
        )
    except Exception as e:  # noqa: BLE001
        log.exception("analyze crashed: %s", type(e).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "error_type": type(e).__name__,
                "wardrobe_item_id": req.wardrobe_item_id,
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
