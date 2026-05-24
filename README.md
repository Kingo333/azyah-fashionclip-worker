# azyah-fashionclip-worker

FashionCLIP wardrobe analysis worker for Azyah.

- Model: `Marqo/marqo-fashionSigLIP`
- Framework: FastAPI on port `PORT` (default `8000`)
- Purpose: analyze ONE wardrobe item image and return structured garment metadata plus a directive `prompt_hint` for Live Cam.
- Called once per uploaded wardrobe item. NOT called per live frame.
- Failure must return clean JSON. It must never affect the FluxRT Live Cam endpoint.

This worker is fully independent of `fluxrt-serverless`. It is built and deployed from this repo only.

## Endpoints

### `GET /ping`

Returns model and device status. No auth.

```json
{
  "ok": true,
  "model_name": "Marqo/marqo-fashionSigLIP",
  "device": "cuda",
  "model_loaded": true,
  "model_load_error": null,
  "min_confidence": 0.38,
  "min_margin": 0.07,
  "allowlist_enabled": false
}
```

### `POST /analyze`

Headers:

```
X-Worker-Token: <WORKER_TOKEN>
Content-Type: application/json
```

Body:

```json
{
  "wardrobe_item_id": "string",
  "image_url": "string",
  "category_hint": "optional string"
}
```

`category_hint` accepted values (case insensitive): `top`, `bottom`, `dress`, `outerwear`, `shoes`, `bag`, `accessory`, `swimwear`, `underwear`. Common aliases such as `tops`, `jeans`, `jacket`, `footwear`, `handbag` are also normalized.

Response (success):

```json
{
  "wardrobe_item_id": "...",
  "image_hash": "...",
  "model_name": "Marqo/marqo-fashionSigLIP",
  "metadata": {
    "category": "...",
    "raw_category_label": "...",
    "item_type_phrase": "...",
    "sleeve_length": "...",
    "garment_length": "...",
    "pattern_type": "...",
    "material_appearance": "...",
    "fit_or_silhouette": "...",
    "logo_or_text": true,
    "body_region_to_replace": "...",
    "body_regions_to_preserve": [],
    "confidence": 0.0,
    "category_hint_used": "applied | overridden | none",
    "scores": {}
  },
  "prompt_hint": "Use the reference image as the source of truth. Apply a ...",
  "confidence": 0.0
}
```

Labels below the confidence or margin threshold are returned as `"unknown"`. The worker does not guess.

## Confidence and margin gating

A label is accepted only if BOTH:

- top probability >= `MIN_CONFIDENCE` (default `0.38`)
- top-1 minus top-2 >= `MIN_MARGIN` (default `0.07`)

Otherwise the field is `"unknown"`. This prevents inventing details like long sleeves when the input is ambiguous.

## category_hint behavior

If `category_hint` maps to a known wardrobe class (top / bottom / dress / outerwear / shoes / bag / accessory / swimwear / underwear), category classification is restricted to that subset. FashionCLIP can override the hint only when its full-set top class is OUTSIDE the hinted subset AND its confidence is clearly higher (>= 0.55 and at least 0.15 above the hinted subset's top class). Sleeve length, garment length, pattern, material, and fit are still scored by FashionCLIP without hint bias.

## prompt_hint behavior

`prompt_hint` is a full directive instruction sentence, not a label list. It uses garment-class-specific preservation rules:

- top: replace torso and sleeves only, preserve pants/bottoms/legs/shoes/face/hair.
- bottom: replace lower-body clothing only, preserve top/shirt/torso/arms/shoes.
- dress: replace full dress region, preserve face/hair/hands/shoes, do not alter dress length or sleeve length.
- outerwear: replace outer layer only, preserve inner top/pants/shoes.
- shoes: replace footwear only, preserve pants/legs/bottoms/torso/top.
- fullbody (swimwear/underwear): replace torso and upper legs only, preserve face/hair/hands/feet.
- accessory: add accessory only, preserve all clothing and body.

## Environment variables (set in RunPod — never commit values)

| Name | Default | Purpose |
| --- | --- | --- |
| `WORKER_TOKEN` | _required_ | Shared secret. Requests must send `X-Worker-Token`. |
| `MODEL_NAME` | `Marqo/marqo-fashionSigLIP` | HF model id. |
| `MIN_CONFIDENCE` | `0.38` | Min top probability to accept a label. |
| `MIN_MARGIN` | `0.07` | Min margin between top-1 and top-2 to accept a label. |
| `PORT` | `8000` | HTTP port. |
| `ALLOWED_IMAGE_HOSTS` | _empty_ | Comma-separated allowlist of image hostnames. Empty = allow all. |
| `IMAGE_FETCH_TIMEOUT_S` | `10` | Image fetch timeout in seconds. |
| `MAX_IMAGE_BYTES` | `15728640` | Max image size in bytes. |
| `HF_TOKEN` | _optional_ | Only if the HF model later becomes gated. |

## Compatibility note (Marqo/marqo-fashionSigLIP)

Inference follows the official Hugging Face model card pattern:

- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`
- `model.get_image_features(pixel_values, normalize=True)`
- `model.get_text_features(input_ids, normalize=True)`
- similarity: `(100.0 * image_features @ text_features.T).softmax(dim=-1)`

The worker does NOT rely on `outputs.logits_per_image`. A `/ping` and `/analyze` smoke test should be run immediately after the first RunPod deploy to confirm `trust_remote_code` resolves correctly inside the container.

## Build (local)

```
docker build -t azyah-fashionclip-worker:dev .
docker run --rm -p 8000:8000 \
  -e WORKER_TOKEN=dev \
  azyah-fashionclip-worker:dev
```

## GHCR images

CI publishes:

- `ghcr.io/kingo333/azyah-fashionclip-worker:fashionclip-worker-v1`
- `ghcr.io/kingo333/azyah-fashionclip-worker:fashionclip-worker-v1-<sha>`

The GHCR package is intended to be Public so RunPod can pull without credentials. If after the first successful push the package is still Private, change visibility at:

`https://github.com/users/Kingo333/packages/container/azyah-fashionclip-worker/settings` → "Change package visibility" → Public.

## Safety

- No tokens, base64, signed URLs, or image bytes are ever logged.
- Image fetches are timeout-bounded and optionally allowlisted via `ALLOWED_IMAGE_HOSTS`.
- Worker failures return JSON errors and never crash the host.
- This worker is completely independent of `fluxrt-serverless` and the FluxRT Live Cam RunPod endpoint.
