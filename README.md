# AnemiCheck

Non-invasive anemia screening from a photo of the palpebral conjunctiva,
using a U-Net (ResNet34 encoder) for segmentation and an EfficientNet-B3
classifier for diagnosis.

## What changed from the original `app.py`

**Bugs fixed:**
- Model filename mismatch: code looked for `unet_model.pth`, the uploaded
  file was `Unet_model_.pth` — this would crash on load. Standardized to
  `models/unet_model.pth` and `models/efficientnet_b3.pth`.
- No headless matplotlib backend — would crash on any server without a
  display. Added `matplotlib.use("Agg")`.
- `torch.load()` without `weights_only=True` — could execute arbitrary code
  from a malicious checkpoint file. Now loads in safe/weights-only mode.
- Hardcoded LAN IP shown to every user (`192.168.1.107:8501`) — removed;
  now an optional `LOCAL_NETWORK_HINT` env var, empty by default.
- No file-size/type validation, no corrupted-image handling, no handling
  for "conjunctiva not found" — added explicit checks and user-facing error
  messages instead of letting the app stack-trace.
- Matplotlib figures were never closed (`plt.close(fig)`) — would leak
  memory over a long-running server process with many users.
- Unpinned dependencies — pinned versions in `requirements.txt` so the
  environment is reproducible.
- `models.efficientnet_b3(pretrained=False)` used the deprecated
  `pretrained=` argument — switched to `weights=None`.
- A `<script>` block called `getUserMedia()` on every page load just to log
  to console — removed; `st.camera_input` already requests camera access
  when actually used.

**One issue I did *not* silently fix — please read this:**
The classifier's raw sigmoid output appears inverted relative to the
intended label (the original code force-flipped it with `1 - prediction`,
with a comment acknowledging this). I kept that behavior so the app's
output doesn't change, but moved it to an explicit, documented flag:
`config.INVERT_CLASSIFIER_OUTPUT` (default `true`). This is a workaround,
not a fix — for a medical screening tool, the right move is to check the
label mapping used during training (class 0/1 swapped somewhere in the
DataLoader or final layer) and retrain/re-export so this flag is no longer
needed. Shipping an unexplained "flip the diagnosis" patch to production is
worth fixing properly before this goes in front of real users.

## Project structure

```
.
├── app.py              # Streamlit UI and request flow
├── config.py            # All environment-variable-driven settings
├── model_loader.py      # Cached, weights_only-safe model loading
├── inference.py          # Mask cleanup, conjunctiva crop, prediction
├── requirements.txt
├── Dockerfile
├── render.yaml          # Render.com deploy config
├── Procfile              # Railway / Heroku-style platforms
├── .streamlit/config.toml
├── logo.png
└── models/
    ├── unet_model.pth
    └── efficientnet_b3.pth
```

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `UNET_WEIGHTS_PATH` | `models/unet_model.pth` | Override model file location |
| `CLASSIFIER_WEIGHTS_PATH` | `models/efficientnet_b3.pth` | Override model file location |
| `LOGO_PATH` | `logo.png` | Override logo location |
| `FORCE_CPU` | `false` | Force CPU inference even if a GPU is visible |
| `INVERT_CLASSIFIER_OUTPUT` | `true` | See note above — keep `true` until the model is retrained |
| `LOCAL_NETWORK_HINT` | *(empty)* | Optional LAN URL hint for local demos; leave unset in production |
| `MASK_MIN_AREA` | `500` | Minimum contour area kept during mask cleanup |
| `BBOX_PADDING` | `15` | Padding (px) around the detected conjunctiva crop |

## Deployment — read this before picking a platform

This is a **stateful Python app holding two PyTorch models in memory**
(~140 MB of weights, plus PyTorch/torchvision/OpenCV themselves). That
ruled out a few platforms the prompt asked about, and why:

- **Vercel / Netlify / GitHub Pages** — these serve static sites and
  short-lived serverless functions. They cannot run a long-lived Python
  process, don't support this dependency stack, and have function/payload
  size limits far below what PyTorch + the model weights need. Deploying
  there isn't a config problem — it's the wrong category of platform.

What actually fits, in order of effort:

1. **Hugging Face Spaces** (free, easiest) — Spaces has native Streamlit
   support and Git LFS for large model files. Push this repo, add the
   `.pth` files via `git lfs track "*.pth"`, set `INVERT_CLASSIFIER_OUTPUT`
   etc. as Space secrets/variables if you want non-default values.
2. **Render** — use the included `Dockerfile` / `render.yaml`. Pick at
   least the "Standard" plan; the free tier's RAM is too small for two
   loaded models plus PyTorch.
3. **Railway** — use the included `Dockerfile` or `Procfile`.
4. **Self-hosted / any VPS** — `docker build -t anemicheck . && docker run
   -p 8501:8501 anemicheck`.

I can't push to any of these for you — I have no deployment credentials —
but every file above is ready to hand to whichever platform you pick. If
you tell me which one, I'll walk through the exact steps (or write a
GitHub Actions workflow to automate it).

## Disclaimer

This tool provides an indicative screening result only and does not
replace professional medical diagnosis. Consult a physician for a reliable
assessment.
