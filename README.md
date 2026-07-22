# AgriShield Training on Railway (CPU only)

> **Important:** Railway does **not** offer GPU instances
> ([docs](https://docs.railway.com/guides/ai-agent-workers)).
> This service trains on **CPU**. Use **2–3 epochs** and batch **16** so jobs finish.
> For real GPU training, use Google Colab, RunPod, Modal, or a local NVIDIA PC,
> then upload the ONNX via admin **Deploy**.

Deploy this folder as a Railway service so admin **Start Training** works without Heroku.

## 1. Create the Railway service

1. Sign up at [https://railway.app](https://railway.app)
2. **New Project** → **Deploy from GitHub** (or empty + CLI)
3. Set the **Root Directory** to `railway_training_service`
4. Railway will build with `Dockerfile` or Nixpacks + `Procfile`

### CLI (from this folder)

```bash
cd railway_training_service
npx @railway/cli login
npx @railway/cli init
npx @railway/cli up
npx @railway/cli domain
```

Copy the public URL, e.g. `https://agrishield-training-production.up.railway.app`

## 2. Railway environment variables

In Railway → Variables:

| Variable | Value |
|----------|--------|
| `PHP_API_BASE` | `https://agrishield.bccbsis.com/api/training` |
| `DEFAULT_EPOCHS` | `3` |
| `DEFAULT_BATCH_SIZE` | `16` |
| `TRAINING_TIMEOUT_SEC` | `21600` (6 hours; optional) |

Optional: `TRAINING_API_KEY` if you enable API key checks on PHP endpoints.

Use a plan with **enough RAM** (2GB+ recommended). Free/trial may OOM on PyTorch.

## 3. Point AgriShield (Hostinger / local) at Railway

Edit `training_service_config.php` on the server:

```php
$TRAINING_SERVICE_URL = 'https://YOUR-APP.up.railway.app';
```

Or set the env var `TRAINING_SERVICE_URL` to the same URL.

Detection stays on **Vercel**. Training uses **Railway** only.

## 4. Verify

```bash
curl https://YOUR-APP.up.railway.app/health
```

Expect `"status":"ok"`, `"php_api":"connected"`, and `"cuda_available":false`.

Then in admin → Training → **Start Training** with **3 epochs**.

## Option 2: External dataset ZIP (no Hostinger image folder)

1. Zip `Dataset20260715folder` on your PC **or** share a ZIP file on Google Drive.
2. In Railway → Variables set:

```
DATASET_ZIP_URL=https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
```

Sharing must be **Anyone with the link**. Prefer a **FILE** link (folder links fail past 50 files).

3. On the PHP site, keep `$USE_EXTERNAL_DATASET_ZIP = true` in `training_service_config.php`.
4. Admin → **Start Training** — Railway downloads the ZIP, trains, uploads ONNX back via PHP.

## Flow

```
Admin Start Training
  → PHP creates job + POST Railway /train
  → Railway downloads DATASET_ZIP_URL (or PHP dataset ZIP)
  → Trains (CPU PyTorch — no GPU on Railway)
  → Uploads ONNX via upload_model.php
  → Admin Deploy → Vercel serves new model
```
