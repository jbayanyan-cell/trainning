# AgriShield Training on Railway

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
| `DEFAULT_EPOCHS` | `10` |
| `DEFAULT_BATCH_SIZE` | `8` |

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

Expect `"status":"ok"` and `"php_api":"connected"`.

Then in admin → Training → **Start Training**.

## Option 2: External dataset ZIP (no Hostinger image folder)

1. Zip `Dataset20260715folder` on your PC **or** share the folder on Google Drive.
2. In Railway → Variables set either:

```
# Folder (what you have now):
DATASET_ZIP_URL=https://drive.google.com/drive/folders/1H_TDkyyCZus54yH92vgFVwsjsSPK5Z1Y?usp=sharing

# OR a single ZIP file share link:
# DATASET_ZIP_URL=https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
```

(Drive folder links are downloaded with `gdown`. Sharing must be **Anyone with the link**.)

4. Redeploy Railway after pushing `dataset_source.py`.
5. On the PHP site, keep `$USE_EXTERNAL_DATASET_ZIP = true` in `training_service_config.php`.
6. Admin → **Start Training** — Railway downloads the ZIP, trains, uploads ONNX back via PHP.

Later improvements: replace the ZIP with a newer one (captures added), update the Drive file / URL, train again.


## Flow

```
Admin Start Training
  → PHP creates job + POST Railway /train
  → Railway downloads DATASET_ZIP_URL (or PHP dataset ZIP)
  → Trains (CPU PyTorch)
  → Uploads ONNX via upload_model.php
  → Admin Deploy → Vercel serves new model
```
