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

## 5. After training finishes

Models tab → **Deploy** so ONNX syncs for Vercel detection.

## Flow

```
Admin Start Training
  → PHP creates job + POST Railway /train
  → Railway downloads dataset ZIP from PHP
  → Trains (CPU PyTorch)
  → Uploads ONNX via upload_model.php
  → Admin Deploy → Vercel serves new model
```
