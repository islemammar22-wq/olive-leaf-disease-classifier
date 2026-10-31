# Olive Leaf Disease Classifier

Image classifier for olive leaf diseases, built with PyTorch (ResNet18 transfer learning) and served through a FastAPI endpoint.
Built as a hands-on project to learn PyTorch end to end: data loading, training, evaluation, error analysis, deployment.

**Classes:** Healthy, Aculus olearius (olive bud mite), Olive peacock spot

> **Honest summary:** 92% accuracy on the held-out test split, but the training data is studio-style (single leaf, plain light background) from one source. The model fails on at least one real field photo, so it should not be assumed to generalize to orchard photos. See [Limitations](#limitations).

## Results

| Step | Setup | Test accuracy |
|---|---|---|
| Multiclass v1 | Frozen ResNet18, class-weighted loss, 5 epochs | 78% |
| v2 | Same, no class weights | 88% |
| v3 | No class weights, 15 epochs | 89% |
| v4 | Unfroze `layer4` (fine-tuning), 8 epochs | 91% |
| **v5 (final)** | + augmentation (flip, rotation, color jitter), 10 epochs | **92%** |

Final model confusion matrix (rows = actual, columns = predicted; order: Healthy, aculus_olearius, olive_peacock_spot):

```
[[186  18  16]
 [  2 189   9]
 [  5   6 249]]
```

| Class | Precision | Recall |
|---|---|---|
| Healthy | 0.96 | 0.85 |
| aculus_olearius | 0.89 | 0.94 |
| olive_peacock_spot | 0.91 | 0.96 |

Only 7 of 460 diseased leaves were predicted as Healthy.

Note: the same test split was used to compare versions, and there is no separate validation split, so these numbers are slightly optimistic.

## Data

- **Binary version (first model):** 954 images (382 diseased, 572 healthy), black-background leaf scans. 99% validation accuracy after class weighting (diseased recall 0.94 to 0.97).
- **Multiclass version:** public Olive Leaf dataset, 3,400 images (train 2,720 / test 680), collected in an orchard in Denizli, Turkey.
  - Train counts: Healthy 830, Aculus olearius 690, Olive peacock spot 1,200.

Datasets are not included in this repo.

## Findings

1. **Class weighting can backfire.** On the 3-class problem, inverse-frequency weighting dropped accuracy from 88% to 78% and made one class absorb many errors (74 healthy leaves predicted as aculus). The imbalance was mild, so weighting overcorrected. It helped on the binary problem and hurt here.
2. **Training longer did little; fine-tuning helped.** 5 to 15 epochs gave +1 point. Unfreezing the last ResNet block gave +2, with signs of overfitting (train ~99%, validation loss rising), which augmentation reduced.
3. **Background removal is not a safe fix.** With the binary model (black-background training data), pasting the leaf on black after `rembg` flipped a failing studio photo from 94.5% healthy to 85.7% diseased. With the multiclass model (light backgrounds), `rembg` + white background dropped accuracy on a 90-image same-source sample from 88.9% to 70.0%. The final API keeps it off by default.
4. **Confidence is not reliability.** On a field photo of peacock spot, the model predicted aculus at 99.6% (no background removal) and Healthy at 99.9% (with it). A tight crop still gave aculus at 96.3%. Grad-CAM showed attention split between the leaf and the surrounding clutter on the full photo. This is one photo, not a measured failure rate.

## Limitations

- Training and test data come from the same source and imaging style, so test accuracy says little about orchard photos.
- One known field failure (peacock spot predicted as aculus/Healthy). The real field accuracy has not been measured yet.
- Softmax confidence is overconfident on out-of-distribution photos.
- Healthy recall is 0.85: some healthy leaves are flagged as diseased.

## Next steps

- Build a small labeled field test set (photos taken in orchards) and report accuracy on it.
- Add field-condition training data and stronger augmentation (brightness, crops, backgrounds).
- Separate validation and test splits.
- Return "uncertain, retake photo" for low-confidence or out-of-distribution inputs.
- Frontend with disease descriptions, treatment tips and multilingual UI.

## Project structure

```
main.py               # v1: binary healthy/diseased training
train_multiclass.py   # multiclass training + evaluation (final model: olive_multiclass_v3.pth)
app.py                # FastAPI inference service
app_streamlit.py      # Streamlit web UI (EN/FR/AR)
external_test.py      # accuracy on a folder of labeled images, with/without rembg
gradcam.py            # Grad-CAM heatmaps
requirements.txt
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt

# train (expects dataset/train and dataset/test with one folder per class)
python train_multiclass.py

# serve
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`, use **POST /predict**, upload a leaf photo. The response contains the predicted class, confidence and per-class probabilities.

Streamlit web UI (English / French / Arabic, disease description, treatment and prevention notes, and an OpenCV photo-quality check for blur, brightness and leaf coverage; the check only warns, it does not change the prediction). A feedback form can save user-confirmed labels locally (`feedback/`, not committed) for later retraining; the model does not learn live. The app also shows a Grad-CAM heatmap of where the model looked, and the feedback form records whether the heatmap highlighted the right area:

```bash
streamlit run app_streamlit.py
```

Grad-CAM: `python gradcam.py path/to/image.jpg` saves a side-by-side heatmap PNG.

## Stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, FastAPI, Uvicorn, rembg (optional preprocessing).
## Run with Docker

```bash
docker build -t olive-leaf-api .
docker run -p 8000:8000 -v rembg_cache:/root/.rembg olive-leaf-api
```

Then open http://localhost:8000/docs and upload an olive leaf image to the prediction endpoint.

**Notes**
- The first request downloads the background-removal model used by `rembg` (about 1 GB). The `-v rembg_cache:/root/.rembg` option keeps it, so it's downloaded only once.
- Loading that model needs a lot of memory. If the container stops with `Exited (137)`, increase the memory available to Docker (around 10 GB worked here).
