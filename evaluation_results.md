# PatternGuard Evaluation Results

## Environment

- OS: Windows-11-10.0.26200-SP0.
- CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (12 logical processors).
- Python / PyTorch / OpenCV: 3.14.8 / 2.14.1+cpu / 5.0.0.

## Held-out textile classifier evaluation

- Dataset: 170 images, 11 classes.
- Split: 118 train / 26 validation / 26 test.
- Best validation epoch: 14 of 15.
- Fine-tuning time: 88.04 seconds on cpu.
- Held-out accuracy: 88.5%.
- Macro precision / recall / F1: 90.9% / 89.4% / 86.1%.
- Weighted F1: 88.2%.
- Coverage at 80% confidence: 88.5%; accuracy among accepted predictions: 100.0%.

| Class | Test images | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| binakol_concha_concha | 1 | 100.0% | 100.0% | 100.0% |
| binakol_kinkinelleng | 2 | 66.7% | 100.0% | 80.0% |
| binakol_kulalabang | 2 | 100.0% | 50.0% | 66.7% |
| binakol_kusikus | 4 | 100.0% | 100.0% | 100.0% |
| binakol_paddak_ti_pusa | 2 | 100.0% | 100.0% | 100.0% |
| pinilian_sinan_agatul | 3 | 100.0% | 100.0% | 100.0% |
| pinilian_sinan_bayabas | 3 | 100.0% | 100.0% | 100.0% |
| pinilian_sinan_bituwon | 1 | 33.3% | 100.0% | 50.0% |
| pinilian_sinan_init | 1 | 100.0% | 100.0% | 100.0% |
| pinilian_sinan_kabalyo | 4 | 100.0% | 100.0% | 100.0% |
| pinilian_sinan_sabong | 3 | 100.0% | 33.3% | 50.0% |

The final test subset is held out from optimizer updates and model selection. The dataset is small, so per-class test estimates have substantial sampling uncertainty.

## Synthetic defect and restoration evaluation

- Synthetic test images: 5.
- Pixel precision / recall / F1 / IoU: 44.8% / 100.0% / 61.9% / 44.8%.
- Mean defect-region MAE before / after: 122.56 / 15.59.
- Mean SSIM to clean reference after repair: 0.9908.
Metrics use synthetic black rectangles as known ground truth. They do not establish performance on naturally occurring textile damage.

## End-to-end time and memory

Warm-path benchmark results (seconds; peak RSS increase in MiB):

| Input | Repeats | Mean | Median | P95 | Mean peak RSS increase |
|---|---:|---:|---:|---:|---:|
| 256x256 | 10 | 0.605 | 0.199 | 2.142 | 2.19 MiB |
| 512x512 | 10 | 0.565 | 0.210 | 1.924 | 2.16 MiB |
| 1024x1024 | 10 | 0.750 | 0.368 | 2.278 | 7.19 MiB |

Warm, in-process call to webapp._analyze_image, including the DSP audit, classifier inference, metrics, and PNG encoding; excludes HTTP upload decoding and PDF export.

## Complexity

- Time: For an input HxW image, decoding/conversion and resizing require O(HW) work. After resize to N pixels (currently 256x256), FFT autocorrelation is O(N log N); fixed-window threshold/morphology operations are approximately O(N). For C connected defect components, donor search is O(C * Q * P), where P is the padded component patch area and Q is bounded by 2,352 candidate offsets in the current nested search. Classifier inference uses a fixed 224x224 input, so its cost is O(F) per image for model operations F. Training for E epochs over D images costs O(E * D * F) with forward/backward work. The benchmark provides empirical inference timings.
- Space: Image decoding and the original RGB image use O(HW) memory. The fixed-size DSP maps use O(N) memory with multiple constant-sized arrays; donor matching uses temporary O(P) patches. Classifier inference uses O(M + A) memory for M model parameters and activations A. Training uses O(M + B*A) memory for model parameters, optimizer state, and activations for batch size B. Peak RSS deltas are measured empirically by the benchmark.
- Limitations: Runtime and RSS are hardware-, operating-system-, and library-dependent. Synthetic defect scores do not replace evaluation on independently expert-labeled damaged textiles.
