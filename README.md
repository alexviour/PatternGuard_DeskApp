# PatternGuard

PatternGuard includes the original desktop application and a mobile-friendly
browser interface.

## Run the browser app on your computer

Install the dependencies, then start the local web server:

```powershell
python -m pip install -r requirements.txt
python webapp.py
```

Open `http://localhost:5000` on the computer running the app. To use it from a
phone or tablet on the same Wi-Fi network, open `http://<computer-local-IP>:5000`
on that device. Both devices must be on the same non-guest Wi-Fi network, and
the computer's Windows network profile and firewall must allow private-network
connections on port 5000. If using a trusted home network, set its Windows
network profile to **Private** and allow inbound TCP port 5000 on the **Private**
profile only. Do not expose the development server directly to the public
internet.

The browser app accepts PNG and JPEG images up to 10 MB. Uploaded images are
processed in memory or temporary files; cached results are limited to five and
expire after 30 minutes. It includes textile analysis, classification, metrics,
PDF and PNG exports, the textile reference archive, capture tips, and wallpaper
group guide. Before analysis, you can optionally paint over tears or holes that
automatic screening may miss; marked areas are added to the reconstruction
mask. After reviewing a reconstruction, you can also mark missed or unrepaired
areas in the results page and submit them for another restoration pass. Repairs
are estimates from repeated patterns and cannot guarantee recovery of the
original weave.

## Benchmark textile restoration algorithms

Run the reproducible benchmark from the repository root:

```powershell
python benchmark_restoration.py
```

The script compares DSP template alignment, Telea inpainting, and
Navier–Stokes inpainting on four dataset textiles with disk, scratch, and tear
masks. It prints masked-pixel MAE and PSNR, the system's SSIM, Accuracy,
Recall, FPR, IoU, and SFS metrics, plus median restoration time. Per-case
measurements are written to `restoration_benchmark_results.csv`. All
algorithms receive the same known synthetic damage mask. The benchmark
therefore evaluates restoration and restoration-derived SFS metrics, not the
independent accuracy of the defect detector: the predicted mask used for
Recall, FPR, and IoU is inferred from pixels changed during restoration.
The SFS component values follow `calculate_restoration_metrics`: SSIM is
computed over pixels outside the supplied defect mask, while Accuracy combines
restoration-change coverage inside the mask with unchanged-pixel integrity
outside it. Therefore, SFS and its components should be interpreted alongside
MAE/PSNR against the clean reference; they are not a direct perceptual score
against the original undamaged textile. Timing excludes image loading, damage
generation, and DSP repeat-spacing estimation.

## Evaluate the complete system

Run the system-level evaluation separately from the three-algorithm restoration
benchmark:

```powershell
python system_evaluation.py
```

It writes per-case measurements to `system_evaluation_cases.csv` and a
machine-readable summary to `system_evaluation_summary.json`. The evaluation
reports textile-classification accuracy, macro precision/recall/F1, and
inference time; wallpaper-group agreement and time; defect-mask accuracy,
recall, false-positive rate, and IoU; and DSP restoration SSIM, Accuracy,
Recall, FPR, IoU, SFS, MAE, PSNR, and elapsed time. Disk, scratch, and tear
defects are generated synthetically against one image from each textile class.

The repository classifier was trained on the images in `dataset`, and its
checkpoint does not contain a validation split or epoch history. Thus the
default classification figures are resubstitution diagnostics, not independent
test accuracy, and overfit/underfit cannot be concluded from them. To evaluate
classification on an independent `ImageFolder` directory with the same
class-folder names, run:

```powershell
python system_evaluation.py --test-dataset path\to\held-out\dataset
```

This additionally reports the train-to-test accuracy gap as a diagnostic, not
as a definitive overfit/underfit determination. Wallpaper-group scores compare
the DSP output to the group strings in `cultural_data.py`; those labels have
not been expert-verified and should be treated as provisional agreement rather
than ground-truth accuracy. Detector/restoration scores use synthetic damage,
so they do not establish real-world performance on naturally damaged fabric.
The system-level script does not alter or combine the results in the
three-algorithm benchmark.

## Visualize DSP analysis and restoration

Generate the document-ready, reproducible DSP figures from a clean textile
sample:

```powershell
python generate_dsp_process_figures.py
```

The script saves `dsp_pipeline_stages.png`, `dsp_before_after.png`, and
`dsp_pipeline_run.json` in the `dsp_process_figures\` folder beside the script,
regardless of the terminal's current directory. It prints the absolute folder
and figure paths when complete; the terminal itself does not display the
images. By default it uses
`dataset\pinilian_sinan_bituwon\sinan_bituwon_01.png`; choose another source
with `--image path\to\textile.png` and another destination with
`--output-dir path\to\figures`. The demo adds one clearly labeled synthetic
black disk to the sample so that the before/after figure has a known clean
reference. It then runs the actual DSP functions; it does not claim that the
synthetic disk is a real textile defect.

### DSP process represented in the figures

1. **Read and resize:** the uploaded RGB image is converted to BGR for OpenCV
   and resized to 256 x 256 pixels by `webapp._analyze_image`.
2. **Grayscale conversion:** the resized image is converted to one intensity
   channel. This gray image feeds repeat-spacing estimation; defect isolation
   also makes its own grayscale image from BGR.
3. **FFT autocorrelation:** `run_algorithm_i_autocorrelation` computes the
   two-dimensional FFT, multiplies it by its complex conjugate, inverse
   transforms the power spectrum, and centers the autocorrelation map. It
   suppresses the central peak to estimate horizontal and vertical repeat
   spacing (`a1`, `a2`).
4. **Wallpaper-symmetry vote:** local tiles are compared with mirrored and
   rotated versions using normalized cross-correlation. Votes and tile aspect
   ratio select a heuristic wallpaper-group label; this is a rule-based
   estimate, not a learned or calibrated classifier.
5. **Defect isolation:** `run_algorithm_ii_defect_isolation` applies a dark
   pixel threshold, compares grayscale pixels with a local box-filter mean,
   thresholds structural differences, then uses morphology to form a defect
   mask.
6. **Donor patch search and calibration:** `run_algorithm_iii_template_alignment`
   finds connected components in the mask, searches nearby lattice-aligned
   candidate patches, and scores candidate borders with normalized
   cross-correlation. It rejects donors overlapping the defect mask, then
   adjusts donor color/contrast to the surrounding ring.
7. **Reconstruction:** the calibrated donor patch is copied into the masked
   region. If no suitable donor patch is found, the component is repaired with
   OpenCV Telea inpainting.
8. **Review:** compare the damaged input and reconstructed output; the report
   also provides the FFT, local-deviation response, and binary mask so
   intermediate behavior can be reviewed.

This is a deterministic digital signal/image-processing pipeline, not a
data-trained DSP model. Its measured behavior depends on the input textile,
detector mask, repeat-spacing estimate, and whether a valid donor patch is
available. The detector may mark intact high-contrast weave as well as damage;
the figures and JSON record disclose the example's synthetic damage and
detected-mask size.

## Test classifier overfitting and underfitting

To measure the classifier's fit over training epochs, create a fresh model
using a stratified train/validation split:

```powershell
python evaluate_classifier_fit.py
```

The script does not load or overwrite the application checkpoint. It trains a
fresh ImageNet-initialized MobileNetV3-Small and saves epoch-level
training/validation loss and accuracy, a split manifest, confusion matrices
(CSV and PNG) for both splits, fit curves, and a JSON summary under
`classifier_fit_evaluation\`. Set `--epochs`, `--seed`,
`--validation-fraction`, or `--output-dir` to configure and reproduce the run.
The run prints the validation confusion matrix and per-class/overall
true-positive (TP), false-positive (FP), false-negative (FN), true-negative
(TN), precision, recall, and F1 tables directly in the terminal. The same
per-class tables are saved as `train_classification_summary.csv` and
`validation_classification_summary.csv`.
Open PowerShell in the repository folder and run:

```powershell
python evaluate_classifier_fit.py
```

Epoch progress and final result tables appear in that PowerShell window. To
show the saved validation summary later, run:

```powershell
Import-Csv .\classifier_fit_evaluation\validation_classification_summary.csv | Format-Table -AutoSize
```

The fit label is a heuristic based on train/validation accuracy and validation
loss trends; the small validation sample is not a final independent test set
and cannot establish real-world generalization by itself.

## Deploying a free evaluation copy to Render

The repository includes a Render Blueprint in `render.yaml`. To deploy:

1. Push the project to GitHub, including the classifier model and reference
   images used by the web app.
2. In Render, create a **Blueprint** and connect the GitHub repository and the
   branch you want to deploy.
3. Review the Blueprint and create the service. Render will build the app and
   provide a public `onrender.com` URL when deployment completes.

The Blueprint uses Render's free web-service plan for evaluation. Free
services can spin down after inactivity, so the first request may take about a
minute to wake up. Their filesystem is ephemeral, and CPU and memory are
limited; the app therefore uses one Gunicorn worker. Analysis can be slow or
fail if the free instance runs out of resources. Result links are held in
process memory and expire after 30 minutes, so they are not durable.

Use a paid service for dependable availability or sustained use. Before
sharing publicly, confirm you have permission to publish all included textile
images and model files, and consider abuse/rate protections for public uploads.

Run the original desktop application with `python main.py`.