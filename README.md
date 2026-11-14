# PatternGuard

PatternGuard includes the original desktop application and a mobile-friendly
browser interface.

## Run the browser app on your computer

Create the project environment, install dependencies, then start the local
web server:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe webapp.py
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

### Supported-textile input screening

Before either application runs the DSP audit, a separate binary
MobileNetV3-Small gate checks whether an image appears to contain textile,
fabric, or a repeating woven pattern. Its acceptance threshold is 75%, so
inputs below that probability are rejected before DSP processing; a passing
score is a screening result, not proof that an image is textile.
Passing this gate allows the DSP audit to run even when the 11-class
classifier cannot confidently identify a Pinilian or Binakol type; in that
case no specific class or cultural label is assigned.

The training set includes 960 supplemental images from eight texture
categories in the Describable Textures Dataset (DTD): woven, knitted,
braided, meshed, interlaced, fibrous, lacelike, and gauzy. These labels
describe visual texture properties and do not guarantee that every image is
fabric. DTD is provided for research use; retain its source information and
verify its terms before redistributing its images or derived materials. See
the [official DTD page](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and the
paper by Cimpoi et al., *Describing Textures in the Wild* (CVPR 2014).

At the requested 75% cutoff, this DTD-enhanced checkpoint accepted 222 of 226
textile validation images (98.23%) and rejected 65 of 66 non-textile
validation images (98.48%). On the held-out test split it accepted 223 of
226 textile images (98.67%) and rejected all 66 non-textile images (100%).
The validation split therefore contained one false acceptance; the held-out
test contained three false textile rejections. On these same splits, the
previous gate accepted 72.12% and 75.22% of textile images, respectively,
while rejecting all non-textile samples. These are small, curated,
image-level splits: the non-textile samples do not include the reported
portrait or random screenshot, so these results do not establish that such
inputs will always be rejected. The trainer's recall-oriented threshold
selector chose 15.67%; the active checkpoint deliberately retains the
requested 75% cutoff. The run report records the fixed-cutoff metrics
separately from metrics at the automatically selected threshold.

To train with additional textile/fabric examples arranged as class folders,
pass an ImageFolder root with `--additional-textile-datasets` when running
`evaluate_textile_input_gate.py`. Only add images whose source terms permit
this research use, keep attribution/license records, and evaluate on a
separate test collection containing zoomed-out fabrics and unrelated
screenshots.

The gate weights are stored in `models\textile_input_gate.pth`; the existing
11-class weights remain in `models\pattern_classifier.pth`. The gate can be
retrained and evaluated with `evaluate_textile_input_gate.py`. Per-image
credits for its Open Images non-textile training samples are in
`models\textile_input_gate_attribution.csv`; retain these credits when
redistributing the model. The source archives identify the images as CC BY
2.0 and Open Images annotations as CC BY 4.0.

## Benchmark textile restoration algorithms

Run the reproducible benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe benchmark_restoration.py
```

The benchmark evaluates every PNG image in every textile-class folder and
applies six fixed synthetic masks per source: small/large disks, thin/wide
scratches, and small/large tears. All algorithms receive identical damaged
images and masks, with the clean source image as ground truth. The included
dataset currently supplies 170 images across 11 textile classes, producing
1,020 paired cases. PatchMatch is a seeded, single-scale exemplar
implementation: it
propagates nearest-neighbor patch offsets and uses shrinking random search to
find intact donor patches; it is inspired by recent non-local patch-based
inpainting work, not a reproduction of that paper's full multi-scale method.
The DSP's internal Telea fallback remains unchanged, but Telea is no longer a
separately benchmarked baseline. The script prints masked-pixel MAE and PSNR,
SSIM, restoration Accuracy, and Recall/FPR/IoU/SFS as unavailable, plus median
restoration time. Per-case measurements are written to
`restoration_benchmark_results.csv`; overall and damage-type summaries with
source-image paired bootstrap intervals for MAE differences are written to
`restoration_benchmark_summary.json`. Timing defaults to one warm-up and five
timed calls per method and case, and can be adjusted with `--warmup-runs` and
`--timed-runs`.

Use masked RGB MAE and PSNR against the clean image as the primary
reconstruction measures. SSIM and restoration Accuracy use the clean reference
and known mask; Accuracy combines the fraction of damaged pixels reconstructed
within 12 intensity levels of the clean image with unchanged-pixel integrity
outside the mask. Recall, FPR, IoU, and SFS are N/A in this restoration-only
benchmark because it supplies the known mask directly to each restoration
method and does not evaluate a detector. Use the complete-system evaluation
below for detector Recall, FPR, and IoU against known synthetic damage masks.
The system evaluation also reports the SFS using its documented weights; this
composite is implementation-defined, not an expert-validated perceptual score.
Bootstrap intervals resample source images, rather than treating each mask
applied to an image as an independent sample; they are descriptive for this
dataset and do not replace evaluation on external data. Timing excludes image
loading, damage generation, and DSP repeat-spacing estimation.

## Evaluate the complete system

Run the system-level evaluation separately for system-only metrics:

```powershell
.\.venv\Scripts\python.exe system_evaluation.py
```

It writes per-case measurements to `system_evaluation_cases.csv` and a
machine-readable summary to `system_evaluation_summary.json`. The evaluation
reports textile-classification accuracy, macro precision/recall/F1, and
inference time; wallpaper-group agreement and time; defect-mask accuracy,
recall, false-positive rate, and IoU; and DSP restoration SSIM, Accuracy,
Recall, FPR, IoU, SFS, MAE, PSNR, and elapsed time. Disk, scratch, and tear
defects are generated synthetically against every source image by default,
using the same six damage variants as the restoration benchmark (currently
170 images × 6 variants = 1,020 cases). Classification remains one result per
source image (170 images). For a quick subset run, add `--max-images 11`.

In the latest full 1,020-case run, detector pixel accuracy is **98.79%**,
balanced accuracy is **99.38%**, and recall is **100.00%**. The ground-truth
mean per-case IoU is **83.74%** (the separately pooled pixel IoU is **60.86%**),
with a **1.24%** false-positive rate. The system's mean restoration accuracy is
**73.89%**, clean-region SSIM is **98.32%**, and project-defined SFS is
**88.63%**, below the **95.00% project target**. Pixel accuracy is shown with
balanced accuracy and IoU because the many clean pixels can make pixel
accuracy look high even when mask overlap is weaker.

For each case, SFS is
`0.35 × Restoration accuracy + 0.25 × SSIM + 0.15 × Recall + 0.15 × Specificity + 0.10 × IoU`,
where specificity is `100 − FPR`. Restoration accuracy is `0.60 × repair
accuracy + 0.40 × clean-region integrity`; repair accuracy is the fraction of
ground-truth damaged grayscale pixels reconstructed within **12** intensity
levels (inclusive) of the clean source, and clean-region integrity is the
fraction of ground-truth clean pixels preserved within the same tolerance.
SSIM uses the standard 0–1 similarity scale, 8-bit data range, and clean
ground-truth pixels. Recall, specificity, and IoU compare the predicted damage
mask with the synthetic ground-truth mask. Case scores are averaged equally;
the pooled IoU instead aggregates pixels across all cases, so both values are
reported and are not interchangeable.

The weights prioritize reconstruction (35% threshold-based accuracy and 25%
structural similarity), balance missed defects against false alarms (15% each
for recall and specificity), and retain a 10% spatial-overlap term. They are
project-defined, not empirically calibrated; several components overlap, so
the SFS is a transparent system composite rather than an independent
perceptual or symmetry-fidelity measure. The 95.00% figure is a project target,
not an independently validated performance threshold. The detector's
dark-pixel threshold is calibrated to the generated synthetic mark intensities;
these results do not establish sensitivity to naturally damaged or
differently colored textiles.

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

The interactive web app does not have a clean reference image or complete
damage annotations for an arbitrary upload, so its MAE, PSNR, restoration
Accuracy, and detector Recall/FPR/IoU are shown as N/A rather than computed
against the damaged upload. It separately shows a provisional reference-free
Estimated SFS: an equal-weight mean of outside-mask preservation,
repair-boundary continuity, and SSIM. These are uncalibrated plausibility
heuristics, not ground-truth performance scores, and should not be compared
with system-evaluation SFS.
The system-level synthetic evaluation continues to calculate the original
ground-truth metrics by comparing reconstructions with clean source images and
detected masks with known synthetic ground-truth masks.

### Run the panel evaluation and print metric tables

For a live terminal demonstration, run the unit tests and the full system
evaluation without starting the slower three-algorithm benchmark or generating
a Word report:

```powershell
.\.venv\Scripts\python.exe run_panel_evaluation.py
```

This evaluates all 170 dataset images with six synthetic damage variants
(1,020 cases), then prints the evaluation method, SFS formula, weighted SFS
components, restoration metrics, detector metrics, and separate classifier
diagnostics as terminal tables. The 34-image classifier validation experiment
is not the sample size for the system damage evaluation. Per-case results and
the summary JSON are saved under `evaluation_report\panel_evaluation\`.

### Run all tests and create the Word evaluation report

Run the unit tests, full system-level synthetic evaluation, three-algorithm
benchmark, and DSP demonstration with one command, then create a Word document
covering Objectives 1–3 with results, source-code listings, tables, figures,
complexity analysis, and limitations:

```powershell
.\.venv\Scripts\python.exe run_project_evaluation.py
```

The Word report is saved as
`evaluation_report\PatternGuard_Objectives_1_to_3_Results.docx`; system and
algorithm per-case CSVs, JSON summaries, and PNG figures are saved in
`evaluation_report\`.
Objective 1 includes source-code listings for the webapp's resize-to-restoration
call sequence and the DSP spacing, symmetry, defect-mask, and reconstruction
functions. Objective 2 reports the system's SFS and textile classification,
including classifier-fit learning curves and confusion matrices. The runner
uses the existing classifier-fit history; run
`python evaluate_classifier_fit.py` first to refresh those fit-evaluation
artifacts. Objective 3 compares DSP, PatchMatch, and Navier-Stokes with masked
reconstruction quality, measured median restoration milliseconds, and
time/space complexity. The benchmark runs all 1,020 paired cases by default,
but the report runner uses one timed restoration call per method/case (3,060
restorations total) to finish faster. It prints progress every 100 cases.
Increase timing repeats for a more stable timing estimate with:

```powershell
.\.venv\Scripts\python.exe run_project_evaluation.py --benchmark-timed-runs 5
```

Quality metrics still cover all 1,020 cases regardless of timing repeats.
Use VS Code's **Run Python File in Terminal** action (not an Output-only
Code Runner action) to see the run directly in the integrated terminal. Test
and evaluation progress is streamed as it runs. After the benchmark, figures,
and Word report finish, the console prints aligned tables for the test count,
Objective 2 system metrics, Objective 3 algorithm results, and paired MAE
intervals, followed by output paths. Running an individual test file reports
test pass/fail only; run the full command above to generate the evaluation
tables.

Install requirements first if needed; this script requires `python-docx` to
write the Word document. If the existing report is open in Word and cannot be
overwritten, the script saves the new document with an `_updated` suffix.

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
   detects multiple off-center peaks to estimate two lattice vectors and
   reports low-confidence or insufficient lattice evidence as uncertain.
4. **Wallpaper-symmetry vote:** local tiles are compared with mirrored and
   rotated versions using normalized cross-correlation. Votes and tile aspect
   ratio select a heuristic wallpaper-group label. Inconsistent votes and
   groups that require unmeasured lattice-centering evidence are reported as
   uncertain; this is a rule-based estimate, not a learned or calibrated
   classifier.
5. **Defect isolation:** `run_algorithm_ii_defect_isolation` applies a dark
   pixel threshold, compares grayscale pixels with a local box-filter mean,
   thresholds structural differences, then uses morphology to form a defect
   mask.
6. **Donor patch search and calibration:** `run_algorithm_iii_template_alignment`
   finds connected components in the mask, searches nearby clean candidate
   patches, and scores candidate borders with normalized cross-correlation.
   When the wallpaper-group operation votes and lattice vectors are reliable,
   donor origins are registered to the repair window's lattice phase and
   shortlisted at integer shifts of the measured lattice basis. It tests only
   supported symmetry operations on those clean donors. Candidate transforms
   are centered at lattice-derived positions, warped with padding to avoid
   clipping, and scored against the undamaged boundary. A transformed donor
   is used only when its boundary match improves meaningfully; donor overlap
   with the defect mask is rejected, and the selected donor is calibrated to
   the surrounding ring.
7. **Reconstruction:** the calibrated donor patch is copied into the masked
   region. If symmetry evidence is unavailable or inconclusive, or no
   transformed candidate improves the boundary match, the existing
   untransformed donor is retained. If no suitable donor patch is found, the
   component is repaired with OpenCV Telea inpainting.
   The web analysis result also keeps internal per-output symmetry status and
   per-region selection evidence; these fields are not shown in the audit UI.
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
.\.venv\Scripts\python.exe evaluate_classifier_fit.py
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
The Objectives 1–3 Word report embeds the existing fit curves and validation
confusion matrix below Objective 2, separately from the deployed checkpoint's
in-sample classification results. Re-run this fit command before the report
runner if you want to refresh those fit-evaluation artifacts.
Open PowerShell in the repository folder and run:

```powershell
.\.venv\Scripts\python.exe evaluate_classifier_fit.py
```

Epoch progress and final result tables appear in that PowerShell window. To
show the saved validation summary later, run:

```powershell
Import-Csv .\classifier_fit_evaluation\validation_classification_summary.csv | Format-Table -AutoSize
```

The fit label is a heuristic based on train/validation accuracy and validation
loss trends; the small validation sample is not a final independent test set
and cannot establish real-world generalization by itself.

## Deploying for respondent testing with Google Cloud Run

Cloud Run is a better fit than a free sleep-on-idle web host for a scheduled
respondent session. Its free usage allowance is subject to Google's current
pricing, region, account, and billing requirements. Billing must be enabled,
and usage beyond the free allowance can be charged. Review
[Cloud Run pricing](https://cloud.google.com/run/pricing) and configure a
budget alert before inviting respondents.

The included `Dockerfile` installs CPU-only PyTorch and packages the app for
Cloud Run. From a terminal with the Google Cloud CLI installed and authenticated:

```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
gcloud run deploy patternguard `
  --source . `
  --region asia-southeast1 `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi `
  --cpu 1 `
  --concurrency 4 `
  --max 1 `
  --min 1 `
  --timeout 600
```

Keep the single-instance limit while testing because audit results are stored
in process memory. The app retains up to 25 results for two hours, enough for a
20-respondent session, but results are lost if the instance restarts. Analysis
requests are serialized by the app, so allow time for respondents' uploads to
finish rather than having everyone submit at once. Keeping one minimum
instance warm reduces cold starts during the session; it can incur charges
while idle. After data collection, scale to zero to avoid idle-instance charges:

```powershell
gcloud run services update patternguard --region asia-southeast1 --min 0
```

The app is publicly accessible when deployed with `--allow-unauthenticated`.
Share the URL only with intended respondents and review the privacy and
consent requirements for any uploaded images.

Render remains configured in `render.yaml` as an alternative. Its free web
services can sleep when idle and have limited CPU and memory, so the first
request may be slow and analysis may fail under resource pressure.

Before sharing publicly, confirm you have permission to publish all included
textile images and model files, and consider abuse/rate protections for public
uploads.

Run the original desktop application with `python main.py`.