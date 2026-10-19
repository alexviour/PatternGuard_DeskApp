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
group guide.

## Evaluate classification and pipeline performance

Run the evaluation from the project root:

```powershell
python evaluate_pipeline.py
```

The evaluator makes a reproducible, stratified train/validation/test split by
textile class (approximately 70%/15%/15%, with at least one validation and
test image per class), fine-tunes the same pretrained MobileNetV3-Small
classifier, selects the epoch using validation accuracy/loss, and reports
held-out accuracy, macro/weighted precision, recall, F1, per-class metrics,
confusion matrix, and coverage/accuracy at the app's 80% confidence cutoff.
The test split is not used for training or model selection. Training uses
pretrained Torchvision weights; the first run may need network access to
download those weights.

It also creates controlled black-rectangle defects on held-out clean textile
images and reports pixel-level detection precision/recall/F1/IoU and
restoration comparisons against the uncorrupted source. These synthetic scores
are only a repeatable sanity check and must not be presented as accuracy on
real-world damage. The repository does not currently contain expert-labeled
damage masks.

Finally, it benchmarks the warmed `webapp._analyze_image` path at 256, 512,
and 1024 square input resolutions, reporting mean/median/P95 latency and
sampled peak process-RSS increase. This includes DSP, classifier inference,
metrics, and PNG encoding, but excludes HTTP upload decoding and PDF export.
The same report explains the pipeline's asymptotic time/space complexity.

The default run can take several minutes because it trains for 15 epochs and
benchmarks multiple inputs over 10 repeats. To adjust the run, for example:

```powershell
python evaluate_pipeline.py --epochs 5 --benchmark-images 3 --benchmark-repeats 1
```

The evaluator writes `evaluation_results.json` and a readable
`evaluation_results.md` to the project root. Results are machine- and
dataset-dependent; report the test-set size and hardware alongside the
measured scores and timings.

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