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
mask. Repairs are estimates from repeated patterns and cannot guarantee recovery
of the original weave.

## Benchmark textile restoration algorithms

Run the reproducible benchmark from the repository root:

```powershell
python benchmark_restoration.py
```

The script compares DSP template alignment, Telea inpainting, and
Navier–Stokes inpainting on four dataset textiles with disk, scratch, and tear
masks. It prints masked-pixel MAE and PSNR plus median restoration time, and
writes per-case measurements to `restoration_benchmark_results.csv`. All
algorithms receive the same known synthetic damage mask, so this benchmark
measures restoration quality rather than damage-detection accuracy. Timing
excludes image loading, damage generation, and DSP repeat-spacing estimation.

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