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

## Hosting it publicly

The local server is intended for development and trusted local-network testing,
not direct public deployment. Public hosting needs a production WSGI server,
HTTPS, deployment-specific upload limits and temporary result storage, and
protection against unauthorized or excessive use. The classifier runs on CPU,
so hosting capacity and processing time should also be considered.

Run the original desktop application with `python main.py`.