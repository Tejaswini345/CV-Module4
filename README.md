# CSc 8830 Module 4 — Web App (Python / Flask)

This version runs your **actual** `rgb_human_segmentation.py` and
`thermal_human_segmentation.py` functions server-side, so the web app
shows exactly the same results as your PDF report — no JavaScript
reimplementation of the algorithm.

## Files

```
app.py                         <- Flask server + API
rgb_human_segmentation.py      <- your Q1 script (imported, not rewritten)
thermal_human_segmentation.py  <- your Q2 script (imported, not rewritten)
compare_with_sam2.py           <- your comparison metrics (imported)
templates/index.html           <- page layout (Q1/Q2/Q3 tabs)
static/app.js                  <- just fetches results and displays them
static/style.css
assets/                        <- put your images here
requirements.txt
```

## Step 1 — Add your images

Same naming convention as before, in `assets/`:

```
assets/
  sample1_rgb.jpg
  sample1_thermal.jpg
  sample1_sam2_rgb_mask.png
  sample1_sam2_thermal_mask.png
  sample2_rgb.jpg
  ...
```

Since this is real Python, the app genuinely scans the folder on every
page load (`discover_samples()` in `app.py` uses `os.path.isfile`) — no
static-site workaround needed. Add `sample5_...` any time and it just
shows up.

## Step 2 — Install dependencies (one-time)

```bash
pip install -r requirements.txt
```

If you already have `opencv-python` and `numpy` installed from doing the
assignment itself, you probably only need `pip install flask`.

## Step 3 — Run it

```bash
python3 app.py
```

Open **http://localhost:5000** in your browser. That's your whole "run
the app" step — same as running any of your other Python scripts, just
this one keeps running and serves a web page instead of printing to the
terminal and exiting.

To stop it, go back to the terminal and press `Ctrl+C`.

## Running from VS Code

Open this folder in VS Code, open `app.py`, and either:
- Click the ▶ "Run" button in the top-right, or
- Open the built-in terminal (`` Ctrl+` ``) and run `python3 app.py`

Then open `http://localhost:5000` in your browser, same as above.

## Putting it online for your submission link

A Flask app needs a real backend host (not a plain static site like
GitHub Pages, since Python has to keep running). Free options that work
well for a student project:
- **Render** — "New → Web Service" (not "Static Site"), connect your
  repo, set the start command to `python app.py`
- **PythonAnywhere** or **Railway** are similar free alternatives

If your assignment mainly needs the screen recording to show it working,
running it locally with `python3 app.py` and recording that is usually
enough — check with your instructor if a permanently-hosted link is
required.
