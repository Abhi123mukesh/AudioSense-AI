# AudioSense AI — Cinematic Music Discovery

A Streamlit machine-learning demo for recommending tracks with similar numeric sound profiles.

## Run locally

1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal in that folder.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Visual features

- Large AudioSense wordmark and animated equalizer logo
- Dark cinematic gradients, glass-style panels and neon accents
- Subtle entrance, shimmer, hover and equalizer animations
- Responsive layout and reduced-motion support
- Discover, Music Library and Behind the Sound sections (no Presentation Mode)

## Prototype note

The included song names and numeric feature values are synthetic sample data. The app demonstrates content-based recommendation using StandardScaler, weighted features and cosine similarity; it does not analyse real audio recordings or measure real listener preferences.
