# Case File: Verified Video Investigation for Smart Cities

Search city camera footage in plain language, and get back only the moments a vision language model (VLM) can **verify**, packaged as an evidence report.

An operator types something like *"a yellow taxi on a busy avenue"*. Case File finds candidate video segments, asks the VLM to re-check each one against the exact question, drops the weak ones, and exports a timeline report with clips and thumbnails. If nothing passes verification, it says so instead of inventing a match.

Built as a hackathon prototype for the Hafnia Hackathon 2026 (Milestone Systems and NVIDIA). It runs as a single Google Colab notebook.

## How it works

1. **Index:** each video is cut into short segments (5 seconds by default). The VLM describes each segment in two sentences (vehicles, pedestrians, cyclists, signals, notable events).
2. **Search:** your query is embedded and matched against the segment descriptions to shortlist candidates.
3. **Verify:** the VLM looks at each candidate's frames and answers Yes or No to your exact question. The Yes/No probability is the confidence score. Candidates below the threshold are dropped.
4. **Export:** verified segments become a case file: an HTML timeline report, clipped videos, thumbnails and a JSON record, zipped for download.
5. **Benchmark:** a built-in test set scores hit rate, precision, no-match handling and speed, and generates a benchmark report.

## Requirements

- A Google account (for Google Colab; the free tier is enough).
- A **GPU runtime** in Colab. The notebook was run on a free **Tesla T4**.
- 2 to 5 short traffic or city-camera video clips (`.mp4`, `.mov`, `.avi` or `.mkv`).
- Internet access in Colab. The first run downloads the models.

No API keys are needed. The notebook uses these open models:

| Role | Model |
|---|---|
| Vision language model | `Qwen/Qwen2-VL-2B-Instruct` |
| Text embedder | `sentence-transformers/all-MiniLM-L6-v2` |

Check each model's page for its license before using the project beyond the hackathon.

## Setup

1. **Open the notebook in Colab.** Go to [colab.research.google.com](https://colab.research.google.com), choose *File > Upload notebook*, and select `Case_File_Verified_Video_Investigation_for_Smart_Cities.ipynb`.
2. **Turn on the GPU.** Choose *Runtime > Change runtime type*, set the hardware accelerator to **T4 GPU**, and save.
3. **Prepare your videos.** Rename each clip to the camera name you want, such as `cam1.mp4`, `cam2.mp4`, `cam3.mp4`. The **file name becomes the camera name** in results and reports. The benchmark section expects `cam1` to `cam5`, so use those names if you plan to run it unchanged (see [Benchmark](#benchmark)).
4. **Run the cells in order, top to bottom:**
   - **1. Install:** installs the Python packages and `ffmpeg`.
   - **2. Config and imports:** sets the settings and folders, and prints the detected GPU. Confirm it says a GPU, not "none".
   - **3. Load the VLM and embedder:** downloads the models (a few minutes on the first run).
   - **4. Upload videos:** opens a file picker. Select all your clips at once. Run this cell **once**: running it again adds more files to the same folder.
   - **5. Index:** describes every segment. Expect roughly 100 seconds for about 30 five-second segments on a T4.
   - **6. Search and verify:** defines `search()` and runs the built-in sample queries.
   - **7. Export the case file:** builds and downloads `case_001.zip`.
   - **8. Benchmark:** scores the test queries and writes the benchmark report.

If Colab restarts the runtime, run all cells again from the top. Uploaded videos and indexes live in the session and are lost on restart.

## Using it

### Run your own query

After indexing, search in a new cell:

```python
verified, all_results = search("A pedestrian crosses the street")
```

Optional filters narrow the search:

```python
search("A cyclist rides on the road", camera="cam4", t_min=0, t_max=20)
```

Each verified result shows the camera, time range, confidence (`P(yes)`) and the VLM's description.

### Export a case file

```python
zip_path = export_case("A pedestrian crosses the street", verified, case_name="case_002")
```

The zip contains:

| File | What it is |
|---|---|
| `report.html` | Timeline report with thumbnails, descriptions, confidence and links to clips |
| `clips/` | The verified segments as short `.mp4` files |
| `thumbs/` | One thumbnail per segment |
| `case.json` | Full machine-readable record of the query and results |

Unzip it and open `report.html` in a browser. If the download does not start automatically, open the Files panel in Colab and look in `casefile/out/`.

## Settings

Defined in the *Config and imports* cell:

| Setting | Default | Meaning |
|---|---|---|
| `SEGMENT_SECONDS` | 5 | Length of each indexed segment |
| `FRAMES_PER_SEGMENT` | 3 | Frames the VLM sees per segment |
| `MAX_SIDE` | 448 | Frames are resized so the longest side is this many pixels (keeps the T4 fast) |
| `TOP_K` | 8 | Candidates verified per query |
| `VERIFY_THRESHOLD` | 0.60 | Minimum `P(yes)` to count as a verified match |

Shorter segments and more frames find finer events but take longer to index. A lower threshold returns more results; a higher one is stricter.

## Benchmark

Section 8 runs 20 test queries and generates `benchmark_report.html` and `benchmark_results.csv` in `casefile/out/`. The report shows hit rate, precision, how many impossible queries correctly returned no match, average time per query, a per-camera breakdown, and a list of misses.

Before you trust the numbers:

- **Camera names must match.** Test cases refer to `cam1` to `cam5`. If your files have other names, either rename them or edit the camera names in `TESTS`.
- **Tighten the time windows.** The defaults are `(0, 9999)`, meaning "anywhere in that clip", which makes scores optimistic. Watch each clip and set the real start and end seconds. The report warns you while windows are loose.
- **Use negative cases.** A camera of `None` means no match is expected. A good system returns nothing for these.

## Getting sample videos

Free stock clips from [Mixkit](https://mixkit.co/free-stock-video/) work for testing. Mixkit states its clips are free to download without a watermark under the [Mixkit License](https://mixkit.co/license/#videoFree); read it before using clips in a public demo. For a stronger evaluation with real events, use traffic-camera or dashcam datasets such as [NVIDIA PhysicalAI SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces), or Hafnia's own data if your registration includes access. Check each dataset's license.

## Using a different VLM (for example Hafnia's)

The model is called from one function, `vlm(images, prompt, max_new_tokens, yes_no)`, in section 3. To switch models, replace that function. It must return the generated text, or a `(text, p_yes)` pair when `yes_no=True`. The rest of the notebook does not change.

## Known limitations

- **Verifier confidence is not calibrated.** The small model can answer "Yes" with very high confidence for clips that only loosely match, so a high `P(yes)` is not proof. Always review the clip and the description.
- **Retrieval depends on descriptions.** If the VLM describes a segment poorly, text search can miss it. Verification only filters the shortlist; it cannot recover segments that were never shortlisted.
- **Small model, short clips.** Qwen2-VL-2B is chosen to run on a free T4. Larger models will describe and verify better.
- **Stock footage has few real events.** Violations, near-misses and collisions rarely appear in stock clips, so event queries are best tested with dashcam or traffic-camera data.
- **Prototype only.** Descriptions are model output. They must be reviewed by a human before being used as evidence.

## Troubleshooting

| Problem | Fix |
|---|---|
| "GPU: none" printed | *Runtime > Change runtime type > T4 GPU*, then re-run from the top |
| Duplicate cameras such as `cam1 (1)` | The upload cell was run twice. Delete the copies, then re-run the video list and index cells: `!rm /content/casefile/videos/*"(1)"*` |
| Benchmark hit rate is 0% | Camera names in `TESTS` do not match your file names |
| `IndexError` when exporting | A video segment could not be read. Re-encode the clip: `!ffmpeg -i input.mov -c:v libx264 cam1.mp4` |
| Export fails with an `ffmpeg` error | Re-run the install cell and check `!ffmpeg -version` |
| `NameError: verified` or `search` | Run the earlier cells (sections 5 and 6) in this session first |
| No verified match for a query you expected to match | Rephrase it, lower `VERIFY_THRESHOLD` (for example 0.4), or check the clip actually shows it |
| Download did not start | Open the Files panel and download from `casefile/out/` |
| Out-of-memory error | Lower `MAX_SIDE` or `FRAMES_PER_SEGMENT`, or restart the runtime |

## Project structure

```
Case_File_Verified_Video_Investigation_for_Smart_Cities.ipynb   # the whole project
README.md
```

Files created at runtime in Colab:

```
/content/casefile/
├── videos/                 # your uploaded clips
└── out/
    ├── case_001.zip        # exported case file
    ├── case_001/           # report.html, clips/, thumbs/, case.json
    ├── benchmark_report.html
    └── benchmark_results.csv
```

## Credits

Built for the Hafnia Hackathon 2026 by Milestone Systems and NVIDIA. Uses Qwen2-VL (Alibaba Cloud), Sentence-Transformers, OpenCV and FFmpeg. Sample footage from Mixkit.
