# Dataset Information

## About the Data

This project uses **Google Maps hospital reviews from Bengaluru, India** — a publicly available, sentiment-labeled dataset scraped from Google Maps.

## Where to Download

| Source | Link |
|---|---|
| Kaggle (primary) | [Hospital Reviews Bengaluru](https://www.kaggle.com/datasets/) ← search "Bengaluru hospital reviews Google Maps" |
| Alternate | Check the project Issues tab for community-shared mirrors |

## Expected Schema

Place your file as: `data/raw/reviews.csv`

| Column | Type | Description |
|---|---|---|
| `review_text` | str | Raw review text |
| `sentiment` | str/int | Label: `positive`/`negative` or `1`/`0` |
| `hospital_name` | str | Name of the hospital |
| `rating` | float | Star rating 1–5 (if available) |
| `review_date` | str | Date of review (optional) |

> If your column names differ, update `src/config.py` accordingly.

## Privacy Note

No personally identifiable information (PII) should be present. If reviewer names appear in your copy of the dataset, remove them before running the pipeline.

## License

Check the Kaggle dataset page for terms of use. Raw data is **not committed** to this repository.
