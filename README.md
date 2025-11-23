# AIPI 520 Project 2 – ClinicalTrials Processing

This repo contains the scripts/instructions I use to fetch and preprocess
ClinicalTrials.gov XML data for Project 2.

## 1. Download the raw XML dump
ClinicalTrials.gov publishes an official archive (~11 GB). Keep it **outside**
this repo to stay under GitHub’s size limits.

```bash
mkdir -p raw_data
cd raw_data
wget https://clinicaltrials.gov/AllPublicXML.zip
unzip AllPublicXML.zip
cd ..
```

You should now have folders like `raw_data/NCT0000xxxx/*.xml`.

## 2. Build the XML path index
Replicating the instructor notebook, collect every XML path once:

```bash
mkdir -p data
find raw_data -path "*/NCT*/*.xml" | sort > data/all_xml
```

## 3. Parse with `process_trials.py`
`process_trials.py` walks `data/all_xml`, extracts modeling-friendly fields
(IDs, sponsors, phase, interventions, eligibility, etc.), and emits NDJSON
(one JSON object per line).

```bash
python process_trials.py \
  --index-file data/all_xml \
  --output data/trials_summary.ndjson \
  --max-records 1000   # optional cap for quick smoke tests
```

Share `data/trials_summary.ndjson` with teammates through Drive/Box/etc.,
or have them rerun the same script locally; the raw data stays outside Git.

## 4. Label + model
Load the NDJSON in pandas/Spark, define success labels (e.g., based on
`overall_status` and outcomes), engineer features, and train the classifier
required by Project 2.

## Why no data in Git?
Directories like `ctg-public-xml/` and `data/*.ndjson` quickly exceed 25 MB and
aren’t necessary to reproduce the pipeline. Everyone can rebuild the dataset
following the steps above, keeping the repo light and compliant.