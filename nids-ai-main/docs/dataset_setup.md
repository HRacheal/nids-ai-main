# Obtaining the UNSW-NB15 dataset

The dataset is **not included** in this repository or the submission package:
its licence permits free academic use but does not address redistribution.
Download it yourself as described below (about 600 MB of CSV files).

## Option 1: official source (preferred)

1. Open <https://research.unsw.edu.au/projects/unsw-nb15-dataset> and follow
   the "HERE" link to the UNSW SharePoint folder.
2. From the CSV folder download `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv`,
   `NUSW-NB15_features.csv` and `UNSW-NB15_LIST_EVENTS.csv`; from its
   training/testing sub-folder download `UNSW_NB15_training-set.csv` and
   `UNSW_NB15_testing-set.csv`. Optionally `NUSW-NB15_GT.csv`.
3. Do not download the pcap, BRO or Argus folders (about 100 GB).

When this project was built (Nov 2026) the SharePoint folder refused
automated access (HTTP 403), so Option 2 was used.

## Option 2: Kaggle mirror (what this project used)

Third-party mirror, verified against the official documentation (see
`docs/dataset_profile.md`, section 2):
<https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15> (version 1).
Download the archive from the page (or `kaggle datasets download
mrwellsdavid/unsw-nb15`) and extract the CSVs. The optional ground-truth file
comes from <https://www.kaggle.com/datasets/alextamboli/unsw-nb15>.

## Where to put the files

```
data/raw/unsw_nb15/
    UNSW-NB15_1.csv  UNSW-NB15_2.csv  UNSW-NB15_3.csv  UNSW-NB15_4.csv
    NUSW-NB15_features.csv  UNSW-NB15_LIST_EVENTS.csv
    UNSW_NB15_training-set.csv  UNSW_NB15_testing-set.csv
    _mirror2_alextamboli/NUSW-NB15_GT.csv          (optional)
```

Keep the original file names and never open/save the files in Excel.

**Note on the training/testing names.** In the available copies the two
file names are swapped relative to the official documentation. The code
identifies the files by SHA-256 and uses the 175,341-record file for
training, as the official documentation specifies; you do not need to
rename anything.

## Verify

```powershell
.venv\Scripts\python.exe scripts\validate_dataset.py
```

Every line must say `OK`. A checksum mismatch means the file differs from
the copy used for the published results.
