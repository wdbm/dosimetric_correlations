# dosimetric_correlations

# note

This project is a work in progress.

# setup

Install the following:

- NumPy
- pandas
- Scikit-learn
- TensorBoard
- TensorFlow

# use

The input data should be CSV with the following fields:

- features:
    - PTV vol (cc)
    - Lungs-GTV vol (cc)
    - Lungs-GTV-PTV vol (cc)
    - Lungs-GTV in PTV vol (cc)
    - KBP Lungs (cc)
    - Lungs-GTV - KBP Lungs (cc)
    - Heart vol (cc)
    - Heart in PTV vol (cc)
- targets:
    - V5 (%)
    - V20 (%)
    - Mean Lungs-GTV (Gy)
    - V30 (%)
    - Mean (Gy)

The rightmost columns should be the targets.

Manually remove missing values from data.

Preprocess the CSV data such that all features are scaled to (-1, 1):

```Bash
./preprocess_CSV_file.py --infile=data.csv --outfile=preprocessed_data.csv
```

Train and evaluate on preprocessed CSV data:

```Bash
./cures_cancer_2.py --help

./cures_cancer_2.py --infile=preprocessed_data.csv
```

The basic accuracy evaluation is a printout of the percentage differences of the model hypothesis output values from input data values.
