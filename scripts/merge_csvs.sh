rm analysis_dataset.csv
awk '(NR == 1) || (FNR > 1)' *.csv > analysis_dataset.csv