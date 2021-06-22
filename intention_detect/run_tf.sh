DATASET="IVR"
for ws_mode in 0 1 2 3; do
    python -W ignore tfidf.py -dataset ${DATASET} -ws_mode ${ws_mode} > log.txt
done