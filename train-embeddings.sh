python3 ./poincare-embeddings/embed.py \
       -dim 8 \
       -lr 0.3 \
       -epochs 10000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 16 \
       -manifold poincare \
       -dset ./data/rels.csv \
       -checkpoint ./trained_embeddings/poincare-8.pth \
       -batchsize 1024 \
       -eval_each 1 \
       -sym \
       -debug \
       -sparse \
       -gpu 3 \
       -train_threads 1

