"""Run the publisher's exact Euclidean neighbour index without loading the neural runtime."""
import argparse
from pathlib import Path

import faiss
import numpy as np


def main(folder, sizes):
    faiss.omp_set_num_threads(1)
    x = np.ascontiguousarray(np.load(folder / "index_train.npy", allow_pickle=False), dtype=np.float32)
    q = np.ascontiguousarray(np.load(folder / "index_query.npy", allow_pickle=False), dtype=np.float32)
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    for k in sizes:
        results = [index.search(q[i:i+8], k)[1] for i in range(0, len(q), 8)]
        np.save(folder / f"index_{k}.npy", np.concatenate(results), allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("size", nargs="+", type=int)
    args = parser.parse_args()
    main(args.folder, args.size)
