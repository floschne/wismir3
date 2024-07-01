import argparse
import os
import shlex
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download_part(start: int, stop: int, script: str):
    wikicaps_downloader = Path(script)
    assert wikicaps_downloader.exists()
    assert start < stop

    # run the shell script with different parts of the data
    sp.run(shlex.split(str(wikicaps_downloader) +
                       f" {start} {stop}"), capture_output=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_workers", default=10, type=int,
                        help="Number of worker threads to download the dataset in parallel.")
    parser.add_argument("--download_script", default=os.getcwd() + "/download_full_corpus_without_captions.sh",
                        type=str,
                        help="Number of worker threads to download the dataset in parallel.")
    args = parser.parse_args()

    # distribute the data over the workers
    num_total_imgs = 3825132
    imgs_per_worker = num_total_imgs // args.num_workers
    parts = [(w_i * imgs_per_worker, (w_i + 1) * imgs_per_worker - 1)
             for w_i in range(args.num_workers)]
    parts[-1] = (parts[-1][0], 3825132)

    print("Workers (start, stop) index:")
    print(parts)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for w_i in range(args.num_workers):
            executor.submit(download_part, *parts[w_i], script=args.download_script)
