#!/usr/bin/env bash
# Fetch the cross-domain few-shot benchmark datasets.
#
#   bash scripts/download_data.sh isic
#   bash scripts/download_data.sh chestx     # see the note below -- not automatic
#
# EuroSAT needs nothing: torchvision downloads it on first use.

set -euo pipefail

DATA=${DATA:-data}
mkdir -p "$DATA"

have() { [ -e "$1" ]; }

isic() {
    # ISIC 2018 Task 3 (= HAM10000), 7 classes, ~10k images, 2.6 GB.
    local dir="$DATA/isic2018"
    mkdir -p "$dir"

    local base=https://isic-challenge-data.s3.amazonaws.com/2018
    for part in ISIC2018_Task3_Training_Input ISIC2018_Task3_Training_GroundTruth; do
        if have "$dir/$part"; then
            echo "[skip] $part already present"
            continue
        fi
        echo "[get ] $part"
        curl -L --fail --retry 3 --continue-at - \
             -o "$dir/$part.zip" "$base/$part.zip"
        unzip -q -o "$dir/$part.zip" -d "$dir"
        rm -f "$dir/$part.zip"
    done

    # HAM10000_metadata.csv carries lesion_id, which is the ONLY way to know that two
    # images are views of the same physical lesion. Without it, a split by image leaks
    # and accuracy is inflated for reasons unrelated to the method. The ISIC S3 bucket
    # does not serve it, so it has to come from the Harvard Dataverse copy of HAM10000.
    if ! have "$dir/HAM10000_metadata.csv"; then
        echo
        echo "NOTE: HAM10000_metadata.csv is not in the ISIC bucket and must be"
        echo "      fetched from the Harvard Dataverse HAM10000 record, then placed at"
        echo "          $dir/HAM10000_metadata.csv"
        echo "      Without it every image is treated as a separate lesion and the"
        echo "      leakage-safe split cannot do its job. ~10k images come from ~7.5k"
        echo "      lesions, so this is a real effect, not a theoretical one."
        echo
    fi
    echo "[done] ISIC in $dir"
}

chestx() {
    # NIH ChestX-ray14: 112,120 images, ~42 GB, 12 tarballs.
    #
    # There is no stable public direct link. The NIH Box share that the community
    # scripts used (nihcc.box.com/shared/static/...) now returns 404, and so does the
    # folder landing page. Verified 2026-10-12. Two working routes:
    #
    #   1. Kaggle (needs an account and an API token at ~/.kaggle/kaggle.json):
    #        pip install kaggle
    #        kaggle datasets download -d nih-chest-xrays/data -p data/chestx
    #        cd data/chestx && unzip -q data.zip
    #
    #   2. Manual: download the 12 images_*.tar.gz and Data_Entry_2017.csv from the
    #      NIH ChestX-ray14 release, extract them all under data/chestx/.
    #
    # The loader indexes images by filename recursively, so either the flat layout or
    # the per-tarball images_001/images/... layout works as-is.
    local dir="$DATA/chestx"
    mkdir -p "$dir"

    if have "$dir/Data_Entry_2017.csv"; then
        echo "[ok  ] ChestX manifest present; loader will index $(find "$dir" -name '*.png' | wc -l) images"
        return 0
    fi

    if command -v kaggle >/dev/null 2>&1 && [ -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "[get ] ChestX via Kaggle (~42 GB, this will take a while)"
        kaggle datasets download -d nih-chest-xrays/data -p "$dir"
        (cd "$dir" && unzip -q -o data.zip && rm -f data.zip)
        echo "[done] ChestX in $dir"
        return 0
    fi

    cat >&2 <<'MSG'
ChestX-ray14 cannot be fetched automatically.

  The NIH Box links the usual scripts rely on are dead (404 as of 2026-10-12),
  and no Kaggle CLI or credentials were found.

  To enable the Kaggle route:
      pip install kaggle
      # create an API token at kaggle.com -> Account -> Create New API Token
      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json
      bash scripts/download_data.sh chestx

  Or place the 12 extracted images_*.tar.gz directories plus Data_Entry_2017.csv
  under data/chestx/ by hand.
MSG
    return 1
}

case "${1:-}" in
    isic)   isic ;;
    chestx) chestx ;;
    *)      echo "usage: $0 {isic|chestx}" >&2; exit 1 ;;
esac
