#!/bin/bash

#useage: ./download_full_corpus_without_captions.sh START_INDEX END_INDEX

shopt -s extglob

numre='^[0-9]+$'

START=$(($1 + 0))
END=$(($2 + 0))

if ! [[ $1 =~ $numre ]]; then
  echo "Start index must be a numer (integer)!"
  echo "Useage: ./download_full_corpus_without_captions.sh START_INDEX END_INDEX"
  exit 1
fi

if ! [[ $2 =~ $numre ]]; then
  echo "End/stopping index must be a numer (integer)!"
  echo "Useage: ./download_full_corpus_without_captions.sh START_INDEX END_INDEX"
  exit 2
fi

if (($START > $END)); then
  echo "End/stopping index must larger than starting index!"
  echo "Useage: ./download_full_corpus_without_captions.sh START_INDEX END_INDEX"
  exit 3
fi

i=0
dirnumber=0

while read p; do

  if ((i % 10000 == 0)); then
    dirnumber=$((dirnumber + 1))
    echo $dirnumber
  fi

  if ((i < $START)); then
    i=$((i + 1))
    continue
  fi

  if ((i > $END)); then
    i=$((i + 1))
    break
  fi

  if ! [[ -d $dirnumber ]]; then
    mkdir $dirnumber
  fi

  #Create download cache:
  mkdir tmp
  cd tmp

  #Download image:
  cp ../wikimgrab.pl .
  perl ./wikimgrab.pl "$p"
  mv !(wikimgrab.pl|download_dev_test.sh|+([0-9])) ../$dirnumber

  #Cleanup:
  cd ..
  rm -rf tmp
  i=$((i + 1))

done <../images_retrieval.lst
