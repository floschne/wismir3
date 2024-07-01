import os
import sys
from argparse import Namespace

sys.path.append(str(os.getcwd()))

from main import main


def run_pipeline_with(metadata_generator: str):
    opts = Namespace()
    opts.config = f'./configs/config_localhost_test_{metadata_generator}.yml'

    main(opts)


def test_pipeline_with_nltk():
    # just test that it runs w/o exception
    run_pipeline_with('nltk')


def test_pipeline_with_spacy():
    # just test that it runs w/o exception
    run_pipeline_with('spacy')


def test_pipeline_with_polyglot():
    # just test that it runs w/o exception
    run_pipeline_with('polyglot')
