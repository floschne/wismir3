import hashlib
import re
import time
import urllib.parse
from collections import Counter
from enum import Enum, unique
from pathlib import Path
from typing import Union, Tuple, List
from urllib.error import HTTPError, URLError

import nltk
import numpy as np
import pandas as pd
import regex
import requests
import spacy
from PIL import Image, UnidentifiedImageError
from loguru import logger
# from pandarallel import pandarallel
from polyglot.downloader import downloader
from polyglot.text import Text
from readability import Readability
from readability.exceptions import ReadabilityException
from skimage import io
from spacy_readability import Readability
from tqdm import tqdm

from transformations.image_transformation_base import ImageTransformationBase


@unique
class ImageOutputFormat(str, Enum):
    NPY = "npy"
    NPZ = "npz"
    PNG = "png"
    JPG = "jpg"


@unique
class MetadataGeneratorBackend(str, Enum):
    SPACY = "spacy"
    NLTK = "nltk"
    POLYGLOT = "polyglot"


def build_wikimedia_url(wikimedia_file_id: str, width: int, direct: bool = True) -> str:
    if direct:
        # see wikigrab.pl
        image = wikimedia_file_id.replace(" ", "_")
        image = re.sub(r'^(File|Image):', '', image)
        image = image[0].upper() + image[1:]
        digest = str(hashlib.md5(image.encode('utf-8')).hexdigest()).lower()
        a = digest[0]
        b = digest[0:2]

        image = urllib.parse.quote(image)  # encode special chars

        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{b}/{image}/{width}px-{image}"

    quoted = urllib.parse.quote(wikimedia_file_id)
    return f"https://commons.wikimedia.org/w/index.php?title=Special:FilePath&file={quoted}&width={width}"


def persist_img(img, dst: Path, wikicaps_id: int, img_out_format: ImageOutputFormat) -> Tuple[int, str]:
    logger.debug(f"Persisting image with WikiCaps ID {wikicaps_id} at {str(dst)}...")
    if img_out_format == ImageOutputFormat.NPY:
        np.save(str(dst), img)
    elif img_out_format == ImageOutputFormat.NPZ:
        np.savez_compressed(str(dst), 'img', img)
    elif img_out_format == ImageOutputFormat.PNG or img_out_format == ImageOutputFormat.JPG:
        io.imsave(str(dst), img)

    return wikicaps_id, str(dst)


def download_wikimedia_img(wikimedia_file_id: str,
                           wikicaps_id: int,
                           dst_path: Path,
                           img_out_format: ImageOutputFormat,
                           width: int = 500,
                           download_with_skimage=False) -> Tuple[int, Union[str, None]]:
    assert dst_path.is_dir(), "Destination path is not a directory!"
    dst = dst_path.joinpath(f"wikicaps_{wikicaps_id}.{img_out_format}")
    if dst.exists():
        logger.warning(f"File {str(dst)} already exists!")
        return wikicaps_id, str(dst)

    # try to download image from direct URL
    url = build_wikimedia_url(wikimedia_file_id, width)

    # wikimedia header for user agent
    # https://meta.wikimedia.org/wiki/User-Agent_policy

    headers = {
        'User-Agent': 'User-Agent: WICSMMIR_ETL_DOWNLOAD_BOT/0.1 (https://github.com/floschne/wicsmmirETL; floschne@github.com) wicsmmir-etl/0.1'
    }
    try:
        logger.debug(f"Downloading image with WikiCaps ID {wikicaps_id} from {url}...")
        if download_with_skimage:
            img = io.imread(url)
        else:
            resp = requests.get(url, stream=True, allow_redirects=True, timeout=.5, headers=headers)
            if resp.status_code == 200:
                img = np.asarray(Image.open(resp.raw))
            else:
                raise ConnectionError()
    except (HTTPError, TimeoutError, URLError, ConnectionError) as e:
        logger.warning(f"Error while trying to download '{wikimedia_file_id} from direct URL at {url}'!\n{e}")

        # retry download from indirect URL
        url = build_wikimedia_url(wikimedia_file_id, width, direct=False)
        logger.warning(f"Retrying to download '{wikimedia_file_id}' from WikiMedia from indirect URL at {url}'!")
        try:
            if download_with_skimage:
                img = io.imread(url)
            else:
                resp = requests.get(url, stream=True, allow_redirects=True, timeout=.5, headers=headers)
                if resp.status_code == 200:
                    img = np.asarray(Image.open(resp.raw))
                else:
                    raise ConnectionError()
        except (HTTPError, TimeoutError, URLError, UnidentifiedImageError, ConnectionError, Exception) as e:
            logger.error(f"Error while trying to download '{wikimedia_file_id}' from WikiMedia from {url}!\n{e}")
            return wikicaps_id, None
        else:
            return persist_img(img, dst, wikicaps_id, img_out_format)
    except (UnidentifiedImageError, Exception) as e:
        logger.exception(f"Error while trying to download '{wikimedia_file_id}' from WikiMedia!\n{e}")
        return wikicaps_id, None
    else:
        return persist_img(img, dst, wikicaps_id, img_out_format)


def apply_img_transformations(wikicaps_id: int,
                              img_path: str,
                              transformations: List[ImageTransformationBase]) -> Tuple[int, bool]:
    try:
        with Image.open(img_path) as img:
            for t in transformations:
                logger.debug(f"Applying {t.name} Image Transformation to {img_path}...")
                img = t(img, img_path=img_path)
            return wikicaps_id, True
    except Exception:
        logger.exception(f"Error while applying Image Transformations to {img_path}!")
        return wikicaps_id, False


def generate_corpus_vocab(dataframe: pd.DataFrame,
                          n_spacy_workers: int = 6,
                          spacy_model: str = "en_core_web_lg",
                          backend: MetadataGeneratorBackend = MetadataGeneratorBackend.SPACY) -> pd.DataFrame:
    # TODO support NLTK and Polyglot backends
    if backend != MetadataGeneratorBackend.SPACY:
        raise NotImplementedError("Currently only spaCy backend is supported!")

    logger.info(f"Generating corpus vocabulary using {backend.upper()}...")
    start = time.time()
    # collect ALL tokens and their POS of ALL captions
    tok_pos_cnt = Counter()

    with tqdm(total=len(dataframe)) as pbar:
        spacy_nlp = spacy.load(spacy_model)
        for doc in spacy_nlp.pipe(dataframe['caption'].astype(str), n_process=n_spacy_workers):
            for tok in doc:
                # strange syntax with double tuple to force tuple counting
                tok_pos_cnt.update(((tok.text, tok.pos_),))
            pbar.update(1)

    # transform the counter in a DataFrame
    vocab = pd.DataFrame.from_dict(tok_pos_cnt, orient='index').reset_index()
    vocab = vocab.rename(columns={'index': 'tok_pos', 0: 'count'})
    # split the tuples into own columns
    vocab[['token', 'pos']] = pd.DataFrame(vocab['tok_pos'].tolist(), index=vocab.index)
    # remove the original column
    vocab = vocab.drop('tok_pos', axis=1)
    # set multi index token -> pos -> count
    vocab = vocab.set_index(['token', 'pos'])
    vocab.sort_values(by=['count'], ascending=False, inplace=True)
    logger.info(f"Finished generating corpus vocabulary in {time.time() - start} seconds!")
    return vocab


def generate_caption_stats(dataframe: pd.DataFrame,
                           pos_tag_stats: bool = True,
                           readability_scores: bool = True,
                           n_spacy_workers: int = 6,
                           spacy_model: str = "en_core_web_lg",
                           backend: MetadataGeneratorBackend = MetadataGeneratorBackend.SPACY):
    logger.info(f"Generating caption statistics using {backend.upper()}...")
    start = time.time()

    # Tokens and sentences
    num_tok = []
    num_sent = []
    # Min and Max length of sentences
    min_sent_len = []
    max_sent_len = []

    # Named Entities
    num_ne = []
    ne_texts = []  # surface form of the NEs
    ne_types = []  # types of the NEs

    # POS Tags
    # counts
    num_noun = []  # nouns (cat, dog, house, tree, ...)
    num_propn = []  # proper nouns (Denver, Hamburg, Peter, Tesla, ...)
    num_conj = []  # conjunctions (and, or, ...)
    num_verb = []  # verbs
    num_sym = []  # symbols (!,#,?, ...)
    num_num = []  # numbers (IV, 1 billion, 1312, ...)
    num_adp = []  # adpositions (on, under, in, at, ...)
    num_adj = []  # adjectives (nice, fast, cool, ...)

    # ratios
    ratio_ne_tokens, num_ne_tok = [], []
    ratio_noun_tokens = []
    ratio_propn_tokens = []
    ratio_all_noun_tokens = []

    # readability scores
    fk_gl_score = []
    fk_re_score = []
    dc_score = []

    with tqdm(total=len(dataframe)) as pbar:
        # TODO extract all of this code into an own module and have separate metadata generators for spaCy, nltk, etc.
        if backend == MetadataGeneratorBackend.SPACY:
            # init spacy TODO: download the required model(s)
            spacy_nlp = spacy.load(spacy_model)
            if readability_scores:
                spacy_nlp.add_pipe(Readability())
            # TODO whats a good batch_size?
            for doc in spacy_nlp.pipe(dataframe['caption'].astype(str),
                                      n_process=n_spacy_workers):
                # num tokens
                num_tok.append(len(doc))

                # num sentences
                num_sent.append(len(list(doc.sents)))
                # min/max length of sentences
                min_len = 10000
                max_len = -1
                for s in doc.sents:
                    min_len = min(min_len, len(s))
                    max_len = max(max_len, len(s))
                min_sent_len.append(min_len)
                max_sent_len.append(max_len)

                # named entities
                num_ne.append(len(doc.ents))
                txt, typ = [], []
                for ent in doc.ents:
                    typ.append(ent.label_)
                    txt.append(ent.text)
                ne_texts.append(txt)
                ne_types.append(typ)

                # readability scores
                if readability_scores:
                    fk_gl_score.append(doc._.flesch_kincaid_grade_level)
                    fk_re_score.append(doc._.flesch_kincaid_reading_ease)
                    dc_score.append(doc._.dale_chall)

                # POS Tags
                if pos_tag_stats:
                    noun, propn, conj, verb, sym, num, adp, adj, ne_tok = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    for t in doc:
                        if t.pos_ == 'CONJ':
                            conj += 1
                        elif t.pos_ == 'ADJ':
                            adj += 1
                        elif t.pos_ == 'NOUN':
                            noun += 1
                        elif t.pos_ == 'NUM':
                            num += 1
                        elif t.pos_ == 'PROPN':
                            propn += 1
                        elif t.pos_ == 'SYM':
                            sym += 1
                        elif t.pos_ == 'VERB':
                            verb += 1
                        elif t.pos_ == 'ADP':
                            adp += 1

                        # number of tokens associated with a NE (to compute the ratio)
                        if t.ent_iob_ == 'I' or t.ent_iob_ == 'B':
                            ne_tok += 1

                    num_noun.append(noun)
                    num_propn.append(propn)
                    num_conj.append(conj)
                    num_verb.append(verb)
                    num_sym.append(sym)
                    num_num.append(num)
                    num_adp.append(adp)
                    num_adj.append(adj)

                    num_ne_tok.append(ne_tok)

                pbar.update(1)
        elif backend == MetadataGeneratorBackend.NLTK:
            nltk.download('punkt')
            nltk.download('words')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')
            nltk.download('universal_treebanks_v20')
            nltk.download('maxent_ne_chunker')

            for cap in dataframe['caption'].astype(str):
                # num tokens
                num_tok.append(len(nltk.word_tokenize(cap)))

                # num sentences
                sents = nltk.sent_tokenize(cap)
                num_sent.append(len(sents))

                # min/max length of sentences
                min_len = 10000
                max_len = -1
                s_toks = []
                for s in sents:
                    toks = nltk.word_tokenize(s)
                    s_toks.append(toks)
                    min_len = min(min_len, len(toks))
                    max_len = max(max_len, len(toks))
                min_sent_len.append(min_len)
                max_sent_len.append(max_len)

                # readability scores
                # FIXME currently not usable with NLTK... (because NaN values are dropped)
                #  there is also an error while calling t Readability(cap) ctor...
                if False:
                    try:
                        r = Readability(cap)
                        flesch = r.flesch_kincaid()
                        fk_gl_score.append(flesch.grade_level)
                        fk_re_score.append(flesch.score)
                        dc_score.append(r.dale_chall().score)
                    except (ReadabilityException, Exception):
                        fk_gl_score.append(np.NaN)
                        fk_re_score.append(np.NaN)
                        dc_score.append(np.NaN)

                if pos_tag_stats:
                    sent_pos_tags = nltk.pos_tag_sents(s_toks, 'universal')
                    noun, propn, conj, verb, sym, num, adp, adj, ne_tok = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    for spt in sent_pos_tags:
                        for pt in spt:
                            if pt[1].upper() == 'CONJ':
                                conj += 1
                            elif pt[1].upper() == 'ADJ':
                                adj += 1
                            elif pt[1].upper() == 'NOUN':
                                noun += 1
                            elif pt[1].upper() == 'NUM':
                                num += 1
                            elif pt[1].upper() == 'PROPN':
                                propn += 1
                            elif pt[1].upper() == 'SYM':
                                sym += 1
                            elif pt[1].upper() == 'VERB':
                                verb += 1
                            elif pt[1].upper() == 'ADP':
                                adp += 1

                    num_noun.append(noun)
                    num_propn.append(propn)
                    num_conj.append(conj)
                    num_verb.append(verb)
                    num_sym.append(sym)
                    num_num.append(num)
                    num_adp.append(adp)
                    num_adj.append(adj)

                # named entities
                # we have to tag again with a different tag set (upenn tree) for WAY better NER performance
                num_nes, num_nes_tok = 0, 0
                txt, typ = [], []
                nes_sent = nltk.ne_chunk_sents(nltk.pos_tag_sents(map(nltk.word_tokenize, nltk.sent_tokenize(cap))))
                for nes in nes_sent:
                    for ne in nes:
                        if isinstance(ne, nltk.Tree):
                            num_nes += 1
                            typ.append(str(ne.label()))
                            t = ""
                            for tok in ne:
                                t += tok[0] + " "
                            txt.append(t.strip())
                            num_nes_tok += len(ne)
                num_ne.append(num_nes)
                ne_texts.append(txt)
                ne_types.append(typ)
                num_ne_tok.append(num_nes_tok)

                pbar.update(1)
        elif backend == MetadataGeneratorBackend.POLYGLOT:
            # init
            # pandarallel.initialize()  # FIXME doens't work..
            downloader.download("embeddings2.en")
            downloader.download("ner2.en")
            downloader.download("pos2.en")

            def __gen_polyglot_metadata_per_caption(df, pb):
                d = {
                    'num_tok': 0,
                    'num_sent': 0,
                    'min_sent_len': 0,
                    'max_sent_len': 0,
                    'num_ne': 0,
                    'ne_types': [],
                    'ne_texts': [],
                    'num_nouns': 0,
                    'num_propn': 0,
                    'num_conj': 0,
                    'num_verb': 0,
                    'num_sym': 0,
                    'num_num': 0,
                    'num_adp': 0,
                    'num_adj': 0,
                    'ratio_ne_tok': 0.,
                    'ratio_noun_tok': 0.,
                    'ratio_propn_tok': 0.,
                    'ratio_all_noun_tok': 0.,
                }
                try:
                    caption = str(df['caption']).encode('utf-8')
                    # https://github.com/aboSamoor/polyglot/issues/71
                    # removing "bad unicode" characters to avoid runtime exceptions
                    # caption = str(caption, encoding='utf-8')
                    caption = regex.sub(r"\p{C}", "", caption.decode('utf-8'))

                    pg = Text(caption, hint_language_code='en')
                    pg.language = 'en'
                    # num tokens
                    n_tok = len(pg.words)

                    # num sentences
                    n_sent = len(pg.sentences)

                    # min/max length of sentences
                    min_s_len = 10000
                    max_s_len = -1
                    for s in pg.sentences:
                        min_s_len = min(min_s_len, len(s.words))
                        max_s_len = max(max_s_len, len(s.words))
                    # readability scores
                    # FIXME only available with spacy currently

                    # POS tags
                    n_noun, n_propn, n_conj, n_verb, n_sym, n_num, n_adp, n_adj = 0, 0, 0, 0, 0, 0, 0, 0
                    for pos in pg.pos_tags:
                        if pos[1].upper() == 'CONJ':
                            n_conj += 1
                        elif pos[1].upper() == 'ADJ':
                            n_adj += 1
                        elif pos[1].upper() == 'NOUN':
                            n_noun += 1
                        elif pos[1].upper() == 'NUM':
                            n_num += 1
                        elif pos[1].upper() == 'PROPN':
                            n_propn += 1
                        elif pos[1].upper() == 'SYM':
                            n_sym += 1
                        elif pos[1].upper() == 'VERB':
                            n_verb += 1
                        elif pos[1].upper() == 'ADP':
                            n_adp += 1

                    # named entities
                    num_nes_tok, ne_txt, ne_typ = 0, [], []
                    num_nes = len(pg.entities)
                    for ne in pg.entities:
                        num_nes_tok += len(ne)
                        ne_txt.append(" ".join(ne))
                        ne_typ.append(ne.tag)

                    # compute the rations
                    r_ne_tokens = num_nes_tok / n_tok
                    r_noun_tokens = n_noun / n_tok
                    r_propn_tokens = n_propn / n_tok
                    r_all_noun_tokens = (n_noun + n_propn) / n_tok
                    d = {
                        'num_tok': n_tok,
                        'num_sent': n_sent,
                        'min_sent_len': min_s_len,
                        'max_sent_len': max_s_len,
                        'num_ne': num_nes,
                        'ne_types': ne_typ,
                        'ne_texts': ne_txt,
                        'num_nouns': n_noun,
                        'num_propn': n_propn,
                        'num_conj': n_conj,
                        'num_verb': n_verb,
                        'num_sym': n_sym,
                        'num_num': n_num,
                        'num_adp': n_adp,
                        'num_adj': n_adj,
                        'ratio_ne_tok': r_ne_tokens,
                        'ratio_noun_tok': r_noun_tokens,
                        'ratio_propn_tok': r_propn_tokens,
                        'ratio_all_noun_tok': r_all_noun_tokens,
                    }
                except Exception as e:
                    logger.error(f"Critical error occurred with caption of WikiCaps ID{df['wikicaps_id']}!")
                    logger.error(str(e))
                    return
                finally:
                    pb.update(1)
                    return d

            # FIXME why the hec is this using ALL AVAILABLE CORES?!
            metadata = dataframe.apply(__gen_polyglot_metadata_per_caption, axis=1, result_type='expand', args=(pbar,))
            res = pd.concat([dataframe, metadata], axis=1)
            res.convert_dtypes()

            logger.info(f"Finished adding caption statistics in {time.time() - start} seconds!")
            return res

    # compute the rations
    if pos_tag_stats:
        np_num_tok = np.array(num_tok)
        np_num_noun = np.array(num_noun)
        np_num_propn = np.array(num_propn)
        ratio_ne_tokens = (np.array(num_ne_tok) / np_num_tok)
        ratio_noun_tokens = (np_num_noun / np_num_tok)
        ratio_propn_tokens = (np_num_propn / np_num_tok)
        ratio_all_noun_tokens = ((np_num_noun + np_num_propn) / np_num_tok)

    res = dataframe.copy()

    # add stats as columns to df
    res['num_tok'] = num_tok

    res['num_sent'] = num_sent
    res['min_sent_len'] = min_sent_len
    res['max_sent_len'] = max_sent_len

    res['num_ne'] = num_ne
    res['ne_types'] = ne_types
    res['ne_texts'] = ne_texts

    if pos_tag_stats:
        res['num_nouns'] = num_noun
        res['num_propn'] = num_propn
        res['num_conj'] = num_conj
        res['num_verb'] = num_verb
        res['num_sym'] = num_sym
        res['num_num'] = num_num
        res['num_adp'] = num_adp
        res['num_adj'] = num_adj

        res['ratio_ne_tok'] = ratio_ne_tokens
        res['ratio_noun_tok'] = ratio_noun_tokens
        res['ratio_propn_tok'] = ratio_propn_tokens
        res['ratio_all_noun_tok'] = ratio_all_noun_tokens

    if readability_scores:
        res['fk_re_score'] = fk_re_score
        res['fk_gl_score'] = fk_gl_score
        res['dc_score'] = dc_score

    res.convert_dtypes()  # make sure that ints are not encoded as floats
    logger.info(f"Finished adding caption statistics in {time.time() - start} seconds!")

    return res
