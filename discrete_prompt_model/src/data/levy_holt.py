from typing import Tuple, Union, Dict, Optional, List, Iterable
from torch.utils.data import Dataset
from overrides import overrides
import sys
from .common import PREM_KEY, HYPO_KEY, LABEL_KEY, LANGUAGE_KEY, PATTERNS, ANTIPATTERNS,\
    SENT_KEY, ANTI_KEY, load_patterns, chunks, ANTIPATTERNS_GERMAN, PATTERNS_GERMAN, ANTIPATTERNS_CHINESE, \
    PATTERNS_CHINESE, PATTERNS_DIR, PATTERNS_GERMAN_DIR, PATTERNS_CHINESE_DIR, reconstruct_sentence_from_triple


class LevyHoltBase(Dataset):
    def __init__(self, txt_file, symmetric):
        if txt_file is not None:
            self.data = self.load_dataset(txt_file, symmetric=symmetric)
        else:
            print(f"LevyHolt Dataset init: txt_file is None! This is unexpected if loading fixed existing dataset!",
                  file=sys.stderr)
            self.data = None

    def load_dataset(self, txt_file, symmetric):
        data = []
        with open(txt_file, 'r', encoding='utf8') as f:
            for line in f:
                hypo, prem, label, language = line.strip().split('\t')
                hypo = tuple(h.strip() for h in hypo.split(','))
                prem = tuple(p.strip() for p in prem.split(','))
                label = label == 'True'
                data.extend(self.create_instances(prem, hypo, label, language, symmetric=symmetric))
        return data

    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            language: str,
            symmetric: bool = False
    ) -> List[Dict[str, Union[bool, str]]]:
        raise NotImplementedError(
            "You have to implement `create_instances` in a subclass inheriting from `LevyHoltBase`"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError(
            "You have to implement `__getitem__` in a subclass inheriting from `LevyHoltBase`"
        )


class LevyHoltSentences(LevyHoltBase):

    @overrides
    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            language: str,
            symmetric: bool = False
    ) -> List[Dict[str, Union[bool, str]]]:
        inst = {
            PREM_KEY: reconstruct_sentence_from_triple(prem, lang=language, mode='sent'),
            HYPO_KEY: reconstruct_sentence_from_triple(hypo, lang=language, mode='sent'),
            LABEL_KEY: label,
            LANGUAGE_KEY: language
        }
        return [inst]

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return inst[PREM_KEY], inst[HYPO_KEY], inst[LABEL_KEY], inst[LANGUAGE_KEY]


class LevyHoltPattern(LevyHoltBase):

    # ``is_directional'' means whether 5 -> 3, explicitly directional;
    # ``symmetric'' means whether 3 -> 5, explicitly symmetric.
    def __init__(self, txt_file: str, pattern_file: Optional[str] = None,
                 antipattern_file: Optional[str] = None, best_k_patterns: Optional[int] = None,
                 pattern_chunk_size: int = 5, training: bool = False,
                 curated_auto: bool = False, is_directional: bool = False):
        self.training = training
        self.pattern_chunk_size = pattern_chunk_size

        print(f"TXT_FILE: {txt_file}; IS_DIRECTIONAL: {is_directional};")

        if pattern_file is None:
            self.patterns = PATTERNS_DIR  # if is_directional else PATTERNS
            self.antipatterns = ANTIPATTERNS
            self.handcrafted = True
            self.patterns_de = PATTERNS_GERMAN_DIR  # if is_directional else PATTERNS_GERMAN
            self.antipatterns_de = ANTIPATTERNS_GERMAN
            self.patterns_zh = PATTERNS_CHINESE_DIR  # if is_directional else PATTERNS_CHINESE
            self.antipatterns_zh = ANTIPATTERNS_CHINESE
        else:
            if best_k_patterns is not None and best_k_patterns % pattern_chunk_size != 0:
                print(
                    "WARNING: best_k_patterns should be a"
                    + " multiple of pattern_chunk_size ({})".format(pattern_chunk_size))
            self.patterns = load_patterns(pattern_file, best_k_patterns)
            assert antipattern_file is not None,\
                "pattern_file and antipattern_file must either"\
                + " both be None or both set to file paths."
            self.antipatterns = load_patterns(
                antipattern_file, best_k_patterns)
            self.handcrafted = curated_auto

        symmetric = not is_directional
        super().__init__(txt_file, symmetric=symmetric)

    def create_sent_from_pattern(self, pattern: str, prem: Tuple[str, str, str],
                                 hypo: Tuple[str, str, str], lang: str) -> str:

        if self.handcrafted:
            if lang in ['EN', 'DE']:
                sent = pattern.format(pal=prem[0], prem=prem[1], par=prem[2],
                                      hal=hypo[0], hypo=hypo[1], har=hypo[2])
            elif lang in ['ZH']:
                # prem_subj, prem_vp = reconstruct_sentence_from_triple(prem, lang, mode='s-vp')
                # hypo_subj, hypo_vp = reconstruct_sentence_from_triple(hypo, lang, mode='s-vp')
                prem_sent = reconstruct_sentence_from_triple(prem, lang, mode='sent')
                hypo_sent = reconstruct_sentence_from_triple(hypo, lang, mode='sent')
                # sent = pattern.format(prem_subj=prem_subj, prem_vp=prem_vp,
                #                       hypo_subj=hypo_subj, hypo_vp=hypo_vp)
                sent = pattern.format(prem=prem_sent, hypo=hypo_sent)
            else:
                raise AssertionError
        else:
            assert lang not in ['ZH']
            sent = pattern.format(prem=prem[1], hypo=hypo[1])
        return sent

    def create_single_instance(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            patterns: Iterable[str],
            antipatterns: Iterable[str],
            lang: str) -> Dict[str, Union[bool, str]]:
        inst = {}

        inst[SENT_KEY] = [
            self.create_sent_from_pattern(pat, prem, hypo, lang)
            for pat in patterns
        ]
        inst[ANTI_KEY] = [
            self.create_sent_from_pattern(pat, prem, hypo, lang)
            for pat in antipatterns
        ]
        inst[LABEL_KEY] = label

        return inst

    @overrides
    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            language: str,
            symmetric: bool = False
    ) -> List[Dict[str, Union[bool, str]]]:
        instances = []

        if self.training:
            if language == "EN":
                chunked = [
                    chunks(self.patterns, self.pattern_chunk_size),
                    chunks(self.antipatterns, self.pattern_chunk_size)
                ]
            elif language == 'DE':
                chunked = [
                    chunks(self.patterns_de, self.pattern_chunk_size),
                    chunks(self.antipatterns_de, self.pattern_chunk_size)
                ]
            elif language == 'ZH':
                chunked = [
                    chunks(self.patterns_zh, self.pattern_chunk_size),
                    chunks(self.antipatterns_zh, self.pattern_chunk_size)
                ]
            else:
                raise AssertionError
            for pattern_chunk, antipattern_chunk in zip(*chunked):
                inst = self.create_single_instance(
                    prem, hypo, label, pattern_chunk, antipattern_chunk, lang=language)
                if symmetric:
                    inst_rev = self.create_single_instance(
                    hypo, prem, label, pattern_chunk, antipattern_chunk, lang=language)
                    inst[SENT_KEY] += inst_rev[SENT_KEY]
                    inst[ANTI_KEY] += inst_rev[ANTI_KEY]
                instances.append(inst)
        else:
            if language == 'EN':
                patterns = self.patterns
                antipatterns = self.antipatterns
            elif language == 'DE':
                patterns = self.patterns_de
                antipatterns = self.antipatterns_de
            elif language == 'ZH':
                patterns = self.patterns_zh
                antipatterns = self.antipatterns_zh
            else:
                raise AssertionError
            inst = self.create_single_instance(
                prem, hypo, label, patterns, antipatterns, lang=language)
            if symmetric:
                inst_rev = self.create_single_instance(
                hypo, prem, label, patterns, antipatterns, lang=language)
                inst[SENT_KEY] += inst_rev[SENT_KEY]
                inst[ANTI_KEY] += inst_rev[ANTI_KEY]
            instances.append(inst)

        return instances

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return inst[SENT_KEY], inst[ANTI_KEY], inst[LABEL_KEY]

