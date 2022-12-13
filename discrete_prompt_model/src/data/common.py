from typing import Optional, List, TypeVar, Iterable
import re

PREM_KEY = 'premise'
HYPO_KEY = 'hypothesis'
LABEL_KEY = 'label'
SENT_KEY = 'sentence'
ANTI_KEY = 'neg_sentence'
MASKED_SENT_KEY = 'masked_sentence'
MASKED_ANTI_KEY = 'masked_neg_sentence'
LANGUAGE_KEY = 'language'


PATTERNS = [
    "{pal} {prem} {par}, which means that {hal} {hypo} {har}.",
    "It is not the case that {hal} {hypo} {har}, let alone that {pal} {prem} {par}.",
    "{hal} {hypo} {har} because {pal} {prem} {par}.",
    "{pal} {prem} {par} because {hal} {hypo} {har}.",
    "{hal} {hypo} {har}, which means that {pal} {prem} {par}."
]

PATTERNS_DIR = [
    "{pal} {prem} {par}, which means that {hal} {hypo} {har}.",
    "It is not the case that {hal} {hypo} {har}, let alone that {pal} {prem} {par}.",
    "{hal} {hypo} {har} because {pal} {prem} {par}.",
]

PATTERNS_GERMAN = [
    "{pal} {prem} {par}, das heisst {hal} {hypo} {har}.",
    "Es ist nicht der Fall, dass {hal} {har} {hypo}, geschweige denn {pal} {par} {prem}.",
    "{hal} {hypo} {har} denn {pal} {prem} {par}.",
    "{pal} {prem} {par} denn {hal} {hypo} {har}.",
    "{hal} {hypo} {har}, was bedeuted dass {pal} {par} {prem}."
]

PATTERNS_GERMAN_DIR = [
    "{pal} {prem} {par}, das heisst {hal} {hypo} {har}.",
    "Es ist nicht der Fall, dass {hal} {har} {hypo}, geschweige denn {pal} {par} {prem}.",
    "{hal} {hypo} {har} denn {pal} {prem} {par}.",
]

# TODO: why are the patterns symmetric?
# PATTERNS_CHINESE = [
#     "{prem_subj}{prem_vp}，这也就是说，{hypo_subj}{hypo_vp}。",
#     "{hypo_subj}{hypo_vp}是不成立的，所以{prem_subj}{prem_vp}就更不可能了。",
#     "{prem_subj}{prem_vp}，所以{hypo_subj}{hypo_vp}",
#     "{hypo_subj}{hypo_vp}，所以{prem_subj}{prem_vp}",
#     "{hypo_subj}{hypo_vp}，这意味着{prem_subj}{prem_vp}"
# ]

PATTERNS_CHINESE = [
    "{prem}，这也就是说，{hypo}。",
    "{hypo}是不成立的，所以{prem}就更不可能了。",
    "{prem}，所以{hypo}",
    "{hypo}，所以{prem}",
    "{hypo}，这意味着{prem}"
]

PATTERNS_CHINESE_DIR = [
    "{prem}，这也就是说，{hypo}。",
    "{hypo}是不成立的，所以{prem}就更不可能了。",
    "{prem}，所以{hypo}",
]

NEGATION_NECESSARY = [
    (False, False),
    (False, False),
    (False, False),
    (True, True),
    (True, True)
]

ANTIPATTERNS = [
    "It is not sure that {hal} {hypo} {har} just because {pal} {prem} {par}.",
    "{pal} {prem} {par}. This does not mean that {hal} {hypo} {har}.",
    "The fact that {pal} {prem} {par} does not necessarily mean that {hal} {hypo} {har}.",
    "Even if {pal} {prem} {par}, {hal} maybe {hypo} {har}.",
    "Just because {pal} {prem} {par}, it might still not be true that {hal} {hypo} {har}."
]

ANTIPATTERNS_GERMAN = [
    "Es ist ungewiss dass {hal} {har} {hypo} nur weil {pal} {par} {prem}.",
    "{pal} {prem} {par}. Das bedeutet nicht, dass {hal} {har} {hypo}.",
    "{pal} {prem} {par} heisst nicht, dass {hal} {har} {hypo}.",
    "Selbst wenn {pal} {par} {prem}, {hypo} {hal} vielleicht {har}.",
    "Nur weil {pal} {par} {prem}, kann es trotzdem falsch sein, dass {hal} {har} {hypo}."
]

# ANTIPATTERNS_CHINESE = [
#     "只因为{prem_subj}{prem_vp}，{hypo_subj}未必{hypo_vp}。",
#     "{prem_subj}{prem_vp}。这并不意味着{hypo_subj}{hypo_vp}。",
#     "{prem_subj}{prem_vp}并不一定使得{hypo_subj}{hypo_vp}。",
#     "即使{prem_subj}{prem_vp}，{hypo_subj}也只是有可能{hypo_vp}。",
#     "只从{prem_subj}{prem_vp}来看，{hypo_subj}{hypo_vp}这件事仍有可能不是真的。"
# ]

ANTIPATTERNS_CHINESE = [
    "只因为{prem}，还不能确定{hypo}。",
    "{prem}。这并不意味着{hypo}。",
    "{prem}并不一定导致{hypo}。",
    "即使{prem}，也只是有可能{hypo}。",
    "只从{prem}来看，{hypo}这件事仍有可能不是真的。"
]

ANTI_NEGATION_NECESSARY = [  # this is used only for Sherliic - Teddy
    (False, False),
    (False, False),
    (False, False),
    (False, True),
    (False, False)
]


def choose_examples(examples_A, examples_B, is_reversed: bool):
    if is_reversed:
        return examples_B[0], examples_A[0]
    else:
        return examples_A[0], examples_B[0]


def negate(verb_phrase: str) -> str:
    tokens = re.split(r'\s+', verb_phrase)
    if tokens[0] in ['is', 'are', 'were', 'was']:
        new_tokens = tokens[:1] + ['not'] + tokens[1:]
    else:
        if tokens[0].endswith('s'):
            new_tokens = ['does', 'not', tokens[0][:-1]] + tokens[1:]
        else:
            new_tokens = ['do', 'not', tokens[0][:-1]] + tokens[1:]
    return ' '.join(new_tokens)


def mask_equivalent(self, string: str, mask_token, tokenizer, add_space=True) -> str:
    longer_string = mask_token
    if add_space:
        longer_string = longer_string + ' '
    longer_string = longer_string + string.strip()
    num_tokens = len(
        tokenizer.encode(longer_string, add_special_tokens=False)
    ) - 1
    return " ".join([mask_token] * num_tokens)


def load_patterns(pattern_file: str, best_k_patterns: Optional[int]) -> List[str]:
    patterns = []
    with open(pattern_file, 'r', encoding='utf8') as f:
        for line in f:
            score, pattern = line.strip().split('\t')
            patterns.append(pattern)
            if best_k_patterns is not None and len(patterns) == best_k_patterns:
                break
    return patterns


T = TypeVar('T')


def chunks(lst: List[T], n: int) -> Iterable[List[T]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def reconstruct_sentence_from_triple(tpl, lang, mode):
    def find_all_matches_in_string(string, pattern):
        if len(pattern) == 0:
            print(f"Pattern with zero length!")
            return []
        id_list = []
        offset = 0
        while True:
            cur_id = string.find(pattern, offset)
            if cur_id < 0:
                break
            id_list.append(cur_id)
            offset = cur_id + len(pattern)
        return id_list

    if lang in ['EN', 'DE']:
        assert mode == 'sent'
        return ' '.join(tpl)
    elif lang in ['ZH']:
        if len(tpl[1]) == 0:
            assert len(tpl[2]) == 0
            if mode == 'sent':
                return tpl[0]
            elif mode == 's-vp':
                raise NotImplementedError
            else:
                raise AssertionError
        else:
            assert len(tpl[2]) > 0
        s, v, o = tpl
        if '[UNK][UNK]占位符主语' in s:
            assert '[UNK][UNK]占位符谓语' in v and '[UNK][UNK]占位符宾语' in o
            if mode == 'sent':
                return '占位符占位符占位符'
            elif mode == 's-vp':
                return '占位符', '占位符占位符'
            else:
                raise AssertionError
        if '【介宾】' in v:
            assert '·X·' in v
            v.replace('·X·', '·'+s+'·')
            v.replace('【介宾】', '')
            s = ''  # the ``s'' above is really the adjunct in prepositional phrase, the true subject is unknown, thus replaced with ''
        if '否·' in v:
            assert v.startswith('否·')
            v = '没有·' + v[2:]
        upred_xidxs = find_all_matches_in_string(v, '·X·')
        if len(upred_xidxs) == 0:
            v = v.replace('·', '')
            vp = v + o
        elif len(upred_xidxs) == 1:
            vp = v.replace('·X·', o)
            vp = vp.replace('·', '')
        else:
            raise AssertionError

        if mode == 'sent':
            return s + vp
        elif mode == 's-vp':
            return s, vp
        else:
            raise AssertionError
    else:
        raise AssertionError