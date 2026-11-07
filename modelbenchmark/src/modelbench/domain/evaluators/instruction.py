"""IFEval-style verifiable-instruction evaluator (subset).

IFEval samples bundle a list of *verifiable instructions* with keyword arguments::

    instruction_id_list = ["keywords:existence", "length_constraints:number_words"]
    kwargs              = [{"keywords": ["climate", "policy"]},
                           {"num_words": 120, "relation": "at least"}]

:func:`check_instructions` returns one entry per instruction: ``True`` (passed),
``False`` (failed) or ``None`` (not judgeable, e.g. an unimplemented instruction
id).  ``None`` entries are excluded from the denominator, so partial-coverage
datasets do not silently count as failures - the bias is reported through
``detail["n_unknown"]``.

Implemented instruction families (18 ids):

* ``keywords:existence`` / ``keywords:frequency`` / ``keywords:forbidden_words``
  / ``keywords:letter_frequency``
* ``language:response_language`` (heuristic script + stop-word detector, since no
  ``langdetect`` dependency is allowed; returns ``None`` when undecidable)
* ``length_constraints:number_words`` / ``number_sentences`` /
  ``number_paragraphs`` / ``nth_paragraph_first_word``
* ``detectable_format:number_bullet_lists`` /
  ``number_highlighted_sections`` / ``multiple_sections``
* ``change_case:english_capital`` / ``change_case:english_lowercase``
* ``startend:end_checker`` / ``punctuation:no_comma``
* ``combination:two_responses`` / ``combination:repeat_prompt``

Documented deviations from the reference implementation (kept intentionally
small, all on the *lenient* side):

* ``detectable_format:multiple_sections`` counts section markers and requires at
  least ``num_sections`` of them (the reference compares the raw split length).
* ``detectable_format:number_bullet_lists`` also accepts ``-`` and ``•`` bullets
  in addition to the reference's ``*`` marker.
* ``language:response_language`` is heuristic rather than langdetect-based.

Unknown instruction ids return ``None`` (never raise); every check is wrapped so
that malformed kwargs degrade to ``None`` instead of an exception.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from ..models import RawResponse, Sample, Score

InstructionResult = Optional[bool]
InstructionChecker = Any  # (text: str, kwargs: Dict[str, Any]) -> InstructionResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relation_ok(count: int, target: float, relation: Optional[str]) -> InstructionResult:
    """Apply an IFEval numeric relation; ``None`` for unknown relations."""
    rel = (relation or "").strip().lower()
    if rel in {"less than", "<", "below", "under"}:
        return count < target
    if rel in {"at least", ">=", "minimum", "no less than", "not less than"}:
        return count >= target
    if rel in {"more than", ">", "greater than", "over"}:
        return count > target
    if rel in {"at most", "<=", "maximum", "no more than", "not more than"}:
        return count <= target
    if rel in {"exactly", "equal to", "equals", "==", "equal"}:
        return count == target
    return None


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _split_sentences(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?。！？])(?:\s+|$)", stripped)
    return [part for part in parts if part.strip()]


def _split_paragraphs(text: str) -> List[str]:
    return [part for part in (text or "").split("\n\n") if part.strip()]


def _first_word(paragraph: str) -> str:
    tokens = paragraph.strip().split()
    if not tokens:
        return ""
    return tokens[0].strip("*_`#>.,;:!?\"'“”‘’。，、：；！？()[]{}<>")


def _count_bullet_lines(text: str) -> int:
    return len(re.findall(r"^[ \t]*[\*\-•][ \t]+\S", text or "", flags=re.MULTILINE))


def _count_highlights(text: str) -> int:
    return len(re.findall(r"\*[^*\n]+\*", text or ""))


# --------------------------------------------------------------------------- #
# Language heuristic (no external dependency)
# --------------------------------------------------------------------------- #

_SCRIPT_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "han": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "hiragana": re.compile(r"[\u3040-\u309f]"),
    "katakana": re.compile(r"[\u30a0-\u30ff]"),
    "hangul": re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "thai": re.compile(r"[\u0e00-\u0e7f]"),
    "greek": re.compile(r"[\u0370-\u03ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "latin": re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u017f]"),
}

_LANGUAGE_ALIASES: Dict[str, str] = {
    "en": "en", "eng": "en", "english": "en",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr", "français": "fr", "francais": "fr",
    "de": "de", "deu": "de", "ger": "de", "german": "de", "deutsch": "de",
    "es": "es", "spa": "es", "spanish": "es", "español": "es", "espanol": "es",
    "it": "it", "ita": "it", "italian": "it", "italiano": "it",
    "pt": "pt", "por": "pt", "portuguese": "pt", "português": "pt", "portugues": "pt",
    "nl": "nl", "nld": "nl", "dut": "nl", "dutch": "nl", "nederlands": "nl",
    "zh": "zh", "zho": "zh", "chi": "zh", "chinese": "zh", "中文": "zh", "汉语": "zh",
    "ja": "ja", "jpn": "ja", "japanese": "ja", "日本語": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko", "한국어": "ko",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "ar": "ar", "ara": "ar", "arabic": "ar",
    "hi": "hi", "hin": "hi", "hindi": "hi",
    "th": "th", "tha": "th", "thai": "th",
    "el": "el", "ell": "el", "gre": "el", "greek": "el",
    "he": "he", "heb": "he", "hebrew": "he",
}

#: A few high-frequency function words per Latin-script language.
_STOPWORDS: Dict[str, frozenset] = {
    "en": frozenset(
        "the and is are of to in that it for with as was on be this have from at or "
        "by not but they you we an i".split()
    ),
    "fr": frozenset(
        "le la les de des du un une et est que qui dans pour avec ce cette il elle "
        "je vous nous pas sur au aux se son sa ses plus mais".split()
    ),
    "de": frozenset(
        "der die das und ist ein eine nicht mit für auf von zu den dem des sich auch "
        "es im sie wir ich werden wird sind war".split()
    ),
    "es": frozenset(
        "el la los las de del que y es en un una por con para no se su al lo como "
        "más pero este esta".split()
    ),
    "it": frozenset(
        "il lo la gli le di del che e è in un una per con non si da al come più "
        "questo questa sono".split()
    ),
    "pt": frozenset(
        "o a os as de do da que e é em um uma para com não se por ao mais como "
        "este esta são".split()
    ),
    "nl": frozenset(
        "de het een en van is in op te dat die niet met voor zijn er ook als aan "
        "door maar".split()
    ),
}

_GREETING_HINTS: Dict[str, frozenset] = {
    "fr": frozenset("bonjour merci voici réponse français".split()),
    "de": frozenset("hallo danke bitte deutsch antwort".split()),
    "es": frozenset("hola gracias respuesta español".split()),
    "it": frozenset("ciao grazie risposta italiano".split()),
    "pt": frozenset("olá obrigado resposta português".split()),
    "nl": frozenset("hallo bedankt antwoord nederlands".split()),
}

_SCRIPT_TO_LANGUAGE = {
    "cyrillic": "ru",
    "arabic": "ar",
    "devanagari": "hi",
    "thai": "th",
    "greek": "el",
    "hebrew": "he",
}


def _normalize_language_name(raw: str) -> Optional[str]:
    key = unicodedata.normalize("NFKC", str(raw or "")).strip().lower()
    if not key:
        return None
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # "en-US" / "zh-CN"
    base = re.split(r"[-_]", key)[0]
    return _LANGUAGE_ALIASES.get(base)


def detect_language(text: str) -> Optional[str]:
    """Best-effort language code detection, ``None`` when undecidable.

    Scripts are detected exactly (CJK, Cyrillic, Arabic, ...).  Latin-script
    languages are separated with function-word frequencies.  Short, ambiguous
    input returns ``None`` rather than a guess.
    """
    if not text or not text.strip():
        return None
    counts = {
        name: len(pattern.findall(text)) for name, pattern in _SCRIPT_PATTERNS.items()
    }
    total = sum(counts.values())
    if total == 0:
        return None
    ratios = {name: value / total for name, value in counts.items()}

    # CJK: kana/hangul take precedence over Han because Japanese mixes all three.
    kana = ratios["hiragana"] + ratios["katakana"]
    if kana >= 0.05:
        return "ja"
    if ratios["hangul"] >= 0.20:
        return "ko"
    if ratios["han"] >= 0.30:
        return "zh"
    for script, language in _SCRIPT_TO_LANGUAGE.items():
        if ratios[script] >= 0.30:
            return language
    if ratios["latin"] < 0.30:
        return None

    tokens = re.findall(r"[^\W\d_]+", text.lower())
    if not tokens:
        return None
    token_set = set(tokens)
    scores: Dict[str, float] = {}
    for language, stopwords in _STOPWORDS.items():
        hits = sum(1 for token in tokens if token in stopwords)
        scores[language] = hits / len(tokens)
    for language, hints in _GREETING_HINTS.items():
        if token_set & hints:
            scores[language] = scores.get(language, 0.0) + 0.5
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.10:
        return None
    if best_score <= runner_up:
        # Ambiguous between Latin-script languages: do not judge.
        return None
    return best_language


# --------------------------------------------------------------------------- #
# Individual instruction checks
# --------------------------------------------------------------------------- #

def _check_keywords_existence(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    keywords = kwargs.get("keywords") or kwargs.get("keyword") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not keywords:
        return None
    haystack = text.lower()
    return all(str(keyword).lower() in haystack for keyword in keywords)


def _check_keywords_frequency(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    keyword = kwargs.get("keyword")
    frequency = _as_float(kwargs.get("frequency"))
    if keyword is None or frequency is None:
        return None
    count = text.lower().count(str(keyword).lower())
    return _relation_ok(count, frequency, kwargs.get("relation"))


def _check_forbidden_words(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    forbidden = kwargs.get("forbidden_words") or kwargs.get("forbidden_word") or []
    if isinstance(forbidden, str):
        forbidden = [forbidden]
    if not forbidden:
        return None
    haystack = text.lower()
    return all(str(word).lower() not in haystack for word in forbidden)


def _check_letter_frequency(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    letter = kwargs.get("letter")
    frequency = _as_float(kwargs.get("let_frequency", kwargs.get("frequency")))
    if letter is None or frequency is None:
        return None
    count = text.lower().count(str(letter)[0].lower())
    relation = kwargs.get("let_relation", kwargs.get("relation"))
    return _relation_ok(count, frequency, relation)


def _check_response_language(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    wanted_raw = kwargs.get("language")
    if wanted_raw is None:
        return None
    wanted = _normalize_language_name(str(wanted_raw))
    if wanted is None:
        return None
    detected = detect_language(text)
    if detected is None:
        return None
    return detected == wanted


def _check_number_words(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    target = _as_float(kwargs.get("num_words"))
    if target is None:
        return None
    return _relation_ok(_word_count(text), target, kwargs.get("relation"))


def _check_number_sentences(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    target = _as_float(kwargs.get("num_sentences"))
    if target is None:
        return None
    return _relation_ok(len(_split_sentences(text)), target, kwargs.get("relation"))


def _check_number_paragraphs(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    target = _as_float(kwargs.get("num_paragraphs"))
    if target is None:
        return None
    return len(_split_paragraphs(text)) == int(target)


def _check_nth_paragraph_first_word(
    text: str, kwargs: Dict[str, Any]
) -> InstructionResult:
    num_paragraphs = _as_float(kwargs.get("num_paragraphs"))
    nth = _as_float(kwargs.get("nth_paragraph"))
    first_word = kwargs.get("first_word")
    if num_paragraphs is None or nth is None or first_word is None:
        return None
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) != int(num_paragraphs):
        return False
    index = int(nth) - 1
    if index < 0 or index >= len(paragraphs):
        return False
    return _first_word(paragraphs[index]).lower() == str(first_word).strip().lower()


def _check_number_bullet_lists(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    target = _as_float(kwargs.get("num_bullets"))
    if target is None:
        return None
    return _count_bullet_lines(text) == int(target)


def _check_number_highlighted_sections(
    text: str, kwargs: Dict[str, Any]
) -> InstructionResult:
    target = _as_float(kwargs.get("num_highlights"))
    if target is None:
        return None
    return _count_highlights(text) == int(target)


def _check_multiple_sections(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    splitter = kwargs.get("section_spliter") or kwargs.get("section_splitter")
    target = _as_float(kwargs.get("num_sections"))
    if not splitter or target is None:
        return None
    occurrences = text.count(str(splitter))
    return occurrences >= int(target)


def _check_english_capital(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    letters = re.findall(r"[A-Za-z]", text or "")
    if not letters:
        return False
    return all(letter.isupper() for letter in letters)


def _check_english_lowercase(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    letters = re.findall(r"[A-Za-z]", text or "")
    if not letters:
        return False
    return all(letter.islower() for letter in letters)


def _check_end_checker(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    end_phrase = kwargs.get("end_phrase")
    if end_phrase is None:
        return None
    return text.rstrip().endswith(str(end_phrase))


def _check_no_comma(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    return "," not in (text or "")


def _check_two_responses(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    parts = (text or "").split("******")
    if len(parts) != 2:
        return False
    return parts[0].strip() != "" and parts[1].strip() != ""


def _check_repeat_prompt(text: str, kwargs: Dict[str, Any]) -> InstructionResult:
    prompt = kwargs.get("prompt_to_repeat")
    if prompt is None:
        return None
    response_norm = " ".join(str(text or "").split())
    prompt_norm = " ".join(str(prompt).split())
    if not prompt_norm:
        return None
    # Tolerate a markdown blockquote / bullet prefix before the repeated prompt.
    response_norm = response_norm.lstrip("> -*•").lstrip()
    return response_norm.lower().startswith(prompt_norm.lower())


_HANDLERS: Dict[str, InstructionChecker] = {
    "keywords:existence": _check_keywords_existence,
    "keywords:frequency": _check_keywords_frequency,
    "keywords:forbidden_words": _check_forbidden_words,
    "keywords:letter_frequency": _check_letter_frequency,
    "language:response_language": _check_response_language,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _check_nth_paragraph_first_word,
    "detectable_format:number_bullet_lists": _check_number_bullet_lists,
    "detectable_format:number_highlighted_sections": _check_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "change_case:english_capital": _check_english_capital,
    "change_case:english_lowercase": _check_english_lowercase,
    "startend:end_checker": _check_end_checker,
    "punctuation:no_comma": _check_no_comma,
    "combination:two_responses": _check_two_responses,
    "combination:repeat_prompt": _check_repeat_prompt,
}


def supported_instructions() -> List[str]:
    """Return the sorted list of implemented IFEval instruction ids."""
    return sorted(_HANDLERS)


def check_instructions(
    response_text: str,
    instruction_id_list: Sequence[str],
    kwargs: Sequence[Dict[str, Any]],
) -> List[InstructionResult]:
    """Check every instruction; returns a list aligned with ``instruction_id_list``.

    ``None`` means "cannot judge" (unknown instruction id, missing/unknown
    kwargs or an internal error); such entries are excluded from the IFEval
    denominator by :func:`score_instructions`.
    """
    text = response_text if isinstance(response_text, str) else ""
    ids = list(instruction_id_list or [])
    kwargs_list = list(kwargs or [])
    results: List[InstructionResult] = []
    for index, instruction_id in enumerate(ids):
        raw_kwargs = kwargs_list[index] if index < len(kwargs_list) else None
        instruction_kwargs = raw_kwargs if isinstance(raw_kwargs, dict) else {}
        if not isinstance(instruction_id, str):
            results.append(None)
            continue
        handler = _HANDLERS.get(instruction_id.strip())
        if handler is None:
            results.append(None)
            continue
        try:
            outcome = handler(text, instruction_kwargs)
        except Exception:  # pragma: no cover - defensive: malformed kwargs
            outcome = None
        results.append(None if outcome is None else bool(outcome))
    return results


def _extract_instruction_specs(sample: Sample) -> tuple:
    """Normalise metadata variants into (ids, kwargs_list)."""
    metadata = sample.metadata or {}
    ids = metadata.get("instruction_id_list") or metadata.get("instruction_ids") or []
    kwargs_list = metadata.get("kwargs") or metadata.get("instruction_kwargs") or []
    if isinstance(ids, dict):  # tolerate {id: kwargs} form
        kwargs_list = list(ids.values())
        ids = list(ids.keys())
    if isinstance(kwargs_list, dict):
        kwargs_list = [kwargs_list.get(instruction_id, {}) for instruction_id in ids]
    # Tolerate the list-of-dicts form: [{"instruction_id": ..., "kwargs": {...}}, ...]
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], dict):
        specs = [item for item in ids if isinstance(item, dict)]
        kwargs_list = [item.get("kwargs") or {} for item in specs]
        ids = [item.get("instruction_id") or item.get("id") for item in specs]
    return list(ids or []), list(kwargs_list or [])


def score_instructions(sample: Sample, response: RawResponse) -> Score:
    """Score an IFEval sample in the strict (instruction-level) convention.

    ``correct`` is True only when **all judgeable instructions pass** and at
    least one instruction is judgeable; ``score`` is the pass ratio over the
    judgeable subset (partial credit).  ``detail`` keeps the per-instruction
    result, the pass/judge counts and how many ids were unimplemented.
    """
    try:
        text = (response.text or "") if response is not None else ""
        ids, kwargs_list = _extract_instruction_specs(sample)
        results = check_instructions(text, ids, kwargs_list)
        judged = [result for result in results if result is not None]
        passed = sum(1 for result in judged if result)
        detail_results = [
            {
                "instruction_id": ids[index] if index < len(ids) else None,
                "kwargs": kwargs_list[index] if index < len(kwargs_list) else {},
                "passed": result,
                "judged": result is not None,
            }
            for index, result in enumerate(results)
        ]
        if not judged:
            return Score(
                correct=False,
                score=0.0,
                detail={
                    "error": "no judgeable instructions",
                    "n_instructions": len(ids),
                    "n_unknown": len(results),
                    "results": detail_results,
                    "first_200_chars": text[:200],
                },
            )
        correct = passed == len(judged)
        return Score(
            correct=correct,
            score=passed / len(judged),
            detail={
                "n_passed": passed,
                "n_judged": len(judged),
                "n_unknown": len(results) - len(judged),
                "n_instructions": len(ids),
                "results": detail_results,
                "first_200_chars": text[:200],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
