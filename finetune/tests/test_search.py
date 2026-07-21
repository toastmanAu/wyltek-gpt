import pytest

from finetune.bakeoff.search import find_max_seqlen


def test_finds_exact_threshold():
    # everything up to and including 4096 fits
    calls = []

    def fits(n):
        calls.append(n)
        return n <= 4096

    assert find_max_seqlen(fits, lo=512, hi=16384, step=512) == 4096


def test_returns_zero_when_nothing_fits():
    assert find_max_seqlen(lambda n: False, lo=512, hi=8192, step=512) == 0


def test_returns_hi_when_everything_fits():
    assert find_max_seqlen(lambda n: True, lo=512, hi=8192, step=512) == 8192


def test_is_logarithmic_not_linear():
    calls = []

    def fits(n):
        calls.append(n)
        return n <= 4096

    find_max_seqlen(fits, lo=512, hi=16384, step=512)
    # 32 candidate steps -> must probe far fewer than 32 times
    assert len(calls) <= 8, f"too many probes: {calls}"


def test_result_is_multiple_of_step():
    result = find_max_seqlen(lambda n: n <= 5000, lo=512, hi=16384, step=512)
    assert result % 512 == 0
    assert result == 4608


def test_never_probes_same_value_twice_when_lo_equals_hi():
    calls = []

    def fits(n):
        calls.append(n)
        return True

    find_max_seqlen(fits, lo=512, hi=512, step=512)
    assert len(calls) == len(set(calls)), f"duplicate probes: {calls}"


def test_never_probes_same_value_twice_on_narrow_range():
    calls = []

    def fits(n):
        calls.append(n)
        return True

    find_max_seqlen(fits, lo=512, hi=1024, step=512)
    assert len(calls) == len(set(calls)), f"duplicate probes: {calls}"


def test_never_probes_same_value_twice_with_threshold():
    calls = []

    def fits(n):
        calls.append(n)
        return n <= 600

    find_max_seqlen(fits, lo=512, hi=16384, step=512)
    assert len(calls) == len(set(calls)), f"duplicate probes: {calls}"


def test_non_aligned_lo_raises_value_error():
    with pytest.raises(ValueError):
        find_max_seqlen(lambda n: True, lo=600, hi=16384, step=512)
