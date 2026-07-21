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
