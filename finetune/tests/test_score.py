from finetune.bakeoff.score import extract_answer, is_correct


def test_extracts_last_number():
    assert extract_answer("First 5, then 12. The answer is 18.") == "18"


def test_strips_commas_and_currency():
    assert extract_answer("Total cost is $1,250") == "1250"


def test_handles_negative():
    assert extract_answer("The change is -42") == "-42"


def test_handles_decimal():
    assert extract_answer("It comes to 3.5 hours") == "3.5"


def test_returns_none_when_no_number():
    assert extract_answer("I am not sure about this one.") is None


def test_is_correct_parses_gsm8k_gold_marker():
    gold = "She had 5 apples and ate 2.\n#### 3"
    assert is_correct("So the answer is 3", gold) is True


def test_is_correct_rejects_wrong_answer():
    gold = "#### 3"
    assert is_correct("The answer is 4", gold) is False


def test_is_correct_false_when_no_prediction_number():
    assert is_correct("I don't know", "#### 3") is False


def test_is_correct_ignores_thousands_separator_mismatch():
    assert is_correct("The answer is 1,250", "#### 1250") is True
