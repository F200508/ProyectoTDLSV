from tdlsv.inference.text_builder import TextBuilder


def _feed(builder, letter, confidence, times):
    confirmed = None
    for _ in range(times):
        confirmed = builder.update(letter, confidence)
    return confirmed


def test_confirms_letter_after_stability_frames():
    tb = TextBuilder(stability_frames=5, confidence_threshold=0.8)
    confirmed = _feed(tb, "A", 0.9, times=4)
    assert confirmed is None
    assert tb.text == ""

    confirmed = _feed(tb, "A", 0.9, times=1)
    assert confirmed == "A"
    assert tb.text == "A"


def test_low_confidence_does_not_confirm():
    tb = TextBuilder(stability_frames=3, confidence_threshold=0.8)
    _feed(tb, "A", 0.5, times=10)
    assert tb.text == ""


def test_no_hand_resets_stability():
    tb = TextBuilder(stability_frames=5, confidence_threshold=0.8)
    _feed(tb, "A", 0.9, times=4)
    tb.update(None, 0.0)
    confirmed = _feed(tb, "A", 0.9, times=4)
    assert confirmed is None
    confirmed = tb.update("A", 0.9)
    assert confirmed == "A"


def test_same_letter_does_not_repeat_until_hand_changes():
    tb = TextBuilder(stability_frames=3, confidence_threshold=0.8)
    _feed(tb, "A", 0.9, times=3)
    assert tb.text == "A"

    confirmed = _feed(tb, "A", 0.9, times=10)
    assert confirmed is None
    assert tb.text == "A"

    tb.update(None, 0.0)
    confirmed = _feed(tb, "A", 0.9, times=3)
    assert confirmed == "A"
    assert tb.text == "AA"


def test_different_letter_confirms_independently():
    tb = TextBuilder(stability_frames=3, confidence_threshold=0.8)
    _feed(tb, "A", 0.9, times=3)
    _feed(tb, "B", 0.9, times=3)
    assert tb.text == "AB"


def test_add_space_backspace_clear():
    tb = TextBuilder(stability_frames=3, confidence_threshold=0.8)
    _feed(tb, "A", 0.9, times=3)
    tb.add_space()
    assert tb.text == "A "
    tb.backspace()
    assert tb.text == "A"
    tb.clear()
    assert tb.text == ""
