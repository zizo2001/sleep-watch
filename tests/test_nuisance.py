from sleepwatch.nuisance import (DAMP_FACTOR, DAMP_THRESHOLD, NuisanceMap)

BOX = (0.5, 0.5, 0.1, 0.1)          # small box, mid-frame
FAR = (0.0, 0.0, 0.05, 0.05)


def test_hits_accumulate_and_damp_kicks_in(tmp_path):
    m = NuisanceMap(tmp_path / "map.json")
    assert m.damp_factor(BOX) == 1.0
    for _ in range(int(DAMP_THRESHOLD)):
        m.hit(BOX)
    assert m.damp_factor(BOX) == DAMP_FACTOR
    assert m.damp_factor(FAR) == 1.0          # other regions untouched


def test_decay_prunes_and_unlearns(tmp_path):
    m = NuisanceMap(tmp_path / "map.json")
    for _ in range(3):
        m.hit(BOX)
    for _ in range(20):                        # many quiet nights: 3×0.9^20<0.5
        m.decay()
    assert m.damp_factor(BOX) == 1.0
    assert m.cells == {}                       # fully pruned


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "map.json"
    m = NuisanceMap(p)
    m.hit(BOX)
    m.hit(BOX)
    m.save()
    m2 = NuisanceMap(p)
    m2.hit(BOX)
    assert m2.damp_factor(BOX) == DAMP_FACTOR  # 3 hits across two lives


def test_corrupt_file_starts_fresh(tmp_path):
    p = tmp_path / "map.json"
    p.write_text("{not json")
    m = NuisanceMap(p)
    assert m.cells == {}
