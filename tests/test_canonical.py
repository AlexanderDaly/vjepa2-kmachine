from kabbalistic_machine import KabbalisticMachine


def test_canonical():
    km = KabbalisticMachine(3, 4, lambda x: 0)
    km.analyse()
    rep = km._canonical((0,1,2,0), km._all_letter_perms())
    assert rep == (0,0,1,2)
