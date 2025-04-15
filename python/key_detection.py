# marc browning
# detects key signature from chord data

import chord_detection as cd
import pretty_midi, math
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# generates every possible major or minor key
def _keys(notes=None, qualities=None):
    res = []
    if notes == None: notes = [i for i in range(12)]
    if qualities == None: qualities = [0, 1]
    for j in qualities:
        for i in notes: res.append((i, j))
    return res

def _key_tuple_to_name(key, notes=None, qualities=None):
    if notes == None: notes = cd.NOTES
    if qualities == None: qualities = ["Maj", "Min"]
    tonic = notes[key[0]]
    quality = qualities[key[1]]
    return f"{tonic} {quality}"

print(_key_tuple_to_name((1, 1)))
