"""
A chord is a tuple (T, Q) where:
- T is a tonic note: T \in Z_12
- Q is a chord pattern:

Example: chord = (0, MAJ_PATTERN) # C maj
"""

import pretty_midi, math
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# chord patterns
MAJ_PATTERN = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1] # maj7
MIN_PATTERN = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0] # min7
DOM_PATTERN = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0] # dom7
DIM_PATTERN = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1] # half-dim
# possible notes
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# turn a chord tuple into a string (helper function)
def _chord_tuple_to_name(chord):
    note = NOTES[chord[0] % 12]

    q = chord[1]
    quality = None
    if q == MAJ_PATTERN: quality = "maj"
    elif q == MIN_PATTERN: quality = "min"
    elif q == DOM_PATTERN: quality = "dom"
    elif q == DIM_PATTERN: quality = "dim"

    return f"{note}{quality}"

# return all 48 possible chords (helper function)
def _all_possible_chords():
    res = []
    for i in range(12):
        res.append((i, MAJ_PATTERN))
        res.append((i, MIN_PATTERN))
        res.append((i, DOM_PATTERN))
        res.append((i, DIM_PATTERN))
    return res

# normalize 12xN chroma array (helper function)
def _normalize(chroma): 
    # sum at each time step (i dont like numpy)
    chroma_sums = chroma.sum(axis=0, keepdims=True)
    # make an empty copy of chroma
    res = np.zeros_like(chroma)
    # for each time step...
    for i in range(chroma.shape[1]):
        # i-th time step sum
        chroma_sum = chroma_sums[0, i]
        # check if == 0
        if chroma_sum == 0: res[:, i] = 0
        else: res[:, i] = chroma[:, i] / chroma_sum
    else: return res

# return normalized 12xN chroma array of a midi file
def chroma(file, fs=2.0): return _normalize(pretty_midi.PrettyMIDI(file).get_chroma(fs=fs))

# use a hidden markov model to predict key
def hmm_based_chords_from_chromas(chromas, fs=2.0, var=0.05):
    # gaussian HMM for cont. data over 48 possible chords
    model = GaussianHMM(n_components = 48, n_iter = 100, covariance_type="full")

    # initialize the transition matrix
    transmat  = np.ones((48, 48)) / 48.0
    model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    
    states = _all_possible_chords()

    # initialize the key profiles
    profiles = []
    # roll the chord patterns to match tonic
    for i in states: profiles.append(np.roll(i[1], i[0]))
    # normalize profiles and set means to the profiles
    profiles = np.array(profiles)
    model.means_ = profiles/profiles.sum(axis=1, keepdims=True)

    # all chords start equally likely
    model.startprob_ = np.ones(48) / 48.0

    # set the covariance matrix (small variance around chords)
    model.covars_ = np.tile(np.identity(12)*var, (48, 1, 1))

    # make the predictions
    return (model.predict(chromas.T))

def chord_seq(hmm_predictions):
    chords = _all_possible_chords()
    return [_chord_tuple_to_name(chords[i]) for i in hmm_predictions]

# example

file = "./midi/II.mid"
chroma_data = chroma(file)
hmm_out = hmm_based_chords_from_chromas(chroma_data)

plt.plot(hmm_out, label="Predicted Chord Sequence of (Arpeggio)")

file = "./midi/I.mid"
chroma_data = chroma(file)
hmm_out = hmm_based_chords_from_chromas(chroma_data)

plt.plot(hmm_out, label="Predicted Chord Sequence (Chords)")

plt.xlabel("Time Slice")
plt.ylabel("Chord")
yt = [i for i in range(48)]
plt.yticks(yt, [_chord_tuple_to_name(i) for i in _all_possible_chords()])
plt.title("Arpeggiated Comparison (variance = .05)")
plt.legend()
plt.show()

