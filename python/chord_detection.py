"""
A chord is a tuple (T, Q) where:
- T is a tonic note: T \in Z_12
- Q is a chord pattern:

Example: chord = (0, MAJ_PATTERN) # C maj
"""

# marc browning
# detects chords from MIDI over time slices

import pretty_midi, math
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# possible notes
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# generate chord patterns given a strength tuple where:
"""
S = (s0, s1, s2, s3, s4):
    s0 is the weight given to the root note
    s1 is the weight given to the third
    s2 is the weight given to the fifth
    s3 is the weight given to the seventh
    s4 is the weight on every other note
"""
# returns (maj, min, dom, dim) according to strength values
# (helper function)
def _generate_chord_patterns(strength_values=None):
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)
    maj = [sv[0], sv[4], sv[4], sv[4], sv[1], sv[4], sv[4], sv[2], sv[4], sv[4], sv[4], sv[3]] # maj7
    min = [sv[0], sv[4], sv[4], sv[1], sv[4], sv[4], sv[4], sv[2], sv[4], sv[4], sv[3], sv[4]] # min7
    dom = [sv[0], sv[4], sv[4], sv[4], sv[1], sv[4], sv[4], sv[2], sv[4], sv[4], sv[3], sv[4]] # dom7
    dim = [sv[0], sv[4], sv[4], sv[1], sv[4], sv[4], sv[2], sv[4], sv[4], sv[4], sv[4], sv[3]] # half-dim
    return (maj, min, dom, dim)

# turn a chord tuple into a string (helper function)
def _chord_tuple_to_name(chord, strength_values=None):
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)

    note = NOTES[chord[0] % 12]
    patterns = _generate_chord_patterns(strength_values=sv)

    q = chord[1]
    quality = None
    if q == patterns[0]: quality = "maj"
    elif q == patterns[1]: quality = "min"
    elif q == patterns[2]: quality = "dom"
    elif q == patterns[3]: quality = "dim"

    return f"{note}{quality}"

# return all 48 possible chords (helper function)
def _all_possible_chords(strength_values=None):
    res = []
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)
    p = _generate_chord_patterns(strength_values=sv)
    for i in range(12):
        res.append((i, p[0]))
        res.append((i, p[1]))
        res.append((i, p[2]))
        res.append((i, p[3]))
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
def hmm_based_chords_from_chromas(chromas, fs=2.0, var=0.05, strength_values=None):
    # init chord patterns
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)

    # all possible states and # of states
    states = _all_possible_chords(strength_values=sv)
    n = len(states)

    # gaussian HMM for cont. data over 48 possible chords
    model = GaussianHMM(n_components = n, n_iter = 100, covariance_type="full")

    # initialize the transition matrix
    transmat  = np.ones((n, n)) / float(n)
    model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    

    # initialize the key profiles
    profiles = []
    # roll the chord patterns to match tonic
    for i in states: profiles.append(np.roll(i[1], i[0]))
    # normalize profiles and set means to the profiles
    profiles = np.array(profiles)
    model.means_ = profiles/profiles.sum(axis=1, keepdims=True)

    # all chords start equally likely
    model.startprob_ = np.ones(n) / float(n)

    # set the covariance matrix (small variance around chords)
    model.covars_ = np.tile(np.identity(12)*var, (n, 1, 1))

    # make the predictions
    return (model.predict(chromas.T))

# turn hmm observations into a chord sequence
def chord_seq(hmm_predictions, strength_values=None):
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)
    chords = _all_possible_chords(strength_values=sv)
    return [_chord_tuple_to_name(chords[i], strength_values=sv) for i in hmm_predictions]


# plot predicted key with preset values and label
def plot_file(file, label, fs=2.0, var=0.05, strength_values=None):
    # i dont need all of this handling, i know
    sv = strength_values
    if sv == None: sv = (1.0, 1.0, 1.0, 1.0, 0.0)
    chroma_data = chroma(file, fs=fs)
    hmm_out = hmm_based_chords_from_chromas(chroma_data, fs=fs, var=var, strength_values=sv)
    plt.plot(hmm_out, label=label)

def main():
    file = "./midi/simple.mid"

    print(hmm_based_chords_from_chromas(chroma(file)))

    plot_file(file, "Predicted Chords (On/Off Heuristic, var=0.05)")
    # plot_file(file, "Predicted Chords (V->I Heuristic)", strength_values=(1.0, 0.333, 0.666, 0.333, 0.0))
    # plot_file(file, "Predicted Chords (Guide Tone Heuristic)", strength_values=(0.666, 1.0, 0.333, 1.0, 0.0))
    # plot_file(file, "Predicted Chords (Loose Heuristic)", strength_values=(1.0, 0.7, 0.4, 0.8, 0.1))

    # plt.plot(hmm_out, label="Predicted Chord Sequence (Chords)")

    plt.xlabel("Time Slice")
    plt.ylabel("Chord")
    yt = [i for i in range(48)]
    plt.yticks(yt, [_chord_tuple_to_name(i) for i in _all_possible_chords()])
    plt.title("Arpeggiated V-I")
    plt.legend()
    plt.show()

if __name__ == "__main__": main()

