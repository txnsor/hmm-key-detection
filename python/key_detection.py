# marc browning
# detects key signature from chord data

import chord_detection as cd
import pretty_midi, math
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# key profiles
def_maj = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
def_min = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

KRUMHANSL_SCHMUCKLER_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_SCHMUCKLER_MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


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

# get rid of chroma slices where the chord doesnt change, "flattening" the data
def _flatten_chord_changes(chords):
    print(chords)
    res = []
    prev = None
    for chord in chords:
        if prev != chord:
            res.append(chord)
            prev = chord
    return res

def hmm_based_keys_from_chords(chords, var=0.05, major=def_maj, minor=def_min):
    
    # all possible states and # of states
    states = _keys()
    n = len(states)


    # gaussian HMM for cont. data over 24 possible keys
    model = GaussianHMM(n_components = n, n_iter = 100, covariance_type="full")

    # initialize the transition matrix
    transmat  = np.ones((n, n)) / float(n)
    model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    

    # initialize the key profiles
    profiles = []
    # roll the chord patterns to match tonic

    # TODO: reimpl for modes at some point
    for i in range(12): profiles.append(np.roll(major, i))
    for i in range(12): profiles.append(np.roll(minor, i))

    # normalize profiles and set means to the profiles
    profiles = np.array(profiles)
    model.means_ = profiles/profiles.sum(axis=1, keepdims=True)

    # all chords start equally likely
    model.startprob_ = np.ones(n) / float(n)

    # set the covariance matrix (small variance around keys)
    model.covars_ = np.tile(np.identity(12)*var, (n, 1, 1))

    # make the predictions

    # get the observations in a better form
    o = []
    for chord in chords: 
        c = cd._all_possible_chords()[chord]
        o.append(np.roll(c[1], c[0]))
    o = np.array(o)

    return (model.predict(o))



res = hmm_based_keys_from_chords(cd.hmm_based_chords_from_chromas(cd.chroma("./midi/VI.mid")), major=KRUMHANSL_SCHMUCKLER_MAJOR, minor=KRUMHANSL_SCHMUCKLER_MINOR)
keys = [_key_tuple_to_name(_keys()[j]) for j in res]

plt.xlabel("Time")
plt.ylabel("Key")
plt.plot(keys, label="test")
yt = [i for i in range(24)]
plt.yticks(yt, [_key_tuple_to_name(_keys()[j]) for j in range(24)])
plt.title("Test")
plt.legend()
plt.show()


