# Marc Browning
# IGME-589 Final Project
# Proof of Concept: HMM (Pre-trained) MIDI Key and Chord Interpreter
# Based off of https://gist.github.com/Astroneko404/1dcde11576e510e964882bbbafaeb050

# this library is awesome
import pretty_midi as pm
# numpy :/
import numpy as np

# i could have just read the original paper but thank you Astroneko404 for the preformatted weights and mode table
# thank you :DD

MODE_TABLE = [
    "C Major", "C Minor", "C# Major", "C# Minor", "D Major", "D Minor", "D# Major", "D# Minor", "E Major",
    "E Minor", "F Major", "F Minor", "F# Major", "F# Minor", "G Major", "G Minor", "G# Major", "G# Minor",
    "A Major", "A Minor", "A# Major", "A# Minor", "B Major", "B Minor"
]

KRUMHANSL_SCHMUCKLER_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_SCHMUCKLER_MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# normalize 12xN chroma array
def normalize(chroma): 
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

def key_probability(chroma, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR):
    res = []
    # major keys
    # weigh the data set and use pearson correlation
    for i in range(12): res.append(np.corrcoef(chroma, MAJOR_PROFILE)[0, 1])
    # minor keys
    for i in range(12): res.append(np.corrcoef(chroma, MINOR_PROFILE)[0, 1])
    return res # likelyhood vector

file = "./midi/VI.mid"
chroma = normalize(pm.PrettyMIDI(file).get_chroma(fs=2))
for i in range(chroma.shape[0]): print(f"Time {i/2.0:08f}s, KK-Vector: {np.max(key_probability(chroma[:, i]))}")