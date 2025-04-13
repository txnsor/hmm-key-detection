# Marc Browning
# IGME-589 Final Project
# Proof of Concept: HMM (Pre-trained) MIDI Key Interpreter
# Based off of https://gist.github.com/Astroneko404/1dcde11576e510e964882bbbafaeb050

### This approach can be similarly done with chords as the hidden variables, but
### for brevity sake, as well as since this is a proof of concept, that is omitted.

# this library is awesome
import pretty_midi as pm
# numpy :/
import numpy as np
# gaussian because continuous data
from hmmlearn.hmm import GaussianHMM

# i could have just read the original paper but thank you Astroneko404 for the preformatted weights and mode table
# thank you :DD

MODE_TABLE_KHS = [
    "C Major", "C Minor", "C# Major", "C# Minor", "D Major", "D Minor", "D# Major", "D# Minor", "E Major",
    "E Minor", "F Major", "F Minor", "F# Major", "F# Minor", "G Major", "G Minor", "G# Major", "G# Minor",
    "A Major", "A Minor", "A# Major", "A# Minor", "B Major", "B Minor"
]

# nevermind, i had to use a seperate mode table for HMM labelling

MODE_TABLE_HMM = [
    "C major", "C# major", "D major", "D# major", "E major", "F major",
    "F# major", "G major", "G# major", "A major", "A# major", "B major",
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor"
]

KRUMHANSL_SCHMUCKLER_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_SCHMUCKLER_MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PRESET_KEY_VARIANCE_MAJ = np.array([0.01, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.01, 0.1, 0.1, 0.1, 0.05])
PRESET_KEY_VARIANCE_MIN = np.roll(PRESET_KEY_VARIANCE_MAJ, -2)

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

def key_probability_vector(chroma, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR):
    res = []
    # major keys
    # weigh the data set and use pearson correlation
    for i in range(12): res.append(np.corrcoef(chroma, MAJOR_PROFILE)[0, 1])
    # minor keys
    for i in range(12): res.append(np.corrcoef(chroma, MINOR_PROFILE)[0, 1])
    return res # likelyhood vector

def hmm_find_key(file, fs=2.0, kp_major=KRUMHANSL_SCHMUCKLER_MAJOR, kp_minor=KRUMHANSL_SCHMUCKLER_MINOR):
    # init the model
    model = GaussianHMM(n_components=len(MODE_TABLE_HMM), n_iter=100)
    # initially, each key is equally likely
    model.transmat_ = np.full((24, 24), 1.0/24.0)
    # init key profiles
    profiles = []
    # shift the tonic for each profile
    for i in range(12): profiles.append(np.roll(kp_major, i))
    for i in range(12): profiles.append(np.roll(kp_minor, i))
    # normalize
    profiles = np.array(profiles)
    profiles = profiles/profiles.sum(axis=1, keepdims=True)
    model.means_ = profiles

    # set up covars
    # model.covars_ = np.vstack([
    #     [np.roll(PRESET_KEY_VARIANCE_MAJ, i) for i in range(12)],
    #     [np.roll(PRESET_KEY_VARIANCE_MIN, i) for i in range(12)]
    #     ])
    model.covars_ = np.ones((24, 12)) * 0.01
    # set up start probabilities
    model.startprob_ = np.ones(len(MODE_TABLE_HMM)) / len(MODE_TABLE_HMM)
    
    # get keys!!
    chroma = normalize(pm.PrettyMIDI(file).get_chroma(fs=fs)).T
    res = model.predict(chroma)
    return res
    

# compares to a key profile
def naive_profile_similarity(file, fs=2.0, kp_major=KRUMHANSL_SCHMUCKLER_MAJOR, kp_minor=KRUMHANSL_SCHMUCKLER_MINOR):
    res = {}
    # get chroma
    chroma = normalize(pm.PrettyMIDI(file).get_chroma(fs=fs))
    # init res (time : key pairs)
    for i in range(chroma.shape[0]): res.update({i/fs : MODE_TABLE_KHS[np.argmax(key_probability_vector(chroma[:, i], kp_major, kp_minor))]})
    return res

file = "./midi/simple.mid"
# print(naive_profile_similarity(file))
res = hmm_find_key(file)
print(res)