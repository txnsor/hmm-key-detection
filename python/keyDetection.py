# this library is awesome
import pretty_midi as pm
# numpy :/
import numpy as np
# gaussian because continuous data
from hmmlearn.hmm import GaussianHMM
# matplotlib for data visuals
import matplotlib.pyplot as plt

# both methods use different 

MODE_TABLE = [
    "C major", "C# major", "D major", "D# major", "E major", "F major", "F# major", "G major", "G# major", 
    "A major", "A# major", "B major", "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor"
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

# use a given set of weights to predict key for a chroma slice
def key_probability_vector(chroma, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR):
    res = []
    # major keys
    # weigh the data set and use pearson correlation
    for i in range(12): res.append(np.corrcoef(chroma, np.roll(MAJOR_PROFILE, i))[0, 1])
    # minor keys
    for i in range(12): res.append(np.corrcoef(chroma, np.roll(MINOR_PROFILE, i))[0, 1])
    return np.array(res) # likelyhood vector

# predict key using weights for an array of chroma slices
def weight_based_key(file, fs=2.0, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR):
    res = []
    chroma = normalize(pm.PrettyMIDI(file).get_chroma(fs=fs))
    for i in range(chroma.shape[1]): res.append(np.argmax(key_probability_vector(chroma[:, i], MAJOR_PROFILE, MINOR_PROFILE)))
    return res

# use a hidden markov model to predict key
def hmm_based_key_from_chromas(chromas, fs=2.0, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR, covars=np.ones((24, 12)) * 0.15):
    # gaussian HMM for cont. data over 24 keys
    model = GaussianHMM(n_components = 24, n_iter = 100)
    # initialize the transition matrix
    # high self transition probability (to minimize modulation)
    # normalized
    transmat  = np.ones((24, 24)) / 24.0
    model.transmat_ = transmat / transmat.sum(axis = 1, keepdims = True)
    # initialize the key profiles
    profiles = []
    for i in range(12): profiles.append(np.roll(MAJOR_PROFILE, i))
    for i in range(12): profiles.append(np.roll(MINOR_PROFILE, i))
    # normalize profiles and set means to the profiles
    profiles = np.array(profiles)
    model.means_ = profiles/profiles.sum(axis=1, keepdims=True)
    # set up the covariance matrix
    # NOTE: this can vary, depending of level of key variance, as well as stability of notes in a key
    model.covars_ = covars
    # all keys start equally likely
    model.startprob_ = np.ones((24)) / 24.0
    # make the predictions
    return model.predict(chromas.T)

def hmm_based_key(file, fs=2.0, MAJOR_PROFILE=KRUMHANSL_SCHMUCKLER_MAJOR, MINOR_PROFILE=KRUMHANSL_SCHMUCKLER_MINOR, covars=(np.ones((24, 12)) * 0.15)):
    chromas = normalize(pm.PrettyMIDI(file).get_chroma(fs=fs))
    return hmm_based_key_from_chromas(chromas, fs, MAJOR_PROFILE, MINOR_PROFILE, covars)

# TEST 1: BASIC COVARS ON V-I

simple = "./midi/simple.mid"
res_khs = weight_based_key(simple)

# covars generated from KHS, squared logarithmic
covars = []
for i in range(12): covars.append(np.roll(KRUMHANSL_SCHMUCKLER_MAJOR, -i))
for i in range(12): covars.append(np.roll(KRUMHANSL_SCHMUCKLER_MINOR, -i))
covars = np.log(np.array(covars))**2
covars /= np.max(covars)

res_hmm = hmm_based_key(simple, covars=covars)

print([MODE_TABLE[i] for i in res_khs])
print([MODE_TABLE[i] for i in res_hmm])

plt.plot(res_khs, label = "KHS Predicted")
plt.plot(res_hmm, label = "HMM Predicted")
plt.plot([0 for i in range(len(res_khs))], label = "Theoretical")
plt.xlabel("Time Slice")
plt.ylabel("Key")
yt = [i for i in range(24)]
plt.yticks(yt, MODE_TABLE)
plt.legend()
plt.title("Key Changes in simple.mid")
plt.show()
