#

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

BELLMAN_BUDGE_MAJOR = [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.8, 8.04, 0.62, 10.57]
BELLMAN_BUDGE_MINOR = [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]

CUSTOM_PROFILE_SIMPLE_MAJ = [1.0, 0.1, 0.5, 0.1, 0.7, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.7]
CUSTOM_PROFILE_SIMPLE_MIN = [1.0, 0.1, 0.5, 0.7, 0.1, 0.5, 0.1, 0.9, 0.5, 0.1, 0.7, 0.1]

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

def map_min_maj(a):
    if a < 12: return a
    else: return (a - 21) % 12

simple = "./midi/VI.mid"
res_khs = weight_based_key(simple, fs=1.0)

# covars generated from KHS, squared logarithmic
covars = []
for i in range(12): covars.append(np.roll(KRUMHANSL_SCHMUCKLER_MAJOR, i))
for i in range(12): covars.append(np.roll(KRUMHANSL_SCHMUCKLER_MINOR, i))
covars = 1/(np.array(covars))
covars /= np.max(covars)

covars2 = []
for i in range(12): covars2.append(np.roll(BELLMAN_BUDGE_MAJOR, i))
for i in range(12): covars2.append(np.roll(BELLMAN_BUDGE_MINOR, i))
covars2 = 1/(np.array(covars2))
covars2 /= np.max(covars2)

custom_covars = []
for i in range(12): custom_covars.append(np.roll(CUSTOM_PROFILE_SIMPLE_MAJ, i))
for i in range(12): custom_covars.append(np.roll(CUSTOM_PROFILE_SIMPLE_MIN, i))
custom_covars /= np.max(custom_covars)

res_hmm = hmm_based_key(simple, covars=covars**2, fs=1.0)
res_hmm_2 = hmm_based_key(simple, covars=custom_covars, fs=1.0)
res_hmm_3 = hmm_based_key(simple, MAJOR_PROFILE=CUSTOM_PROFILE_SIMPLE_MAJ, MINOR_PROFILE=CUSTOM_PROFILE_SIMPLE_MIN, covars=custom_covars, fs=1.0)
res_hmm_4 = hmm_based_key(simple, MAJOR_PROFILE=BELLMAN_BUDGE_MAJOR, MINOR_PROFILE=BELLMAN_BUDGE_MINOR, covars=custom_covars, fs=1.0)

# plt.plot([map_min_maj(i) for i in res_khs] , label = "KHS Predicted", linestyle="dotted")
# # plt.plot([map_min_maj(i) for i in res_hmm], label = "HMM Predicted (Inv. Squared KHS Covariance + KHS Weighting)")
# plt.plot([map_min_maj(i) for i in res_hmm_2], label = "HMM Predicted (Custom Covariance + KHS Weighting)")
# plt.plot([map_min_maj(i) for i in res_hmm_3], label = "HMM Predicted (Custom Covariance + Custom Weighting)")
# # plt.plot([0 for i in range(len(res_khs))], label = "Theoretical")
# plt.xlabel("Time Slice")
# plt.ylabel("Key")
# yt = [i for i in range(12)]
# plt.yticks(yt, [
#     "{} / {}".format(MODE_TABLE[i], MODE_TABLE[(i+9)%12 + 12]) for i in range(12)
# ])
# plt.legend()
# plt.title("Key Changes in full.mid")
# plt.show()

plt.plot(res_khs, label = "KHS Predicted", linestyle="dotted")
# plt.plot([map_min_maj(i) for i in res_hmm], label = "HMM Predicted (Inv. Squared KHS Covariance + KHS Weighting)")
plt.plot(res_hmm_2, label = "HMM Predicted (Custom Covariance + KHS Weighting)")
plt.plot(res_hmm_3, label = "HMM Predicted (Custom Covariance + Custom Weighting)")
plt.plot(res_hmm_4, label = "HMM Predicted (Custom Covariance + Bellman-Bulge Weighting)")
# plt.plot([0 for i in range(len(res_khs))], label = "Theoretical")
plt.xlabel("Time Slice")
plt.ylabel("Key")
yt = [i for i in range(24)]
plt.yticks(yt, MODE_TABLE)
plt.legend()
plt.title("Key Changes in full.mid")
plt.show()




### OBSERVATIONS

"""
1. The less observations for key changing, the more accurate the model.
2. The covariance matrix changes the results drastically.
3. The initial weighted data is not very accurate on a local scale, but predicts overall key more accurately.
4. My heuristic is decently accurate.
5. A multi-layer approach might be more effective.
6. The model does NOT like arpeggiation or modes, but can deal with modulation.
"""