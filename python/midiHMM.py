# Marc Browning
# IGME-589 Final Project
# Proof of Concept: HMM (Pre-trained) MIDI Key and Chord Interpreter

import pretty_midi, numpy

def midi_to_pc_sets(file, fs=10):
    # load file
    midi = pretty_midi.PrettyMIDI(file)
    # get chroma
    chroma = midi.get_chroma(fs=fs)
    times = numpy.arange(chroma.shape[1])/fs
    # convert to PCS
    pc_sets = []
    for i in range(chroma.shape[1]):
        # get active PCS at moment i where the chroma isn't 0
        active = numpy.where(chroma[:, i] > 0)[0]
        pc_sets.append((times[i], set(active)))
    return pc_sets



# helper function
def _flatten(pc_sets):
    # find the largest amount 
    res = []
    for pcs in pc_sets:
        for p in pcs: res.append(p)
    return res

a = "./test3.mid"
print(midi_to_pc_sets(a))