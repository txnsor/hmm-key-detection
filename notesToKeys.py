
"""
Understanding a western scale as being one of
twelve pitch class sets (with 7 diatonic
modes cooresponding to one pitch class set,
and each pitch class set representing a
major key) we can find the possible keys 
given the notes played.
"""

C_MAJOR = [0, 2, 4, 5, 7, 9, 11]
def notesToPCS(notes):
    # generate possible pitch class sets
    all_keys = []
    # for each of the chromatic notes
    for i in range(12):
        # generate a pitch class set
        all_keys.append(sorted([(k+i)%12 for k in C_MAJOR]))
    # compare notes
    in_key = True
    res = []
    for key in all_keys:
        for note in notes:
            if note not in key: in_key = False
        if in_key: res.append(key)
        in_key = True
    return res

print(notesToPCS([11, 5]))