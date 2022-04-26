import os
import numpy as np
import re

dataDir = '/u/cs401/A3/data/'


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    nr = ['<s>'] + r[:] + ['</s>']
    nh = ['<s>'] + h[:] + ['</s>']
    lst = np.zeros((len(nr), len(nh)))
    lst[0, :] = np.arange(len(lst[0, :]))
    lst[:, 0] = np.arange(len(lst[:, 0]))
    for i in range(len(lst[:, 0]) - 1):
        for j in range(len(lst[0, :]) - 1):
            sub = lst[i, j] if nr[i + 1] == nh[j + 1] else lst[i, j] + 1
            lst[i + 1, j + 1] = min(lst[i, j + 1] + 1, lst[i + 1, j] + 1, sub)

    lst[-1, -1] -= 1
    i, j = lst.shape[0] - 1, lst.shape[1] - 1
    sub, ins, dlt = 0, 0, 0

    while i > 0 and j > 0:
        if i > 0 and j > 0:
            k = [lst[i - 1, j - 1], lst[i, j - 1], lst[i - 1, j]]
        elif j > 0:
            k = [np.inf, lst[i, j - 1], np.inf]
        else:
            k = [np.inf, np.inf, lst[i - 1, j]]
        k_min = k.index(min(k))
        if k_min == 0:
            if lst[i - 1, j - 1] == lst[i, j] - 1:
                sub += 1
            i -= 1
            j -= 1
        elif k_min == 1:
            ins += 1
            j -= 1
        else:
            dlt += 1
            i -= 1
    wer = np.inf if len(r) == 0 else (lst[-1, -1] / len(r)) * 100
    return round(wer, 3), sub, ins, dlt


def preprocess(info):
    return re.sub(r"[^a-zA-Z0-9\s\[\]]", r"", info).lower().strip().split()


if __name__ == "__main__":
    goo = []
    kal = []
    for r, d, f in os.walk(dataDir):
        for speaker in d:
            trans = open(os.path.join(r, speaker, 'transcripts.txt'), 'r').readlines()
            gf = open(os.path.join(r, speaker, 'transcripts.Google.txt'), 'r').readlines()
            kf = open(os.path.join(r, speaker, 'transcripts.Kaldi.txt'), 'r').readlines()

            for i, j in enumerate(trans):
                gw, gs, gi, gd = Levenshtein(preprocess(j), preprocess(gf[i]))
                goo.append(gw)
                print("{} {} {} {} S:{}, I:{}, D:{}".format(speaker, 'Google', i, gw, gs, gi, gd))

                kw, ks, ki, kd = Levenshtein(preprocess(j), preprocess(kf[i]))
                kal.append(kw)
                print("{} {} {} {} S:{}, I:{}, D:{}\n".format(speaker, 'Kaldi', i, kw, ks, ki, kd))

    print("Google mean: {},  standard deviation:{}".format(np.average(goo), np.std(goo)))
    print("Kaldi mean: {},  standard deviation:{}".format(np.average(kal), np.std(kal)))

