#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import string
import re

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# self added

FUTURE_TENSE = {'will', 'gonna', "'ll"}

MR_TO_NUM = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

# Path for data files
PATH = {"RW": "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv",
        "BNGL": "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"}

CAT_PATH = {"Left": "/u/cs401/A1/feats/Left_IDs.txt",
            "Center": "/u/cs401/A1/feats/Center_IDs.txt",
            "Right": "/u/cs401/A1/feats/Right_IDs.txt",
            "Alt": "/u/cs401/A1/feats/Alt_IDs.txt"}

NP_PATH = {"Left": "/u/cs401/A1/feats/Left_feats.dat.npy",
           "Center": "/u/cs401/A1/feats/Center_feats.dat.npy",
           "Right": "/u/cs401/A1/feats/Right_feats.dat.npy",
           "Alt": "/u/cs401/A1/feats/Alt_feats.dat.npy"}

NP_ARRAY = {"Left": np.load(NP_PATH["Left"]),
            "Center": np.load(NP_PATH["Center"]),
            "Right": np.load(NP_PATH["Right"]),
            "Alt": np.load(NP_PATH["Alt"])}


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # print(3)
    feats = np.zeros(173)
    texts = re.compile("(\S+)/(?=\S+)").findall(comment)
    tags = re.compile("(?<=\S)/(\S+)").findall(comment)
    muti = 0
    feats[16] = comment.count("\n")
    total_len_token = 0
    total_token = len(texts)

    ##############
    # var for 18~29
    aoa = []
    img = []
    fam = []
    vms = []
    ams = []
    dms = []

    for i in range(0, total_token):
        total_len_token += len(texts[i])
        if texts[i].isupper() and len(texts[i]) > 2:
            feats[0] += 1

        text = texts[i].lower()

        if text in bn:
            aoa.append(float(bn[text][0]))
            img.append(float(bn[text][1]))
            fam.append(float(bn[text][2]))
        if text in rw:
            vms.append(float(rw[text][0]))
            ams.append(float(rw[text][1]))
            dms.append(float(rw[text][2]))

        if text in FIRST_PERSON_PRONOUNS:
            feats[1] += 1
        elif text in SECOND_PERSON_PRONOUNS:
            feats[2] += 1
        elif text in THIRD_PERSON_PRONOUNS:
            feats[3] += 1
        elif text in FUTURE_TENSE:
            feats[6] += 1

        if text not in string.punctuation or i == total_token - 1:
            if muti > 1:
                feats[8] += 1
            muti = 0
        else:
            muti += 1
            if text == ",":
                feats[7] += 1

        if tags[i] == "CC":
            feats[4] += 1
        elif "VB" in tags[i]:
            if tags[i] == "VBD":
                feats[5] += 1
            elif tags[i] == "VB" and i > 2 and texts[i - 1] == "to" and tags[i - 2] == "VBG" and texts[i - 2] == "go":
                feats[6] += 1
        elif tags[i] == "NN" or tags[i] == "NNS":
            feats[9] += 1
        elif tags[i] == "NNP" or tags[i] == "NNPS":
            feats[10] += 1
        elif tags[i] == "RB" or tags[i] == "RBR" or tags[i] == "RBS":
            feats[11] += 1
        elif "WP" in tags[i] or tags[i] == "WRB" or tags[i] == "WDT":
            feats[12] += 1

        if text in SLANG:
            feats[13] += 1
    if feats[16] != 0:
        feats[14] = total_token / feats[16]
    if total_token != 0:
        feats[15] = total_len_token / total_token
    ################################################## # 17 end here
    if len(aoa) == 0:
        aoa = [0]
    if len(img) == 0:
        img = [0]
    if len(fam) == 0:
        fam = [0]
    if len(vms) == 0:
        vms = [0]
    if len(ams) == 0:
        ams = [0]
    if len(dms) == 0:
        dms = [0]
    feats[17], feats[18], feats[19] = np.mean(aoa), np.mean(img), np.mean(fam)
    feats[20], feats[21], feats[22] = np.std(aoa), np.std(img), np.std(fam)
    feats[23], feats[24], feats[25] = np.mean(vms), np.mean(ams), np.mean(dms)
    feats[26], feats[27], feats[28] = np.std(vms), np.std(ams), np.std(dms)

    return feats


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    # print(4)
    feats = np.append(feat[:29], NP_ARRAY[comment_class][dir_of_files[comment_class][comment_id]])
    return feats


def main(args):
    # Declare necessary global variables here.
    global rw, bn, dir_of_files
    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # Data read
    rwf = open(PATH["RW"], "r")
    rwf.readline()
    rwline = rwf.readline()
    rw = {}
    while rwline:
        rwlst = rwline.strip("\n").split(",")
        rw[rwlst[1]] = (rwlst[2], rwlst[5], rwlst[8])
        rwline = rwf.readline()
    rwf.close()

    bnf = open(PATH["BNGL"], "r")
    bnf.readline()
    bnline = bnf.readline()
    bn = {}
    while bnline:
        bnlst = bnline.strip("\n").split(",")
        bn[bnlst[1]] = (bnlst[3], bnlst[4], bnlst[5])
        bnline = bnf.readline()
    bnf.close()

    dir_of_files = {"Left": {}, "Center": {}, "Right": {}, "Alt": {}}

    for part in CAT_PATH.keys():
        f = open(CAT_PATH[part], "r")
        cat_data = f.read().split()
        for i in range(len(cat_data)):
            dir_of_files[part][cat_data[i]] = i
        f.close()

    for i, comment in enumerate(data):
        # print("*******")
        # print(i)
        feats[i, 0: 173] = extract1(comment["body"])
        feats[i, 0: 173] = extract2(feats[i, 0: 173], comment["cat"], comment["id"])
        feats[i, -1] = MR_TO_NUM[comment["cat"]]

    feats = feats.astype(np.float32)
    feats = np.nan_to_num(feats)

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
