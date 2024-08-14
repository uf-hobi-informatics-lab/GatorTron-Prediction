from pathlib import Path
import pickle as pkl
from collections import defaultdict, Counter
from itertools import permutations, combinations
from functools import reduce
import numpy as np
import os
import re
import sys
# from annotation2BIO import pre_processing, read_annotation_brat, generate_BIO


def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)

        
def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data


def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt


def save_text(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)

# Change these system path according to the location you store the code
sys.path.append("/home/chenaokun1990/N2C2/NLPpreprocessing")
sys.path.append("home/chenaokun1990/N2C2//NLPreprocessing/text_process")
from annotation2BIO import pre_processing, read_annotation_brat, generate_BIO
MIMICIII_PATTERN = "\[\*\*|\*\*\]"

# max valid cross sentence distance
CUTOFF = 1
# output 5-fold cross validation data
OUTPUT_CV = False

# from sentence_tokenization import logger as l1
# from annotation2BIO import logger as l2
# l1.disabled = True
# l2.disabled = True

def create_entity_to_sent_mapping(nnsents, entities, idx2e):
    loc_ens = []
    
    ll = len(nnsents)
    mapping = defaultdict(list)
    for idx, each in enumerate(entities):
        en_label = idx2e[idx]
        en_s = each[2][0]
        en_e = each[2][1]
        new_en = []
        
        i = 0
        while i < ll and nnsents[i][1][0] < en_s:
            i += 1
        s_s = nnsents[i][1][0]
        s_e = nnsents[i][1][1]

        if en_s == s_s:
            mapping[en_label].append(i)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
        else:
            mapping[en_label].append(i)
            print("first index not match ", each)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
    return mapping

def gen_relations(mappings, ens, e2i, nnsents, nsents, f_stem, do_train=True, valid_comb=None):
    # relations: [{"T24": [0, "adverse ALLERGIES : [s1] Penicillin [e1] .", "Drug", "T24", "13_851"]}]
    relations = {}
    cnt = 0

    for item, key in zip(ens, mappings.keys()):
        if do_train:
            ent, label, e_loc, ent_id = item
        else:
            ent, _, e_loc, ent_id = item
            label = "NonRel"

        if ent_id != None:
            key = ent_id

        i = 0
        txt_index = 0
        while i < len(nsents):
            if nsents[i][0][1][0] <= e_loc[0]:
                txt_index = i
                i += 1
            else:
                break

        ann_sent = ""
        is_end = False
        while not is_end:
            for word, w_loc, _, _, _ in nsents[txt_index]:
                if w_loc[0] == e_loc[0]:
                    ann_sent += "[s] "

                ann_sent += word
                ann_sent += " "

                if w_loc[1] == e_loc[1]:
                    ann_sent += "[e] "
                    is_end = True

            txt_index += 1

        ann_sent = ann_sent[:-1]

        if f_stem == "177-03":
            print(ent, e_loc)
            print(ann_sent)


        relations[key] = relations.get(key,[])+[cnt, label, ann_sent, key, f_stem]
        cnt += 1

    #
    return relations
        

    
def create_training_samples(file_path, valids=None, valid_comb=None):
    fids = []
    root = Path(file_path)
    
    d_med = {}
    d_event = {}

    # Debug
    ls_a_cnter = []
   
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem+".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
        e2i, ens, evt, a_cnter = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        ls_a_cnter.append((txt_fn, a_cnter))

        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        # print(nsents)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)

        medications = gen_relations(mappings, ens, e2i, nnsents, nsents, txt_fn.stem)
        print("**************************************************************************")
        # print("entities:\n", ens)

        for key in medications.keys():
            if d_med.get(key):
                d_med[key].append(medications[key])
            else:
                d_med[key] = [medications[key]]

        for evt_name in evt.keys():
            events = gen_relations(mappings, evt[evt_name], e2i, nnsents, nsents, txt_fn.stem)

            # print("Events: ", evt_name, "\n", evt[evt_name])

            if not d_event.get(evt_name):
                d_event[evt_name] = {}

            for key in events.keys():
                if d_event[evt_name].get(key):
                    d_event[evt_name][key].append(events[key])
                else:
                    d_event[evt_name][key] = [events[key]]

    print(ls_a_cnter)
      
    return d_med, d_event


def create_test_samples(file_path, valids=None, valid_comb=None):
    #create a separate mapping file
    fids = []
    root = Path(file_path)

    preds = {}
    d_event = {}

    
    for txt_fn in root.glob("*.txt"):
        fids.append(txt_fn.stem)
        ann_fn = root / (txt_fn.stem + ".ann")

        # load text
        txt = load_text(txt_fn)
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN)
        e2i, ens, evt = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)

        meds = gen_relations(mappings, ens, e2i, nnsents, nsents, txt_fn.stem, do_train=False)
        print("**************************************************************************")
        # print(relations)

        for key in meds.keys():
            if preds.get(key):
                preds[key].append(meds[key])
            else:
                preds[key] = [meds[key]]

        for evt_name in evt.keys():
            events = gen_relations(mappings, evt[evt_name], e2i, nnsents, nsents, txt_fn.stem)
            d_event[evt_name] = {}

            for key in events.keys():
                if d_event[evt_name].get(key):
                    d_event[evt_name][key].append(events[key])
                else:
                    d_event[evt_name][key] = [events[key]]
            
    return preds, d_event

def to_tsv(data, fn):
    # data: adverse ALLERGIES : [s1] Penicillin [e1] . Drug T24 13_851
    header = "\t".join([str(i+1) for i in range(len(data[0]))])
    with open(fn, "w") as f:
        f.write(f"{header}\n")
        for each in data:
            d = "\t".join([str(e) for e in each])
            f.write(f"{d}\n")


def to_5_cv(data, ofd):
    if not os.path.isdir(ofd):
        os.mkdir(ofd)
    
    np.random.seed(13)
    np.random.shuffle(data)
    
    dfs = np.array_split(data, 5)
    a = [0,1,2,3,4]
    for each in combinations(a, 4):
        b = list(set(a) - set(each))[0]
        n = dfs[b]
        m = []
        for k in each:
            m.extend(dfs[k])
        if not os.path.isdir(os.path.join(ofd, f"sample{b}")):
            os.mkdir(os.path.join(ofd, f"sample{b}"))
        
        to_tsv(m, os.path.join(ofd, f"sample{b}", "train.tsv"))
        to_tsv(n, os.path.join(ofd, f"sample{b}", "dev.tsv"))


def all_in_one(*dd, dn="2022n2c2", status="Train", evt_name = "med"):
    # data: adverse ALLERGIES : [s1] Penicillin [e1] . Drug T24 13_851
    # dd: [{key: [0, "adverse ALLERGIES : [s1] Penicillin [e1] .", "Drug", "T24", "13_851"]}]
    data = []
    # print(dd)
    # print("dd above")
    print("--------------------------------------------------------------------------")
    for d in dd:
        for k, v in d.items():
            # print(v)
            for each in v:
                data.append(each[1:])
    # Change the output path to the place you need
    output_path = f"/home/chenaokun1990/datasets/example"
    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)

    if status == "Train":
        to_tsv(data, p/"train.tsv")
        if OUTPUT_CV:
            to_5_cv(data, p.as_posix())
    elif status == "Dev":
        to_tsv(data, p/"dev.tsv")
    else:
        to_tsv(data, p/"test.tsv")

n2c2_training = "/data/datasets/Aokun/2022_N2C2/2022n2c2_track1/trainingdata_v3/example"

d_med, d_event = create_training_samples(n2c2_training, None, None)

print("d_event:\n", d_event)

all_in_one(d_med, dn="2022n2c2", status='Train')




