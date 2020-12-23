# -*- coding: utf-8 -*-

import difflib

############################################
### HELPER/UTIL FUNCTIONS FOR DATAFRAMES ###
############################################

def get_unique_vals(data, col="type"):
    """
            returns unique values of a column
            call to (e.g.) data.type.unique()
    """
    return getattr(getattr(data, col), 'unique')().tolist()

def find_best_match(outlier, targetlist):
    """
            returns best match (= most similar word) for outlier from target list
    """
    hits = []
    for i, word in enumerate(targetlist):
        similarity = difflib.SequenceMatcher(
            None, outlier.lower(), word.lower()).ratio()
        hits.append(similarity)

    # Index of highest Value in hits
    idx_max = max(range(len(hits)), key=hits.__getitem__)

    return targetlist[idx_max]