#######################################################
#
# Used to aggregate event counts for each day.
#
# Access event count is seperated according to object
# category.
#
# Run python feature_builder.py to execute, assuming
# input files reside in ../data directory. Output file
# will be saved in the same dir.
#
# Output file will be named log_train_event_counts.csv
#
#######################################################

from __future__ import division, print_function
from copy import copy

def log_reader(filename):
    """ iterate through log file """
    with open(filename, "r") as f:
        f.next()
        for line in f:
            row = line.strip().split(",")
            row[1] = row[1][:10]    # extract day from date time
            yield row

def mod_cat_lookup(filename):
    """ build category lookup table from object file """
    table = dict()
    with open(filename, "r") as f:
        f.next()
        for line in f:
            row = line.strip().split(",")
            mod_id = row[1]
            cat = row[2]
            table[mod_id] = cat

    return table

def add_cnt(cnt, event, object_id, cat_table):
    """ update event counter """
    if event == "access":
        try:
            event_type = "access_" + cat_table[object_id]
            cnt[event_type] += 1
        except KeyError:
            cnt["access_unknown"] += 1
    else:
        cnt[event] += 1
        
def main():
    log_filename = "../data/train/log_train.csv"
    object_filename = "../data/object.csv"

    cat_table = mod_cat_lookup(object_filename)
    current_date = None
    current_enroll = None
    cnt = dict(problem=0, video=0, wiki=0, discussion=0, nagivate=0, page_close=0, access_about=0,\
               access_chapter=0, access_combinedopenended=0, access_course=0, access_course_info=0,\
               access_dictation=0, access_discussion=0, access_html=0, access_outlink=0, \
               access_peergrading=0, access_problem=0, access_sequential=0, access_static_tab=0,\
               access_vertical=0, access_video=0, access_unknown=0)
    current_cnt = copy(cnt)

    f = open("../data/train/log_train_event_counts.csv", "w")
    f.write("enrollment_id,date," + ",".join(cnt.keys()) + "\n")
    for entry in log_reader(log_filename):
        enroll = entry[0]
        date = entry[1]
        event = entry[3]
        object_id = entry[4]
        # first entry
        if not current_enroll:
            current_enroll = enroll
            current_date = date
            add_cnt(current_cnt, event, object_id, cat_table)
        # change of enrollment id or date
        elif current_enroll != enroll or current_date != date:
            f.write("{0},{1},{2}\n".format(current_enroll, current_date, ",".join(map(str, current_cnt.values()))))
            current_enroll = enroll
            current_date = date
            current_cnt = copy(cnt)
            add_cnt(current_cnt, event, object_id, cat_table)
        # same date
        else:
            add_cnt(current_cnt, event, object_id, cat_table)
    f.write("{0},{1},{2}\n".format(current_enroll, current_date, ",".join(map(str, current_cnt.values()))))
    f.close()
    
if __name__ == "__main__":
    main()
