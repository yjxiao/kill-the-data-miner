from __future__ import division, print_function

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
        
def main():
    pass
    
if __name__ == "__main__":
    main()
