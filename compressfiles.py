import gzip
import shutil
import os, sys

"""
This script compresses text file into gzip
"""


fpath = '/home/varsha/fl-proj/lingspam_public/lemm_stop/part9'
fl_list = os.listdir(fpath)

for fl in fl_list:
    if ".txt.gz" not in fl:
        print(fl)
        fop = os.path.join(fpath, fl)
        print(type(fop))

        with open(fop, 'rb') as f_in:
            fop += ".gz"
            with gzip.open(fop, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
