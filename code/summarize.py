import argparse
from os import listdir
import re
import numpy as np

'''Searches the path folder for result csv files created by main.py after
running membership or attribute inference attacks using machine learning model
model_type. Then, creates a summary of the results in those files.

The first field of the result file must be rseed, and all files must have the
same fields in the same order. The summary file has all fields that are in the
result files, except that rseed is replacde with param.'''
def summarize(path, model_type):
    regex = '{}-([0-9.]*)\.csv'.format(model_type)
    outfile = '{}/{}-summary.csv'.format(path, model_type)
    summary_rows = [] #(param, summary_row)
    
    for fname in listdir(path):
        m = re.match(regex, fname)
        if m is not None:
            param = float(m.group(1)) #value of param used by main.py
            infile = '{}/{}'.format(path, fname)
            
            #Make sure that the first field is rseed and that all files have the same fields
            with open(infile) as f:
                try:
                    cur_fieldnames = f.readline().split(',')
                    cur_fieldnames = list(map(lambda x: x.strip(), cur_fieldnames))
                    assert cur_fieldnames[0] == 'rseed'
                    assert fieldnames == cur_fieldnames
                except NameError: #if fieldnames does not exist yet (first file read)
                    fieldnames = cur_fieldnames
            
            raw_result = np.loadtxt(infile, skiprows=1, delimiter=',')
            summary_row = np.mean(raw_result, axis=0)
            summary_row = list(map(lambda x: '{:.4f}'.format(x), summary_row))
            summary_rows.append((param, summary_row))
    
    #Replace rseed with param
    fieldnames[0] = 'param'
    for param, summary_row in summary_rows:
        summary_row[0] = str(param)
    
    summary_rows = sorted(summary_rows, key=lambda tup: tup[0]) #sort by param
    
    with open(outfile, 'w') as f:
        f.write(','.join(fieldnames) + '\n')
        for _, summary_row in summary_rows:
            f.write(','.join(summary_row) + '\n')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a summary of the result files created by main.py')
    parser.add_argument('path', help='folder containing the result files')
    parser.add_argument('model', choices=['tree', 'linreg', 'knn'], help='machine learning model used to create the result files')
    args = parser.parse_args()
    
    path = args.path
    model_type = args.model
    
    summarize(path, model_type)