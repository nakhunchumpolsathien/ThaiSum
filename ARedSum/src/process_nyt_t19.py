import xml.etree.ElementTree as ET
import pandas as pd
import os, sys
import tarfile
import argparse
from multiprocess import Pool

SPECIAL_WORDS1 = ["photo", "graph", "chart", "map", "table", "drawing"]
SPECIAL_WORDS = set([x + "s" for x in SPECIAL_WORDS1] + SPECIAL_WORDS1)
#print(SPECIAL_WORDS)

def process_abstract(abstract, min_length=50):
    if abstract is None:
        return None
    abstract = abstract.lower().replace("(m)", "").replace("(s)", "").replace("(l)", "")
    sents = abstract.split(";")
    updated_sents = []
    for sent in sents:
        if sent.strip() in SPECIAL_WORDS:
            continue
        updated_sents.append(sent)
    updated_abstract = ";".join(updated_sents)
    if len(updated_abstract.split()) < min_length:
        return None
    else:
        return updated_abstract

def output_file(path, abstract, full_text):
    with open(path, 'w') as fout:
        fout.write("%s/\n\n" % full_text)
        fout.write("@highlight\n")
        fout.write(abstract)

def parse_xml(f, tgz_fname, output_path, year):
    tree = ET.parse(f)
    root = tree.getroot()
    doc_id = None
    for child in root.findall('head/docdata/doc-id'):
        doc_id = child.attrib['id-string']
    assert doc_id in tgz_fname
    updated_abstract = None
    for child in root.iter('abstract'):
        for summary in child.iter('p'):
            updated_abstract = process_abstract(summary.text)
    if updated_abstract is None:
        return
    #print("{}\t{}".format(doc_id, updated_abstract))

    text_arr = []
    for child in root.iter('block'):
        block = child.attrib['class']
        if block == 'full_text':
            for para in child.iter('p'):
                text_arr.append(para.text)
    full_text = "\n".join(text_arr)
    if not full_text:
        return
    outfname = "%s/%s_%s" % (output_path, year, tgz_fname.replace("./", "").replace("/", "_").replace("xml", "story"))
    print(outfname)
    output_file(outfname, updated_abstract, full_text)

def process_one_year(params):
    years, args = params
    for year in years:
        target_dir = "{}/{}".format(args.input_path, year)
        files = os.listdir(target_dir)
        for fname in files:
            fname = "%s/%s" % (target_dir, fname)
            tar = tarfile.open(fname, "r:gz")
            for member in tar.getmembers():
                #print(member.name)
                f=tar.extractfile(member)
                #print(f)
                if f is not None:
                    parse_xml(f, member.name, args.output_path, year)

def main():
    #all the years
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", type=str, default="/mnt/nfs/work1/kbi/collections/T19_data")
    parser.add_argument("-output_path", type=str, default="/home/kbi/work1kbi/summarization/new_york_times/working")
    parser.add_argument("-min_year", default=1985, type=int)
    parser.add_argument("-interval_years", default=40, type=int)
    parser.add_argument("-n_cpus", default=2, type=int)
    args = parser.parse_args()
    os.system("mkdir -p %s" % args.output_path)
    #[min_year,min_year+interval_years)

    #train_years = []
    years = []
    for dir_name in os.listdir(args.input_path):
        #if int(dir_name) < 1990:
        years.append(dir_name)

    years_list = [years[i:i+args.interval_years] for i in range(0,len(years), args.interval_years)]
    print(years_list)
    para_list = [(x, args) for x in years_list]
    pool = Pool(args.n_cpus)
    for d in pool.imap_unordered(process_one_year, para_list):
        continue

    #for d in pool.imap_unordered(parse_xml, para_list):
    #    continue
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()

