import pandas as pd
import xml.etree.ElementTree as ET
import spacy
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import os
from TermNode import *
#from parseXML import *
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# def iter_docs(author):
#     author_attr = author.attrib
#     for doc in author.iter('document'):
#         doc_dict = author_attr.copy()
#         doc_dict.update(doc.attrib)
#         doc_dict['data'] = doc.text
#         yield doc_dict



def listFilePaths(start :int, end :int, end2 :int):
    '''
    Return 
    list of specific file paths in each dataset
    AND
    list of specific file names in each dataset
    range = end - start: number of files we want to read
    '''

    ## Puts filenames of dataset into two sorted lists
    dirNames = ["DrugBank", "MedLine"]
    # /Users/bchu/Documents/UCSD/Computer Science/LIGN 167 is cwd
    cur_path = os.getcwd() + "/Datasets/"
    ds1_path = cur_path + dirNames[0]
    ds2_path = cur_path + dirNames[1]
    ds1_files = sorted(os.listdir(ds1_path))
    ds2_files = sorted(os.listdir(ds2_path))
    # Each element contains the file path
    ds1_file_paths = [ds1_path + '/' + i for i in ds1_files]
    ds2_file_paths = [ds2_path + '/' + i for i in ds2_files]

    return ds1_file_paths[start:end], ds2_file_paths[start:end2], ds1_files[start:end], ds2_files[start:end2]


def parse_sentence(xml: list) -> dict:
    '''
    parse xml files in dataset to find names and types of drugs
    does not account for links, but isolates the named entities

    @params:
        xml: the current file being passed in

    @returns:
        a dictionary of nodes in format {node name : node}'''

    nodes = dict()

    for line in xml:
        if ("type" and "text" in line) and ("sentence" not in line):
            
            curr_quote = line.find("\"") + 1
            next_quote = line.find("\"", curr_quote)

            kind = line[curr_quote:next_quote]

            curr_quote = line.find("\"", next_quote + 1) + 1
            next_quote = line.find("\"", curr_quote)

            name = line[curr_quote:next_quote].lower()

            if name not in nodes:
                nodes[name] = TermNode(name = name, kind = kind)

    return nodes

def parse_effects2str(xml :list):#, nodes :dict):
    effects = list()
    for line in xml:
        if("sentence" and "text" in line):
            curr_quote = line.find("\"") + 1
            next_quote = line.find("\"", curr_quote)

            kind = line[curr_quote:next_quote]

            curr_quote = line.find("\"", next_quote + 1) + 1
            next_quote = line.find("\"", curr_quote)
            eff = line[curr_quote:next_quote]

            text = eff + " are the effects of"

            effects.append(text)
        # if ("type" and "text" in line) and ("sentence" not in line):
            
        #     curr_quote = line.find("\"") + 1
        #     next_quote = line.find("\"", curr_quote)

        #     kind = line[curr_quote:next_quote]

        #     curr_quote = line.find("\"", next_quote + 1) + 1
        #     next_quote = line.find("\"", curr_quote)

        #     name = line[curr_quote:next_quote].lower()
        #     effects.append(name)
        
    effects = list2str(effects)
    return effects


def parse_file2list(filename, kind="list"):
    '''
    parses file into list of strings or one long string

    @params:
        filename: the current file being passed in
        kind: type want to return, list be default

    @returns: 
        file as a list or string
    '''
    file = open(filename,"r")
    file = file.readlines()
    if(kind.lower() == "string"):
        file = list2str(file)
    return file

def list2str(sents :list):
    sents = " ".join(sents)
    return sents

def parse_sen2corpus(xml :list):
    ## not done yet
    return xml


########################## PROCESSING ##########################

def files2corpus(file_names :list):
    sents = list()
    for n in file_names:
        sents.append(parse_file2list(n))
    corpus = list2str(sents)

def segment_and_tokenize(corpus):
	#make sure to run: 
	# pip install -U pip setuptools wheel
	# pip install -U spacy
	# python -m spacy download en_core_web_sm
	#in the command line before using this!

	#corpus is assumed to be a string, containing the entire corpus
	nlp = spacy.load('en_core_web_sm')
	tokens = nlp(corpus)
	sents = [[t.text for t in s] for s in tokens.sents if len([t.text for t in s])>1]
	sents = remove_infrequent_words(sents)
	sents = [['<START>']+s+['<END>'] for s in sents]
	return sents

def remove_infrequent_words(sents):
    '''
    Take in list of sentences and remove infrequent words.
    '''
    word_counts = {}
    for s in sents:
        for w in s:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1

    threshold = 2
    filtered_sents = []
    for s in sents:
        new_s = []
        for w in s:
            if word_counts[w] < threshold:
                new_s.append('<UNKOWN>')
            else:
                new_s.append(w)
        filtered_sents.append(new_s)
    return filtered_sents


##### filter words
def filter_stop(corpus):
    sp = spacy.load('en_core_web_sm')
    # all_stopwords = sp.Defaults.stop_words
    all_stopwords = "in","to", "an", "a", "the"
    text_tokens = word_tokenize(corpus)
    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
    return tokens_without_sw

##########################  CHECKING  ##########################

## Randomly select 20% of files as test dataset
## 60% as training data, 20% as pre-training

# dirData = listFilePaths(0,9,0)
# file_name = dirData[0][2]
# print(file_name)
# file_Aslist = parse_file2list(file_name)
# print(parse_effects2str(file_Aslist))


# file_aslist = parse_file2list(file_name)
# print(type(file_aslist))
# nodes1 = parse_sentence(file_aslist)
# print(TermNode.getName(nodes1["iron"]))
# print(dirData[2])

# segment_and_tokenize(file_list2str)
dirfiles = listFilePaths(0,5,1)
unfiltered_corpus = list()
for file in dirfiles[0]:
    unfiltered_corpus.append(parse_effects2str(parse_file2list(file)))   
unfiltered_corpus = list2str(unfiltered_corpus)
corpus = list2str(filter_stop(unfiltered_corpus))
# print(corpus)





##########################  TEXT GENERATION  ##########################



prompt = "The combined effects are: "

model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
inputs = tokenizer(corpus + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

prompt_length = len(tokenizer.decode(inputs[0]))
outputs = model.generate(inputs, max_length=1000, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1 :]

print(generated)


# classifier = pipeline("sentiment-analysis")
# result = classifier(file_list2str)[0]
# print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

##########################

# iterator
# testFile = "Abarelix_ddi.xml"
# test_path = '/' + testFile
# # Iterate through the xml file
# tree = ET.parse(ds1_file[0])
# itrE = tree.iter()
# print(len(itrE))
# print(itrE.__sizeof__)
# print(next(itrE))
# print(next(itrE))
# print("hi")
# #print(next(itrE))

# len1 = len(ds1data) # 573 elements
# len2 = len(ds2data) # 142 elements

# path_parent = os.path.dirname(cur_path), to get parent

# xml_data = io.StringIO("XML STRING HERE"")

print("hi")

# etree = ET.parse(xml_data) #create an ElementTree object 
# doc_df = pd.DataFrame(list(iter_docs(etree.getroot())))
# print(type(doc_df))
