from __future__ import division
import nltk
import re
import string as string_lib
import pandas as pd
import numpy
import networkx as nx
import math
import wikipedia
import itertools
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from py2neo import Graph, authenticate

authenticate("localhost:7474","neo4j","abhinav")
graph = Graph("http://localhost:7474/db/data/")
prereq_graph = nx.DiGraph()
un_prereq_graph = nx.Graph()

inlinks = {}
outlinks = {}
final_wiki_concepts = ['Topic', 'Image analysis', 'Digital image', 'Computer vision', 'Object detection', 'Artificial intelligence', 'Artificial intelligence (video games)', 'Pattern recognition', 'Pattern recognition (psychology)', 'Pattern Recognition (novel)', 'Pattern recognition (disambiguation)', 'Pattern Recognition (journal)', 'Machine vision']


def compute_RefD(A,B):

    x = len(set(outlinks[A]).intersection(set(inlinks[B])))/len(outlinks[A])
    y = len(set(outlinks[B]).intersection(set(inlinks[A]))) / len(outlinks[B])

    return (x - y)

def compute_WSS(A,B):

    if len(inlinks[A]) !=0 and len(inlinks[B]) !=0:
        a = math.log10(len(inlinks[A]))
        b = math.log10(len(inlinks[B]))
        a_b = len(set(inlinks[A]).intersection(set(inlinks[B])))
        if a_b:
            x = max(a,b) - math.log10(a_b)
            y = math.log10(13288294) - min(a,b)   # 13288294 are the total number of nodes in the graph (WIKI Statistics:- Content Pages: 5477369 and Total Pages: 43091395)
            return (1-(x/y))
        else:
            return 0
    else:
        return 0

def get_score(concept1, concept2):
    ref = compute_RefD(concept1,concept2)
    wss = compute_WSS(concept1,concept2)
    print("\n Pair: %s and %s has RefD score:%f and WSS score:%f"%(concept1,concept2,ref,wss))

    if ref > 0:
        if ref > 0.02 and wss !=0:
            prereq_graph.add_edge(concept1,concept2)
            un_prereq_graph.add_edge(concept1, concept2)
    else:
        if ref < -0.02 and wss !=0:
            prereq_graph.add_edge(concept2,concept1)
            un_prereq_graph.add_edge(concept1, concept2)


def acquire_inlinks_oulinks():
    for concept in final_wiki_concepts:
        temp = graph.data("MATCH (p0:Page {title:'%s'})-[Link]->(p:Page) RETURN p.title"%concept)
        outlinks[concept] = [temp[x]['p.title'] for x in range(len(temp))]
        temp = graph.data("MATCH (p0:Page {title:'%s'})<-[Link]-(p:Page) RETURN p.title"%concept)
        inlinks[concept] = [temp[x]['p.title'] for x in range(len(temp))]

    # print("\nOutlink Structure",outlinks)

def wiki_based_similarity():
    acquire_inlinks_oulinks()
    pairwise_concepts = list(itertools.combinations(final_wiki_concepts,2))
    for pair in pairwise_concepts:
        get_score(pair[0],pair[1])

## https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx

def main():

    prereq_graph.add_nodes_from(final_wiki_concepts)
    wiki_based_similarity()

    Connected_components = nx.connected_components(un_prereq_graph)
    print("\n Pre-req Graph successfully created")
    # print("\nConnected Components: ")
    # print(Connected_components)
    nx.draw_networkx(prereq_graph,with_labels=True,arrows=True,font_size=6)
    plt.axis('off')
    plt.savefig("graph_prereq.png")
    # plt.show()

if __name__ == "__main__":
    main()