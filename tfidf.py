# try tf-idf, it includes 2 part: do it by myself and do it by scikit library
from __future__ import division
import string
import math
from nltk.corpus import stopwords
import nltk
 
tokenize = lambda doc: doc.lower().split(" ")
 
document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"
 
all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

#download nltk packages
#nltk.download()

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def remove_punctuation(document):
    nopunct = [w for w in document if w not in string.punctuation]
    return ''.join(nopunct)

def remove_all_punctuation(documents):
    new_doc = []
    for document in documents:
        new_doc.append(remove_punctuation(document))
    return new_doc

def remove_stop_word(document):
    normalized = [w for w in document if w.lower() not in stopwords.words('english')]
    return ''.join(normalized)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    
#    for token in tokenized_documents:
#        print(token)
    
    idf = inverse_document_frequencies(tokenized_documents)
#    print (idf)

    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#print(remove_punctuation(document_0))
#print(remove_stop_word(document_0))
#print(remove_stop_word(remove_punctuation(document_0)))
all_documents = remove_all_punctuation(all_documents)
#print(all_documents[0])
tfidf_representation = tfidf(all_documents)
our_tfidf_comparisons = []
for count_0, doc_0 in enumerate(tfidf_representation):
    for count_1, doc_1 in enumerate(tfidf_representation):
        our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))
#print (tfidf_representation[0], document_0)


#in Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
 
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print (tfidf_representation[0])
print (sklearn_representation.toarray()[0].tolist())
print (document_0)

skl_tfidf_comparisons = []
for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
    for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

for x in zip(sorted(our_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
    print (x)
