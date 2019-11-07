import csv, codecs
import xml.etree.ElementTree as ET

form2lemma = {}
map_tags = {"мн." : "S", "со" : "S", "м" : "S", "ж" : "S", "жо" : "S", "мо" : "S", "мо-жо" : "S", "с" : "S", "п" : "A",
            "числ.-п" : "A", "мс-п" : "A", "нсв" : "V", "св" : "V", "св-нсв" : "V", "предл." : "PR", "союз" : "CONJ", "сравн." : "ADV",
            "н" : "ADV", "вводн." : "ADV", "част." : "ADV", "межд." : "ADV", "предик." : "NI", "числ." : "NI", "мест." : "NI",
            "VERB" : "V", "UNKN" : "NI", "PREP" : "PR", "ADJS" : "A", "ADJF" : "A", "NOUN" : "S", "NPRO" : "NI",
            "PRCL" : "ADV", "ADVB" : "ADV", "CONJ" : "CONJ", "INFN" : "V", "GRND" : "V", "PRTS" : "V", "PRTF" : "V",
            "COMP" : "A", "INTJ" : "ADV", "NUMR" : "NI", "PRED" : "NI", "Prnt" : "ADV"}
freq = {}
freq_lemmas = {}
tags_in_freq = {}

def build_form2lemma_from_odict(odict_path):
    with open(odict_path, encoding="cp1251") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            n = len(row)
            for i in range(n):
                if i == 1:
                    continue
                if row[i] == "":
                    continue
                if row[i] not in form2lemma:
                    form2lemma[row[i]] = []
                if (row[0], row[1]) not in form2lemma[row[i]]:
                    form2lemma[row[i]].append((row[0], row[1]))
            line_count += 1

        form2lemma["того"] = [("тот", "мест.")]
        form2lemma["нет"] = [("нет", "част.")]
        form2lemma["остальные"] = [("остальной", "п")]
        form2lemma["остальных"] = [("остальной", "п")]
        form2lemma["консолидироваться"] = [("консолидироваться", "св")]
        form2lemma["жениться"] = [("жениться", "св")]
        form2lemma["акклиматизироваться"] = [("акклиматизироваться", "св"), ("акклиматизироваться", "нсв")]
        form2lemma["сорок"] = [("тот", "числ.")]
        form2lemma["рекомендуется"] = [("рекомендоваться", "нсв")]
        form2lemma["деформироваться"] = [("деформироваться", "нсв")]

        print(f'Processed {line_count} lines.')
        print(form2lemma["было"])
        #print(form2lemma["о"])
        #print(form2lemma["обезболивание"])
        #print(form2lemma["времени"])

def build_for2lemma_from_opcorpora(opcorpora_dict_path):
    #f = codecs.open(opcorpora_dict_path, 'r')
    #u = f.read()
    #print(u[0:80000])

    tree = ET.parse(opcorpora_dict_path)
    root = tree.getroot()

    lemmata = root[2]
    for lemma in lemmata:
        lem = lemma[0].attrib['t']
        tag = lemma[0][0].attrib['v']

        if tag == "PNCT" or tag == "LATN" or tag == "SYMB" or tag == "NUMB" or tag == "ROMN":
            continue

        mapped_tag = map_tags[tag]

        for child in lemma:
            if child.tag == "l":
                continue
            word = child.attrib['t']
            if word not in form2lemma:
                form2lemma[word] = []
            # search for mapped_tag in form2lemma:
            exist = False
            for elem in form2lemma[word]:
                if elem[1] == mapped_tag:
                    exist = True
                    break

            if not exist:
                form2lemma[word].append((lem, mapped_tag))

    #print(form2lemma["ежа"])

def form2lemma_map_tags():
    for elem in form2lemma:
        num = len(form2lemma[elem])
        for i in range(num):
            if form2lemma[elem][i][1] not in map_tags:
                print(elem)
                print(form2lemma[elem][i][0])
                print(form2lemma[elem][i][1])
                print("________")
                continue
            form2lemma[elem][i] = (form2lemma[elem][i][0], map_tags[form2lemma[elem][i][1]])

    print(form2lemma["конец"])

def build_freq(opcorpora_annot_path):
    #f = codecs.open(opcorpora_annot_path, 'r')
    #u = f.read()
    #print(u[0:5000])

    tree = ET.parse(opcorpora_annot_path)
    root = tree.getroot()

    #print(root.tag)

    for text in root:
        paragraphs = text[1]
        for paragraph in paragraphs:
            # print(paragraph.tag, paragraph.attrib)
            for sentence in paragraph:
                # print(sentence.tag, sentence.attrib)
                tokens = sentence[1]
                for token in tokens:
                    word = token.attrib['text']
                    lemma = token[0][0][0].attrib['t']
                    tag = token[0][0][0][0].attrib['v']

                    if word == "почти" or word == "Почти":
                        print("EXIST")

                    if tag == "PNCT" or tag == "LATN" or tag == "SYMB" or tag == "NUMB" or tag == "ROMN":
                        continue

                    if tag not in map_tags:
                        print(token.attrib['text'])
                        print(token[0][0][0].attrib['t'])
                        print(token[0][0][0][0].attrib['v'])
                        print("_________________________________")
                        continue

                    mapped_tag = map_tags[tag]

                    if (word, mapped_tag) not in freq:
                        freq[(word, mapped_tag)] = (0, lemma)
                    freq[(word, mapped_tag)] = (freq[(word, mapped_tag)][0] + 1, lemma)

                    if (lemma, mapped_tag) not in freq_lemmas:
                        freq_lemmas[(lemma, mapped_tag)] = 0
                    freq_lemmas[(lemma, mapped_tag)] = freq_lemmas[(lemma, mapped_tag)] + 1

                    if word not in tags_in_freq:
                        tags_in_freq[word] = []
                    if mapped_tag not in tags_in_freq[word]:
                        tags_in_freq[word].append(mapped_tag)

    print(freq[("решения", "S")])
    print(tags_in_freq["сбор"])

def get_sentences_tokens(dataset_path):
    dataset_file = open(dataset_path, "r")
    raw_text = dataset_file.read()
    sentences = raw_text.split("\n")
    sentences_num = len(sentences)

    print(sentences)

    sentences_tokens = []

    for i in range(sentences_num):
        if sentences[i] == "":
            continue
        sentences[i] = sentences[i].replace(".", "").replace(",", "").replace("?", "").replace("!", "")
        sentences_tokens.append(sentences[i].split(" "))

    print(sentences_tokens)
    return sentences_tokens

def print_ans_odict_only(sentences_tokens):
    lemmatized_sentences = []

    for s in sentences_tokens:
        res = []
        num = -1
        for token in s:
            num += 1
            if token not in form2lemma:
                right_case_token = token.lower()
            else:
                right_case_token = token

            if right_case_token not in form2lemma:
                if token.isupper():
                    res.append(token + "{" + token + "=" + "S" + "}")
                elif token[0].isupper() and num != 0:
                    res.append(token + "{" + token + "=" + "S" + "}")
                elif num == 0:
                    res.append(token + "{" + token.lower() + "=" + "S" + "}")
                else:
                    res.append(token + "{" + token + "=" + "S" + "}")
                continue

            if len(form2lemma[right_case_token]) == 1:
                if form2lemma[right_case_token][0][1] in map_tags:
                    res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags[
                        form2lemma[right_case_token][0][1]] + "}")
                else:
                    res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + "NI" + "}")
            else:
                added = False

                for pair in form2lemma[right_case_token]:
                    if pair[1] == "союз":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags["союз"] + "}")
                        added = True
                        break
                    if pair[1] == "предл.":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags["предл."] + "}")
                        added = True
                        break
                    if pair[1] == "нсв" or pair[1] == "св":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags["нсв"] + "}")
                        added = True
                        break
                    if pair[1] == "п":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags["п"] + "}")
                        added = True
                        break

                if added:
                    continue

                if form2lemma[right_case_token][0][1] not in map_tags:
                    print(token)
                    res.append(token)
                else:
                    res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + map_tags[
                        form2lemma[right_case_token][0][1]] + "}")

        lemmatized_sentences.append(res)

    for lst in lemmatized_sentences:
        ans = " "
        ans = ans.join(lst)
        print(ans)

def get_most_freq_pair_by_infn(word):
    max_freq = 0
    best_ans = None
    for pair in form2lemma[word]:
        lemma = pair[0]
        tag = pair[1]

        if (lemma, tag) not in freq_lemmas:
            continue
        if freq_lemmas[(lemma, tag)] > max_freq:
            max_freq = freq_lemmas[(lemma, tag)]
            best_ans = (lemma, tag)
    return best_ans

def get_most_freq_pair(word):
    if word not in tags_in_freq:
        return get_most_freq_pair_by_infn(word)
    max_freq = 0
    best_ans = None
    for tag in tags_in_freq[word]:
        cur_freq, lemma = freq[(word, tag)]
        if cur_freq > max_freq:
            max_freq = cur_freq
            best_ans = (lemma, tag)
    return best_ans

def find_lemma(word, tag, default):
    lst = form2lemma[word]
    for elem in lst:
        if elem[1] == tag:
            return elem[0]
    return default

def print_ans_dict_and_freq(sentences_tokens):
    lemmatized_sentences = []

    for s in sentences_tokens:
        res = []
        num = -1
        for token in s:
            num += 1
            if token not in form2lemma:
                right_case_token = token.lower()
            else:
                right_case_token = token

            if right_case_token not in form2lemma:
                print(f"Not in dict: {token}")
                if token[0].isupper() and num != 0:
                    res.append(token + "{" + token + "=" + "S" + "}")
                elif num == 0:
                    res.append(token + "{" + token.lower() + "=" + "S" + "}")
                else:
                    res.append(token + "{" + token + "=" + "S" + "}")
                continue

            if len(form2lemma[right_case_token]) == 1:
                res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" +
                        form2lemma[right_case_token][0][1] + "}")
            else:
                best_pair = get_most_freq_pair(right_case_token)
                if best_pair != None:
                    right_lemma = find_lemma(right_case_token, best_pair[1], best_pair[0])
                    res.append(token + "{" + right_lemma + "=" + best_pair[1] + "}")
                    continue

                added = False
                for pair in form2lemma[right_case_token]:
                    if pair[1] == "CONJ":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + "CONJ" + "}")
                        added = True
                        break
                    if pair[1] == "PR":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + "PR" + "}")
                        added = True
                        break
                    if pair[1] == "V":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + "V" + "}")
                        added = True
                        break
                    if pair[1] == "A":
                        res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + "A" + "}")
                        added = True
                        break

                if added:
                    continue

                # choose random lemma (first lemma)
                res.append(token + "{" + form2lemma[right_case_token][0][0] + "=" + form2lemma[right_case_token][0][1] + "}")

        lemmatized_sentences.append(res)

    for lst in lemmatized_sentences:
        ans = " "
        ans = ans.join(lst)
        print(ans)

if __name__ == "__main__":
    odict_path = 'odict.csv'
    dataset_path = "dataset_37845_7.txt"
    opcorpora_annot_path = "annot.opcorpora.no_ambig.nonmod.xml"
    opcorpora_dict_path = "dict.opcorpora.xml"

    #f = codecs.open(odict_path, 'r', 'cp1251')
    #u = f.read()

    #ind = u.find(",н,")
    #print(u[0:1000])

    build_form2lemma_from_odict(odict_path)
    form2lemma_map_tags()

    build_for2lemma_from_opcorpora(opcorpora_dict_path)

    build_freq(opcorpora_annot_path)

    sentences_tokens = get_sentences_tokens(dataset_path)

    print_ans_dict_and_freq(sentences_tokens)

    #print(form2lemma["почти"])
    #print(tags_in_freq["почти"])


