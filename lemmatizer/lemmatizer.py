import csv, codecs

form2ind = {}
map_tags = {"мн." : "S", "м" : "S", "ж" : "S", "жо" : "S", "мо" : "S", "с" : "S", "п" : "A", "нсв" : "V", "св" : "V",
            "предл." : "PR", "союз" : "CONJ", "н" : "ADV", "вводн." : "ADV", "част." : "ADV", "межд." : "ADV"}

def build_form2ind(path):
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
                if row[i] not in form2ind:
                    form2ind[row[i]] = []
                if (row[0], row[1]) not in form2ind[row[i]]:
                    form2ind[row[i]].append((row[0], row[1]))
            line_count += 1

        form2ind["того"] = [("тот", "мест.")]
        form2ind["нет"] = [("нет", "част.")]
        form2ind["остальные"] = [("остальной", "п")]
        form2ind["консолидироваться"] = [("консолидироваться", "св")]
        form2ind["жениться"] = [("жениться", "св")]
        form2ind["акклиматизироваться"] = [("акклиматизироваться", "св"), ("акклиматизироваться", "нсв")]

        print(f'Processed {line_count} lines.')
        #print(form2ind["граждан"])
        #print(form2ind["о"])
        #print(form2ind["обезболивание"])
        #print(form2ind["времени"])

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

def print_ans_odict(sentences_tokens):
    lemmatized_sentences = []

    for s in sentences_tokens:
        res = []
        is_start = True

        for token in s:
            if token not in form2ind:
                right_case_token = token.lower()
            else:
                right_case_token = token

            if right_case_token not in form2ind:
                if token == "":
                    print(s)

                if token[0].isupper() and not is_start:
                    res.append(token + "{" + token + "=" + "S" + "}")
                else:
                    res.append(token + "{" + token + "=" + "S" + "}")
                continue

            if len(form2ind[right_case_token]) == 1:
                if form2ind[right_case_token][0][1] in map_tags:
                    res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags[
                        form2ind[right_case_token][0][1]] + "}")
                else:
                    res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + "NI" + "}")
            else:
                added = False

                for pair in form2ind[right_case_token]:
                    if pair[1] == "союз":
                        res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags["союз"] + "}")
                        added = True
                        break
                    if pair[1] == "предл.":
                        res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags["предл."] + "}")
                        added = True
                        break
                    if pair[1] == "нсв" or pair[1] == "св":
                        res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags["нсв"] + "}")
                        added = True
                        break
                    if pair[1] == "п":
                        res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags["п"] + "}")
                        added = True
                        break

                if added:
                    continue

                if form2ind[right_case_token][0][1] not in map_tags:
                    print(token)
                    res.append(token)
                else:
                    res.append(token + "{" + form2ind[right_case_token][0][0] + "=" + map_tags[
                        form2ind[right_case_token][0][1]] + "}")

            is_start = False

        lemmatized_sentences.append(res)

    for lst in lemmatized_sentences:
        ans = " "
        ans = ans.join(lst)
        print(ans)

if __name__ == "__main__":
    odict_path = 'odict.csv'
    dataset_path = "dataset_37845_2.txt"

    #f = codecs.open(odict_path, 'r', 'cp1251')
    #u = f.read()

    #ind = u.find(",н,")
    #print(u[0:1000])

    build_form2ind(odict_path)

    sentences_tokens = get_sentences_tokens(dataset_path)

    print_ans_odict(sentences_tokens)


