import os


def get_train_dataset():
    contents = []
    labels = []
    for root, dirs, files in os.walk("train"):
        for file in files:
            if file[0] == '.':
                continue
            with open("train/" + file, 'r') as f:
                raw = f.readlines()
                for line in raw:
                    if len(line) == 0:
                        continue
                    split_line = line.split('\t') # 只切割一次
                    try:
                        assert(split_line[1].strip() == "negative" or split_line[1].strip() =="positive" or split_line[1].strip() =="neutral")
                        if split_line[1].strip() == "negative":
                            labels.append(0)
                        elif split_line[1].strip() == "neutral":
                            labels.append(1) 
                        else:
                            labels.append(2)
                        #labels.append(split_line[1].strip())
                        contents.append(split_line[2].strip())
                    except:
                        try:
                            assert(split_line[2].strip() == "negative" or split_line[2].strip() =="positive" or split_line[2].strip() =="neutral")
                            if split_line[2].strip() == "negative":
                                labels.append(0)
                            elif split_line[2].strip() == "neutral":
                                labels.append(1) 
                            else:
                                labels.append(2)
                            #labels.append(split_line[2].strip())
                            contents.append(split_line[3].strip())
                        except:
                            print('error!!!!')
    assert(len(contents)==len(labels))
    return contents, labels

def get_test_dataset():
    labels = []
    contents = []
    with open('test.txt', 'r')as f:
        raw = f.readlines()
        for line in raw:
            if len(line) == 0:
                continue
            split_line = line.split('\t') # 只切割一次
            try:
                assert(split_line[1].strip() == "negative" or split_line[1].strip() =="positive" or split_line[1].strip() =="neutral")
                if split_line[1].strip() == "negative":
                    labels.append(0)
                elif split_line[1].strip() == "neutral":
                    labels.append(1) 
                else:
                    labels.append(2)
                contents.append(split_line[2].strip())
            except:
                try:
                    assert(split_line[2].strip() == "negative" or split_line[2].strip() =="positive" or split_line[2].strip() =="neutral")
                    if split_line[2].strip() == "negative":
                        labels.append(0)
                    elif split_line[2].strip() == "neutral":
                        labels.append(1) 
                    else:
                        labels.append(2)
                    contents.append(split_line[3].strip())
                except:
                    print('error!!!!')
    assert(len(contents)==len(labels))
    return contents, labels 
