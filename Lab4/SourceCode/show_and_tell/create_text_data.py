import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
desc_list = []

def clean_descriptions():
    table = str.maketrans('', '', string.punctuation)
    index = 0
    with open("captions/captions.txt") as openfile:
        for line in openfile:
            index = index+1
            desc = line
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove stop words
            desc = [word for word in desc if word not in stop_words]
            # # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list.append(' '.join(desc))

# save descriptions to file, one per line
def save_descriptions(descriptions, filename,_list):
    total_list = []
    for index in range(len(_list)):
        total_list.append(str(index+1) + ' '+str(_list[index]))
    data = '\n'.join(total_list)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def generate():
    length = len(desc_list)
    train = .7 * length
    validate = .3 * length
    train_list =[]
    validate_list=[]
    for i in range(int(train)):
        train_list.append(str(i+1)+".jpg")
    train_data = '\n'.join(train_list)
    file = open("train/train.txt", 'w')
    file.write(train_data)
    file.close()
    for i in range(int(validate)):
        validate_list.append(str(1+i+int(train))+".jpg")
    validate_data = '\n'.join(validate_list)
    file = open("validation/validation.txt", 'w')
    file.write(validate_data)
    file.close()





# Clean descriptions
clean_descriptions()

# save descriptions
save_descriptions(desc_list, 'descriptions.txt', desc_list)

generate()

