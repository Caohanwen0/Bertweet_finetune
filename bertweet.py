# use Bertweet for sentiment analysis
# SemEval-2017 Task 4A

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertForSequenceClassification
from transformers import BertConfig, AdamW
from sklearn.model_selection import train_test_split
from prepare_dataset_SemVal import get_train_dataset, get_test_dataset
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BERT_MODEL = 'vinai/bertweet-base'
PADDING_LENGTH = 128
ENABLE_GPU = True

epochs = 30
batch_size = 8
learning_rate = 1e-5 # fixed
grad_accumulate = 4 # 4 * 8 = 32
warm_up_ratio = 0

PATH_TO_DATASET = 'data'
CHECKPOINT_PATH = 'saved_models/tweetbert.pt'
SAVED_PATH = 'saved_models/valid_acc.csv'

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, normalization=True)
config = AutoConfig.from_pretrained(BERT_MODEL)

# model = ClassificationModel(config)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=3)
if ENABLE_GPU:
    model = model.to(device)
optimizer = AdamW(model.parameters(), lr = learning_rate) 
cross_entropy  = nn.NLLLoss()

temp_contents, temp_labels = get_train_dataset()
test_text, test_labels = get_test_dataset()

train_text, val_text, train_labels, val_labels = train_test_split(
    temp_contents,
    temp_labels, 
    test_size=0.1, 
    random_state = 2022,
    stratify = temp_labels) 

tokens_train = tokenizer.batch_encode_plus(
    train_text,
    max_length = PADDING_LENGTH,
    padding='max_length',
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text,
    max_length = PADDING_LENGTH,
    padding='max_length',
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length = PADDING_LENGTH,
    padding='max_length',
    truncation=True
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels)

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels)

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

def train():
    print("training...")
    model.train()
    total_loss = 0
    for step,batch in enumerate(train_dataloader):
        if ENABLE_GPU:
            batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch     
        loss = model(sent_id, labels = labels).loss
        total_loss = total_loss + loss.item()
        loss = loss / grad_accumulate 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if ((step + 1) % grad_accumulate == 0) or (step == (len(train_dataloader) - 1)):
            optimizer.step() 
            optimizer.zero_grad()
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

# function for evaluating the model
def evaluate():
    model.eval()
    total_accuracy = 0
    total_preds = []
    for step,batch in enumerate(val_dataloader):
        if ENABLE_GPU:
            batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id).logits
            preds = preds.detach().cpu().numpy() ##
            preds = np.argmax(preds, axis = 1) ## reduced to array of [0,1,2]
            labels = np.array(labels.tolist())
            acc = np.sum(preds == labels) / len(labels) #?????
            total_accuracy = total_accuracy + acc
    avg_acc = total_accuracy / len(val_dataloader)
    return avg_acc

def fine_tunning():
    # set initial acc to zero
    best_valid_acc = 0
    valid_acc_list =[] # store validation accuracy
    saving_cnt = 0 # count storing model times 
    for epoch in range(epochs):
        train_loss = train()
        print(f"in epoch {epoch + 1}, training loss is{train_loss}")
        valid_acc = evaluate()
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print(f"saving the {saving_cnt + 1}.th model")
            torch.save(model.state_dict(), CHECKPOINT_PATH) # save the newest model
            saving_cnt += 1
        valid_acc_list.append(valid_acc)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Accuracy: {valid_acc:.3f}')
    # 将valid_acc_list保存下来,用于日后画图使用
    print(valid_acc_list)

def check_performance():
    with torch.no_grad():
        if ENABLE_GPU:
            preds = model(test_seq.to(device), test_mask.to(device))
        else:
            preds = model(test_seq)
    preds = [prediction.argmax().item() for prediction in preds]
    preds = preds.detach().cpu().numpy()
    print("test_y is:", test_y)
    print("preds is:", preds)
    print(classification_report(test_y, preds))


if __name__ == "__main__":
    print("main starting...")
    fine_tunning()
    check_performance()

