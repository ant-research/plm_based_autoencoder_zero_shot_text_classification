import torch, os, logging, json, random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
import preprocessor as pre
pre.set_options(pre.OPT.URL, pre.OPT.EMOJI)
stopwords_list = stopwords.words('english')
stopwords = {i: 1 for i in stopwords_list}
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def cal_accuracy(preds, targets):
    return accuracy_score(targets, preds)


def cal_microprecision(preds, targets):
    return precision_score(targets, preds, average="micro")


def cal_microrecall(preds, targets):
    return recall_score(targets, preds, average="micro")


def cal_microf1(preds, targets):
    return f1_score(targets, preds, average='micro')


def cal_binaryprecision(preds, targets):
    return precision_score(targets, preds)


def cal_binaryrecall(preds, targets):
    return recall_score(targets, preds)


def cal_binaryf1(preds, targets):
    return f1_score(targets, preds)


def cal_macroprecision(preds, targets):
    return precision_score(targets, preds, average="macro")


def cal_macrorecall(preds, targets):
    return recall_score(targets, preds, average="macro")


def cal_macrof1(preds, targets):
    return f1_score(targets, preds, average='macro')


def cal_weightedf1(preds, targets):
    return f1_score(targets, preds, average='weighted')


def cal_weightedprecision(preds, targets):
    return precision_score(targets, preds, average="weighted")


def cal_weightedrecall(preds, targets):
    return recall_score(targets, preds, average="weighted")


def calculate_perf(preds, targets):
    METRICS2FN = {"Accuracy": cal_accuracy,
                  "micro-F1": cal_microf1,
                  "macro-F1": cal_macrof1,
                  "weighted-F1": cal_weightedf1,
                  "macro-Precision": cal_macroprecision,
                  "macro-Recall": cal_macrorecall,
                  "micro-Precision": cal_microprecision,
                  "micro-Recall": cal_microrecall,
                  "weighted-Precision": cal_weightedprecision,
                  "weighted-Recall": cal_weightedrecall}

    return_dict = {}
    if not isinstance(targets[0], list) and len(set(targets)) == 2:
        METRICS2FN.update({"binary-F1": cal_binaryf1, "binary-Recall": cal_binaryrecall, "binary-Precision": cal_binaryprecision})
    for k, v in METRICS2FN.items():
        return_dict[k] = round(v(preds, targets), 4)
    if isinstance(targets[0], list):
        return_dict["support"] = sum([sum(tgt) for tgt in targets])
    else:
        return_dict["support"] = len(targets)
    return return_dict


def get_cos_sim(reps1, reps2):
    reps1_normalized = reps1.div(reps1.norm(p=2, dim=1, keepdim=True))
    reps2_normalized = reps2.div(reps2.norm(p=2, dim=1, keepdim=True))
    cos_sim_matrix = torch.einsum("ab,cb->ac", reps1_normalized, reps2_normalized)
    return cos_sim_matrix


def get_euclidean_matrix(reps1, reps2):
    return torch.cdist(reps1, reps2, p=2)


def read_jsonl(filepath):
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    return examples


def read_class_names(filepath):
    classes = []
    with open(filepath, "r") as f:
        for line in f:
            classes.append(line.strip())
    return classes


def get_grams(examples_texts, num_features_to_extract=None, ngram_range=(1, 1)):
    # we get need n-grams so here either CountVectorizer or TfidfVectorizer gives us the same results (for computational efficiency, we use CountVectorizer here)
    vectorizer = CountVectorizer(max_features=num_features_to_extract, ngram_range=ngram_range, token_pattern=r"\b[^\d\W]+\b", stop_words=stopwords_list)

    try:
        vectorizer.fit(examples_texts)
        phrases = vectorizer.get_feature_names()
        return phrases
    except:
        return []


def get_grams_from_data(data_path, num_features_to_extract=5000, ngram_range=(1, 1)):
    examples = read_jsonl(data_path)
    # with p.clean, empirically works better
    examples_texts = [pre.clean(e["text"]) for e in examples]
    grams = get_grams(examples_texts, num_features_to_extract=num_features_to_extract, ngram_range=ngram_range)
    return grams


def add_filehandler_for_logger(output_path, logger, out_name="log"):
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(output_path, f"{out_name}.txt"), mode="a")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_label(input_tensor):
    '''soft labeling, following: https://arxiv.org/pdf/2010.07245.pdf'''
    weight = input_tensor ** 2 / torch.sum(input_tensor, dim=0)
    target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
    return target_dist


class MyDataset(Dataset):
    def __init__(self, encoded_examples):
        self.encoded_examples = encoded_examples

    def __getitem__(self, index):
        selected_to_return = {}
        for k, v in self.encoded_examples.items():
            selected_to_return[k] = v[index]
        return selected_to_return

    def __len__(self):
        return len(self.encoded_examples["input_ids"])


def translate(texts, model, tokenizer, language="fr"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors="pt").to(device)
    model.to(device)
    # Generate translation using model
    translated = model.generate(**encoded)
    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts


def back_translate(texts, target_tokenizer, target_model, en_tokenizer, en_model, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer,
                         language=target_lang)
    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                      language=source_lang)
    return back_translated_texts


def combins_permus(stuff):
    import itertools
    to_return = []
    for L in range(1, len(stuff) + 1):
        to_return.extend(list(itertools.combinations(stuff, L)))
    return to_return


def aug_with_bt(text,target_tokenizer, target_model, en_tokenizer, en_model, num_aug=10):
    supported_langs = ['fr', 'es', 'it', 'pt', 'ro', 'ca', 'gl', 'la', 'wa', 'oc', 'sn', 'an', 'co', 'rm']
    lang_seqs = combins_permus(supported_langs)[:num_aug]
    aug_texts = []
    for lang_seq in lang_seqs:
        aug_text = back_translate([text], target_tokenizer, target_model, en_tokenizer, en_model, source_lang="en", target_lang=lang_seq[0])
        for lang in lang_seq[1:]:
            aug_text = back_translate([aug_text], target_tokenizer, target_model, en_tokenizer, en_model, source_lang="en", target_lang=lang)
        aug_texts.extend(aug_text)
    return aug_texts
