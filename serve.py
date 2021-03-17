import argparse
import json
import itertools

import flask
from flask import Flask, request
import torch
import numpy

from run import Runner
from preprocess import get_document
from tensorize import Tensorizer
from transformers import BertTokenizer, ElectraTokenizer

app = Flask(__name__)

SENTENCE_ENDERS = "!.?"
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def build_parser():
    parser = argparse.ArgumentParser(description='Serve Model')
    parser.add_argument('config', help='Config name to use', type=str)
    parser.add_argument('gpu', help='Which GPU to use', type=int)
    parser.add_argument('--model', help='Pre-trained model to use as basis', type=str, nargs="?", default=None)
    parser.add_argument('--port', help='Port to listen on', type=int, default=5000)
    return parser


@app.route('/predict', methods=['POST'])
def predict_text():
    """
    Predicts the clusters in a given text.
    """
    data = request.get_json(force=True)
    words = tokenizer.basic_tokenizer.tokenize(data['text'])
    # very simple sentence splitting
    words_with_sentence_boundaries = list(itertools.chain.from_iterable(
        [[word, ""] if word in SENTENCE_ENDERS else [word] for word in words]
    ))
    predicted_clusters = predict_token_list(words_with_sentence_boundaries)
    for cluster_id, cluster in enumerate(predicted_clusters):
        for pair in cluster:
            words[pair[0]] = "[" + words[pair[0]]
            words[pair[1]] = words[pair[1]] + "]" + str(cluster_id).translate(SUBSCRIPT)
    return " ".join(words)


def predict_token_list(token_list):
    document = get_document('_', token_list, 'german', 384, tokenizer, 'token_only')
    _, example = tensorizer.tensorize_example(document, is_training=False)
    # Remove gold and move to device
    tensorized = [torch.tensor(e).to(runner.device) for e in example[:7]]
    if runner.config['incremental']:
        span_starts, span_ends, mention_to_cluster_id, predicted_clusters = model(*tensorized)
    else:
        _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*tensorized)
        predicted_clusters, _, _ = model.get_predicted_clusters(
            span_starts.cpu().numpy(),
            span_ends.cpu().numpy(),
            antecedent_idx.cpu().numpy(),
            antecedent_scores.detach().cpu().numpy()
        )
    predicted_clusters_words = []
    token_map = tensorizer.stored_info['subtoken_maps']['_']
    for cluster in predicted_clusters:
        current_cluster = []
        for pair in cluster:
            current_cluster.append((token_map[pair[0]], token_map[pair[1]]))
        predicted_clusters_words.append(current_cluster)
    return predicted_clusters_words


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    runner = Runner(args.config, args.gpu)
    model = runner.initialize_model().to(runner.device)
    model.eval()
    tensorizer = Tensorizer(runner.config)
    torch.no_grad()
    if args.model:
        runner.load_model_checkpoint(model, args.model)
    if runner.config['model_type'] == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(runner.config['bert_tokenizer_name'], strip_accents=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(runner.config['bert_tokenizer_name'])
    app.run(port=args.port)
