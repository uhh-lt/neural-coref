import logging
import itertools
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from tensorize import CorefDataProcessor
import util
import time
from run import Runner
from os.path import join
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import MentionModel
import conll
import sys
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class MentionRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_saved_suffix = None

    def initialize_model(self, saved_suffix=None):
        model = MentionModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs_mention'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        pbar = tqdm(total=len(examples_train) * epochs)
        model.train()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                example_gpu = [d.to(self.device) for d in example]
                loss = model(*example_gpu)[3]
                loss.backward()
                loss_history.append(loss)
                loss_during_report += loss
                for optimizer in optimizers:
                    optimizer.step()
                model.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()
                # Report
                if len(loss_history) % conf['report_frequency'] == 0:
                    # Show avg loss during last report interval
                    avg_loss = loss_during_report / conf['report_frequency']
                    loss_during_report = 0.0
                    end_time = time.time()
                    logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                    start_time = end_time

                    tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                    tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                    tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                # Evaluate
                if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                    f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                    if f1 > max_f1:
                        max_f1 = f1
                        self.save_model_checkpoint(model, len(loss_history))
                    logger.info('Eval max f1: %.2f' % max_f1)
                    start_time = time.time()

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)

        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        doc_to_prediction = {}

        negative_examples = 0.0
        positive_examples = 0.0
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                scores, labels, cluster_to_spans, loss = model(*example_gpu)
                doc_to_prediction[doc_key] = cluster_to_spans.values()
                total += len(labels)
                predicted = (scores > 0.5).T.squeeze()
                labels = labels.to(torch.bool)
                positive_examples += sum(labels)
                negative_examples += len(labels) - sum(labels)
                true_positives += sum(predicted & labels)
                false_positives += sum(predicted & torch.logical_not(labels))
                true_negatives += sum(torch.logical_not(predicted) & torch.logical_not(labels))
                false_negatives += sum(torch.logical_not(predicted) & labels)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        metrics = {
            'precision': precision,
            'recall': recall,
            'accuracy': (true_positives + true_negatives) / total, # This is pretty useless here due to the number of true negatives
            'f1': f1,
        }
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)
        logger.info('Positive weight for this distribution: %.5f' % (negative_examples / positive_examples).item())

        subtoken_map = stored_info['subtoken_maps']
        duplicates_removed = {}
        # remove duplicates
        for doc_key, clusters in doc_to_prediction.items():
            handled = set()
            doc_no_duplicates = []
            for cluster_id, mentions in enumerate(clusters):
                new_mentions = []
                for start, end in mentions:
                    start_word, end_word = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                    if (start_word, end_word) in handled:
                        pass
                    else:
                        handled.add((start_word, end_word))
                        new_mentions.append((start, end))
                doc_no_duplicates.append(new_mentions)
            duplicates_removed[doc_key] = doc_no_duplicates

        if official:
            conll_results = conll.evaluate_conll(self.config['conll_scorer'], conll_path, duplicates_removed, stored_info['subtoken_maps'], "/tmp/ev_out")
        return f1.item(), metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i, tensor_example in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_task),
            LambdaLR(optimizers[1], lr_lambda_task),
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        suffix = f'model_{self.name_suffix}_{step}.bin'
        self.last_saved_suffix = suffix
        path_ckpt = join(self.config['log_dir'], suffix)
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = MentionRunner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)

    # Restore best parameters
    runner.load_model_checkpoint(model, runner.last_saved_suffix)
