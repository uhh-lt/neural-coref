import torch
from collections import Counter


class IncrementalEntities:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device

        self.emb = torch.tensor([]).to(device)
        self.count = torch.tensor([]).to(device)
        self.class_most_recent_entity = {}
        self.mention_distance = torch.tensor([]).to(device)
        self.sentence_distance = torch.tensor([]).to(device)
        self.mention_to_cluster_id = {}

    def _check_integrity(self):
        size = self.emb.shape[0]
        assert self.count.shape[0] == size
        assert max(self.class_most_recent_entity.values()) < size
        assert self.mention_distance.shape[0] == size
        assert self.sentence_distance.shape[0] == size
        assert max(self.mention_to_cluster_id.values()) < size

    def __len__(self):
        return len(self.emb)

    def evict(self):
        """
        Evicts entities that are older than the specified thresholds.
        """
        offset = 0
        for i, distance in enumerate(self.mention_distance.clone()):
            distance = distance.item()
            if (
                distance > self.conf["unconditional_eviction_limit"]
                or (distance > self.conf["singleton_eviction_limit"] and self.count[i] == 1)
            ) and len(self) > 1:
                self.emb = torch.cat(
                    [self.emb[: i - offset], self.emb[i + 1 - offset :]], 0
                ).to(self.device)
                self.sentence_distance = torch.cat(
                    [
                        self.sentence_distance[: i - offset],
                        self.sentence_distance[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                self.count = torch.cat(
                    [
                        self.count[: i - offset],
                        self.count[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                self.mention_distance = torch.cat(
                    [
                        self.mention_distance[: i - offset],
                        self.mention_distance[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                new_classes = {}
                for class_, entity_index in self.class_most_recent_entity.items():
                    if entity_index == i - offset:
                        pass
                    elif entity_index > i - offset:
                        new_classes[class_] = entity_index - 1
                    else:
                        new_classes[class_] = entity_index
                self.class_most_recent_entity = new_classes
                new_cluster_ids = {}
                for span, cluster in self.mention_to_cluster_id.items():
                    if cluster == i - offset:
                        pass
                    elif cluster > i - offset:
                        new_cluster_ids[span] = cluster - 1
                    elif cluster < i - offset:
                        new_cluster_ids[span] = cluster
                self.mention_to_cluster_id = new_cluster_ids
                offset += 1

    def add_entity(self, emb, gold_class, span_start, span_end, offset):
        span_start += offset
        span_end += offset
        if len(self) == 0:
            self.emb = emb.unsqueeze(0).to(self.device)
            self.count = torch.ones(1).to(self.device)
            self.class_most_recent_entity[gold_class] = 0
            self.sentence_distance = torch.zeros(1).unsqueeze(0).to(self.device)
            self.mention_distance = (
                torch.zeros(1).unsqueeze(0).type(torch.long).to(self.device)
            )
            self.mention_to_cluster_id[(span_start.item(), span_end.item())] = 0
        else:
            self.emb = torch.cat([self.emb, emb.unsqueeze(0)])
            self.count = torch.cat([self.count, torch.ones(1, device=self.device)])
            self.sentence_distance = torch.cat(
                [self.sentence_distance, torch.zeros(1).unsqueeze(0).to(self.device)]
            )
            self.mention_distance = torch.cat(
                [self.mention_distance, torch.zeros(1).unsqueeze(0).to(self.device)]
            )
            if gold_class:
                self.class_most_recent_entity[gold_class] = self.emb.shape[0] - 1
            self.mention_to_cluster_id[(span_start.item(), span_end.item())] = (
                self.emb.shape[0] - 1
            )
        self.mention_distance += 1

    def update_entity(
        self,
        cluster_to_update,
        emb,
        gold_class,
        span_start,
        span_end,
        update_gate,
        offset=0,
    ):
        span_start += offset
        span_end += offset
        if gold_class is not None:
            self.class_most_recent_entity[gold_class] = cluster_to_update.item()
        self.mention_to_cluster_id[
            (span_start.item(), span_end.item())
        ] = cluster_to_update.item()
        # High values in update gate mean the old representation is mostly replaced
        self.emb = self.emb.clone()
        self.emb[cluster_to_update] = (
            update_gate * emb
            + (torch.tensor(1) - update_gate) * self.emb[cluster_to_update].clone()
        )
        self.count[cluster_to_update] += 1
        self.mention_distance[cluster_to_update] = 0
        self.mention_distance += 1

    def get_result(self, remove_singletons=True):
        """
        Returns 4-tuple (span_starts, span_ends, mention_to_cluster_id, predicted_clusters) representing the entities in the cluster.
        """
        span_starts = []
        span_ends = []
        if remove_singletons:
            counter = Counter(self.mention_to_cluster_id.values())
            non_singletons = set(value for value, count in counter.items() if count > 1)
            cluster_mapping = {v:k for k,v in enumerate(non_singletons)}
            mention_to_cluster_id = {k: cluster_mapping[v] for k, v in self.mention_to_cluster_id.items() if v in non_singletons}
        else:
            mention_to_cluster_id = self.mention_to_cluster_id
        if len(mention_to_cluster_id) == 0:
            predicted_clusters_list = []
        else:
            predicted_clusters_list = [list() for _ in range((max(mention_to_cluster_id.values()) + 1))]
        for (start, end), cluster_id in mention_to_cluster_id.items():
            span_starts.append(start)
            span_ends.append(end)
            predicted_clusters_list[cluster_id].append((start, end))
        if len(mention_to_cluster_id) != 0:
            assert len(predicted_clusters_list) == (max(mention_to_cluster_id.values()) + 1)
        for c in predicted_clusters_list:
            print(c)
        return span_starts, span_ends, mention_to_cluster_id, predicted_clusters_list
