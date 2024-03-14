import math
import pandas as pd
from ir_measures import Bpref
import ir_measures
import numpy as np
from sklearn.metrics import roc_auc_score

class MetricCompute:
    def __init__(self, gt_tagging, image_path2query_ind, image_path2gallery_ind, gallery_ind2id, query_ind2id):
        self.gt_tagging = gt_tagging
        self.image_path2query_ind = image_path2query_ind
        self.image_path2gallery_ind = image_path2gallery_ind
        self.query_ind2id = query_ind2id
        self.gallery_ind2id = gallery_ind2id
        
        self.query_ind2image_path = {v: k for k, v in image_path2query_ind.items()}
        self.gallery_ind2image_path = {v: k for k, v in image_path2gallery_ind.items()}
        self.id2query_ind = {v: k for k, v in query_ind2id.items()}
        self.id2gallery_ind = {v: k for k, v in gallery_ind2id.items()}
        self.labels = {(item[0], item[1]): item[2] for item in self.gt_tagging}
        
    def get_label(self, query, gal_id):
        gallery = self.gallery_ind2image_path[gal_id]
        return self.labels.get((query, gallery), -1)
    
    def get_label_by_fnames(self, query, gallery):
        return self.labels.get((query, gallery), -1)
        
    def calc_dcs(self, gt_tagging, ranked_cands_per_seed, total_gallery_images, alpha=10):
        """This method computes the DCS score described in our submision. It is a metric that measures the quality of the ranking of the gallery images for each seed image. The metric is based on the percentile rank of the correct gallery image in the ranked list of candidates.

        Args:
            gt_tagging: A list of tuples with the ground truth tagging of the gallery images for each seed image. Each tuple is in the form (seed_fname, cand_fname, value), where value is 1 if the gallery image is relevant and 0 otherwise.
            ranked_cands_per_seed: A list of lists with the ranked list of gallery images for each seed image.
            total_gallery_images (int): The total number of gallery images in the dataset.
            alpha (float, optional): an adjustable parameter controlling the shape of the credit curve and the range of p treated as the bottom.. Defaults to 10.

        Returns:
            (float, DataFrame) The DCS score and a DataFrame with the DCS score for each seed image.
        """
        scores = []
    
        scores_data = []

        for (seed_fname, cand_fname, value) in gt_tagging:
            seed_ind = self.image_path2query_ind[seed_fname]
            cand_ind = self.image_path2gallery_ind[cand_fname]
            
            if cand_ind not in ranked_cands_per_seed[seed_ind]:
                continue

            seed_cand_ranks = ranked_cands_per_seed[seed_ind]
            rank = seed_cand_ranks.index(cand_ind)

            percentile_rank = 1 - (rank / total_gallery_images) # We take the completion because the rank is lower when better
            
            if value == 1:
                curr_metric_score = (math.exp(alpha * percentile_rank) - 1) / (math.exp(alpha) - 1)
                scores += [curr_metric_score]
                scores_data += [{"seed": seed_fname, "gallery": cand_fname, "dcs": curr_metric_score}]
            if value == 0:
                curr_metric_score = (math.exp(alpha) - math.exp(alpha * percentile_rank)) / (math.exp(alpha) - 1)
                scores += [curr_metric_score]
                scores_data += [{"seed": seed_fname, "gallery": cand_fname, "dcs": curr_metric_score}]
                
        scores_df = pd.DataFrame(scores_data)
        
        score_per_query_df = scores_df.groupby('seed').agg({"dcs": "mean"})
                
        return score_per_query_df['dcs'].mean(), scores, score_per_query_df
    
    def calc_bpref(self, gt_tagging, ranked_cands_per_seed, cand_score_per_seed):
        """This method computes the Bpref score described in our submision.

        Args:
            gt_tagging: A list of tuples with the ground truth tagging of the gallery images for each seed image. Each tuple is in the form (seed_fname, cand_fname, value), where value is 1 if the gallery image is relevant and 0 otherwise.
            ranked_cands_per_seed: A list of lists with the ranked list of gallery images for each seed image.
            cand_score_per_seed (_type_): A list of lists with the scores of the gallery images for each seed image.

        Returns:
            (float, DataFrame) The Bpref score and a DataFrame with the Bpref score for each seed image.
        """
        qrels = pd.DataFrame(gt_tagging, columns=["query_id", "doc_id", "relevance"])

        run = {
            self.query_ind2image_path[query_idx]: {
                self.gallery_ind2image_path[gallery_idx]: cand_score_per_seed[query_idx][gallery_rank] for gallery_rank, gallery_idx in enumerate(query_cand_ranks)
            }
            for query_idx, query_cand_ranks in enumerate(ranked_cands_per_seed)
        }

        query_measures = list(ir_measures.itercalc([Bpref], qrels, run=run))
        
        bpref_per_seed = pd.DataFrame(query_measures).rename(columns={'value': 'bpref', 'query_id': 'seed'}).drop(columns='measure')
        
        return bpref_per_seed['bpref'].mean(), bpref_per_seed

    def calc_coverage_at_k(self, ranked_cands_per_seed, k=5):
        """This method computes the coverage@k score described in our submision. It is a metric that measures the percentage of annotated gallery images that are in the top-k ranked list of candidates for each seed image.

        Args:
            ranked_cands_per_seed: A list of lists with the ranked list of gallery images for each seed image.
            k (int, optional): The amount of top candodates to evaluate. Defaults to 5.

        Returns:
            _type_: _description_
        """
        ranked_cands_per_seed = np.array(ranked_cands_per_seed)[:, :k]
        
        coverage_data = []
        
        for query in range(len(ranked_cands_per_seed)):
            query_image_path = self.query_ind2image_path[query]
            
            for gallery_rank, gallery_ind in enumerate(ranked_cands_per_seed[query]):
                gallery_image_path = self.gallery_ind2image_path[gallery_ind]
                
                label = self.get_label_by_fnames(query_image_path, gallery_image_path)
                
                coverage_data.append({
                    "seed": query_image_path,
                    "gallery": gallery_image_path,
                    "label": label,
                    "gallery_rank": gallery_rank,
                })
        
        coverage_df = pd.DataFrame(coverage_data)
        
        covered_df = coverage_df[coverage_df['label'] != -1]
        covered_percentage = len(covered_df) / len(coverage_df)
        
        coverage_per_query_df = coverage_df.groupby('seed').apply(lambda group: pd.Series({
            f"coverage@{k}": len(group[group['label'] != -1]) / len(group)
            }
        )).reset_index()
        
        return covered_percentage, coverage_per_query_df
    
    def calc_ehr_at_k(self, ranked_cands_per_seed, ks=[5, 20]):
        """This method computes the EHR@k score described in our submision. It is a metric that measures the percentage of annotated positive gallery images that are in the top-k ranked list of candidates for each seed image out of all annotated pairs in the top-k.

        Args:
            ranked_cands_per_seed: A list of lists with the ranked list of gallery images for each seed image.

        Returns:
            (dict, DataFrame) A dictionary with the EHR@k score for each k and a DataFrame with the EHR@k score for each seed image.
        """
        hits_at_k = {k: 0.0 for k in ks}
        hits_data = []
        num_seeds = set()
        total_annotations_at_k = {k: 0 for k in ks}
        count_annotations_per_seed_at_k = {k: {} for k in ks}
        
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            if value == 1:
                num_seeds.add(seed_fname)
                
            cand_ind = self.image_path2gallery_ind[cand_fname]
            seed_ind = self.image_path2query_ind[seed_fname]
            seed_cand_ranks = ranked_cands_per_seed[seed_ind]
            seed_cand_ranks = [i for i in seed_cand_ranks if self.gallery_ind2id[i] != self.query_ind2id[seed_ind]]

            for k in ks:
                rank = -1
                
                if cand_ind in seed_cand_ranks[:k]:
                    total_annotations_at_k[k] += 1
                    count_annotations_per_seed_at_k[k][seed_fname] = count_annotations_per_seed_at_k[k].get(seed_fname, 0) + 1
                    
                    if value == 1:
                        hits_at_k[k] += 1.0
                        rank = seed_cand_ranks.index(cand_ind) + 1.0
                
                if value == 1:   
                    hits_data += [{
                        "seed": seed_fname,
                        "rank": rank,
                        "k": k,
                    }]
                    
        hits_df = pd.DataFrame(hits_data)
        
        hits_per_query = hits_df.groupby(['seed', 'k']).apply(lambda group: pd.Series({
            f"ehr@{group.name[1]}": len(group[group['rank'] > 0]) / count_annotations_per_seed_at_k[group.name[1]].get(group.name[0], 0) if count_annotations_per_seed_at_k[group.name[1]].get(group.name[0], 0) > 0 else math.nan
        })).reset_index()

        hits_per_query = hits_per_query.pivot(index='seed', columns='k').drop(columns=["level_2"])
        hits_per_query.columns = [f"ehr@{col}" for col in hits_per_query.columns.get_level_values(1)]
        hits_per_query = hits_per_query.reset_index()
        
        hits_per_query = hits_per_query[['seed', *hits_per_query.filter(like="ehr@").columns]]
        
        hits_at_k_met = {k: {'mean': (v / (total_annotations_at_k[k])) if total_annotations_at_k[k] > 0 else math.nan} for k, v in hits_at_k.items()}
        
        return hits_at_k_met, hits_per_query

    def calc_ht_at_k(self, ranked_cands_per_seed, ks=[5, 20]):
        """This method computes the HT@k score described in our submision. It is a metric that measures the percentage of annotated positive gallery images that are in the top-k ranked list of candidates for each seed image.

        Args:
            cand_score_per_seed (_type_): A list of lists with the scores of the gallery images for each seed image.
            ks (list, optional): _description_. Defaults to [5, 20].

        Returns:
            _type_: _description_
        """
        hits_at_k = {k: 0.0 for k in ks}
        hits_data = []
        num_seeds = set()
        
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            if value == 0:
                continue
            
            num_seeds.add(seed_fname)
            cand_ind = self.image_path2gallery_ind[cand_fname]
            seed_ind = self.image_path2query_ind[seed_fname]
            seed_cand_ranks = ranked_cands_per_seed[seed_ind]
            seed_cand_ranks = [i for i in seed_cand_ranks if self.gallery_ind2id[i] != self.query_ind2id[seed_ind]]

            for k in ks:
                rank = -1
                
                if cand_ind in seed_cand_ranks[:k]:
                    hits_at_k[k] += 1.0
                    rank = seed_cand_ranks.index(cand_ind) + 1.0
                    
                hits_data += [{
                    "seed": seed_fname,
                    "rank": rank,
                    "k": k,
                }]

        hits_df = pd.DataFrame(hits_data)
        
        hits_per_query = hits_df.groupby(['seed', 'k']).apply(lambda group: pd.Series({
            f"hr@{group.name[1]}": len(group[group['rank'] > 0]) / group.name[1]
        })).reset_index()

        hits_per_query = hits_per_query.pivot(index='seed', columns='k').drop(columns=["level_2"])
        hits_per_query.columns = [f"hr@{col}" for col in hits_per_query.columns.get_level_values(1)]
        hits_per_query = hits_per_query.reset_index()
        
        hits_per_query = hits_per_query[['seed', *hits_per_query.filter(like="hr@").columns]]
        
        hits_at_k_met = {k: {'mean': v / (k * len(num_seeds))} for k, v in hits_at_k.items()}
        
        return hits_at_k_met, hits_per_query    
    

    def calc_roc_auc(self, cand_score_per_seed, ranked_cands_per_seed):
        """This method computes the ROC AUC score described in our submision.

        Args:
            cand_score_per_seed (_type_): A list of lists with the scores of the gallery images for each seed image.
            ranked_cands_per_seed: A list of lists with the ranked list of gallery images for each seed image.

        Returns:
            _type_: _description_
        """
        scores_data = []
    
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            seed_ind = self.image_path2query_ind[seed_fname]
            cand_ind = self.image_path2gallery_ind[cand_fname]
            
            seed_cand_ranks = ranked_cands_per_seed[seed_ind]
            
            # If the candidate exists in the query's top k predictions
            if cand_ind in seed_cand_ranks:
                rank = seed_cand_ranks.index(cand_ind)
                score = cand_score_per_seed[seed_ind][rank]
                
                scores_data += [{
                    "seed": seed_fname,
                    "score": score,
                    "value": value,
                }]
            
        scores_df = pd.DataFrame(scores_data)
            
        roc_auc_micro = roc_auc_score(scores_df['value'], scores_df['score'])
        
        roc_auc_macro_df = (scores_df
            .groupby('seed')
            .apply(lambda group: roc_auc_score(group['value'], group['score']) if group['value'].nunique() > 1 else group['value'].unique()[0])
            .reset_index()
            .rename({0: 'roc_auc_macro'}, axis=1))
            
        roc_auc_macro = roc_auc_macro_df['roc_auc_macro'].mean()
            
        return roc_auc_macro, roc_auc_micro, roc_auc_macro_df