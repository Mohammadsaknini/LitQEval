from litQeval.eval_utils import (
        get_evaluation_data,
        eval_cosine,
        eval_clustering, 
        eval_hull,
        eval_mvee,
        fscore
        ) 
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import json

# Reevaluate the queries and update the results.xlsx file
if __name__ == "__main__":
    for query in tqdm(json.load(open("data/queries.json"))):
        topic = query["baseline"]
        predicted = query["predicted"]
        data = get_evaluation_data(topic)
        baseline_pubs = data["baseline_pubs"]
        predicted_pubs = data["predicted_pubs"]
        core_pubs = data["core_pubs"]

        baseline_vs = data["baseline_vs"]
        predicted_vs = data["predicted_vs"]
        core_vs = data["core_vs"]
        threshold = data["core_threshold"]

        umap_embeddings = data["umap_embeddings"]
        umap_core_embeddings = data["umap_core_embeddings"]
        core_mean_embedding = data["core_mean_embedding"]

        embeddings = data["embeddings"]
        baseline_embeddings = data["baseline_embeddings"]
        predicted_embeddings = data["predicted_embeddings"]
        core_embeddings = data["core_embeddings"]

        baseline_umap_embeddings = data["baseline_umap_embeddings"]
        predicted_umap_embeddings = data["predicted_umap_embeddings"]
        baseline_core_umap_embeddings = data["baseline_core_umap_embeddings"]
        predicted_core_umap_embeddings = data["predicted_core_umap_embeddings"]

        baseline_cp = set(baseline_pubs["id"]).intersection(core_pubs["id"])
        predicted_cp = set(predicted_pubs["id"]).intersection(core_pubs["id"])
        df = data["df"]


        cos_baseline_relevant, cos_baseline_cores = eval_cosine(
            df, "Baseline", core_mean_embedding, baseline_core_umap_embeddings, baseline_embeddings, core_pubs, topic)
        cos_predicted_relevant, cos_predicted_cores = eval_cosine(
            df, "Predicted", core_mean_embedding, predicted_core_umap_embeddings, predicted_embeddings, core_pubs, topic)

        cluster_baseline_k, cluster_baseline_relevant, cluster_baseline_core = eval_clustering(
            df, "Baseline", baseline_embeddings, baseline_cp, topic)
        cluster_predicted_k, cluster_predicted_relevant, cluster_predicted_core = eval_clustering(
            df, "Predicted", predicted_embeddings, predicted_cp, topic)
        
        if baseline_core_umap_embeddings.size != 0 and baseline_core_umap_embeddings.shape[0] >= 3:
            mvee_baseline_relevant = eval_mvee(df, "Baseline", baseline_core_umap_embeddings, baseline_umap_embeddings, topic, False)
        else:
            mvee_baseline_relevant = pd.DataFrame({"id": []})
        if predicted_core_umap_embeddings.size != 0 and predicted_core_umap_embeddings.shape[0] >= 3:
            mvee_predicted_relevant = eval_mvee(df, "Predicted", predicted_core_umap_embeddings, predicted_umap_embeddings, topic, False)
        else:
            mvee_predicted_relevant = pd.DataFrame({"id": []})

        if baseline_core_umap_embeddings.size != 0 and baseline_core_umap_embeddings.shape[0] >= 3:
            hull_baseline_relevant = eval_hull(df, "Baseline", baseline_core_umap_embeddings, baseline_umap_embeddings, topic, False)
        else:
            hull_baseline_relevant = pd.DataFrame({"id": []})
        if predicted_core_umap_embeddings.size != 0 and predicted_core_umap_embeddings.shape[0] >= 3:
            hull_predicted_relevant = eval_hull(df, "Predicted", predicted_core_umap_embeddings, predicted_umap_embeddings, topic, False)
        else:
            hull_predicted_relevant = pd.DataFrame({"id": []})

        beta = 2
        cos_baseline_precision = cos_baseline_relevant.shape[0] / len(baseline_pubs)
        cos_baseline_recall = len(cos_baseline_cores) / len(core_pubs)
        cos_baseline_f2 = fscore(cos_baseline_precision, cos_baseline_recall, cos_baseline_relevant.shape[0], beta)
        cos_predicted_precision = cos_predicted_relevant.shape[0] / len(predicted_pubs)
        cos_predicted_recall = len(cos_predicted_cores) / len(core_pubs)
        cos_predicted_f2 = fscore(cos_predicted_precision, cos_predicted_recall, cos_predicted_relevant.shape[0], beta)

        cluster_baseline_precision = cluster_baseline_relevant.shape[0] / len(baseline_pubs)
        cluster_baseline_recall = cluster_baseline_core / len(core_pubs)
        cluster_baseline_f2 = fscore(cluster_baseline_precision, cluster_baseline_recall, cluster_baseline_relevant.shape[0], beta)
        cluster_predicted_precision = cluster_predicted_relevant.shape[0] / len(predicted_pubs)
        cluster_predicted_recall = cluster_predicted_core / len(core_pubs)
        cluster_predicted_f2 = fscore(cluster_predicted_precision, cluster_predicted_recall, cluster_predicted_relevant.shape[0], beta)

        mvee_baseline_precision = mvee_baseline_relevant.shape[0] / len(baseline_pubs)
        mvee_baseline_recall = len(baseline_cp) / len(core_pubs)
        mvee_baseline_f2 = fscore(mvee_baseline_precision, mvee_baseline_recall, mvee_baseline_relevant.shape[0], beta)
        mvee_predicted_precision = mvee_predicted_relevant.shape[0] / len(predicted_pubs)
        mvee_predicted_recall = len(predicted_cp) / len(core_pubs)
        mvee_predicted_f2 = fscore(mvee_predicted_precision, mvee_predicted_recall, mvee_predicted_relevant.shape[0], beta)

        hull_baseline_precision = hull_baseline_relevant.shape[0] / len(baseline_pubs)
        hull_baseline_recall = len(baseline_cp) / len(core_pubs)
        hull_baseline_f2 = fscore(hull_baseline_precision, hull_baseline_recall, hull_baseline_relevant.shape[0], beta)
        hull_predicted_precision = hull_predicted_relevant.shape[0] / len(predicted_pubs)
        hull_predicted_recall = len(predicted_cp) / len(core_pubs)
        hull_predicted_f2 = fscore(hull_predicted_precision, hull_predicted_recall, hull_predicted_relevant.shape[0], beta)
        
        results = pd.DataFrame({
            "Query": [predicted, topic],
            "Recall": [cos_predicted_recall, cos_baseline_recall],
            "Cosine Precision": [cos_predicted_precision, cos_baseline_precision],
            "Cosine Relevant": [cos_predicted_relevant.shape[0], cos_baseline_relevant.shape[0]],
            "Cosine F2": [cos_predicted_f2, cos_baseline_f2],
            "Cluster Precision": [cluster_predicted_precision, cluster_baseline_precision],
            "Cluster Relevant": [cluster_predicted_relevant.shape[0], cluster_baseline_relevant.shape[0]],
            "Cluster F2": [cluster_predicted_f2, cluster_baseline_f2],
            "MVEE Precision": [mvee_predicted_precision, mvee_baseline_precision],
            "MVEE Relevant": [mvee_predicted_relevant.shape[0], mvee_baseline_relevant.shape[0]],
            "MVEE F2": [mvee_predicted_f2, mvee_baseline_f2],
            "Hull Precision": [hull_predicted_precision, hull_baseline_precision],
            "Hull Relevant": [hull_predicted_relevant.shape[0], hull_baseline_relevant.shape[0]],
            "Hull F2": [hull_predicted_f2, hull_baseline_f2]
        }, index=["Predicted", "Baseline"])
        
        try:
            old_results = pd.read_excel("results.xlsx", index_col=0)
            results = pd.concat([old_results, results]).drop_duplicates(subset=["Query"]).round(3)
            results.to_excel("results.xlsx")
        except FileNotFoundError:
            results.to_excel("results.xlsx")
