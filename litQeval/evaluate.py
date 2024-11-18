from sklearn.metrics.pairwise import cosine_similarity
from litQeval.eval_utils import get_data, evaluate_recall, fscore, mvee, is_inside_ellipse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import json

# Reevaluate the queries and update the results.xlsx file
if __name__ == "__main__":
    for query in tqdm(json.load(open("queries.json"))):
        baseline = query["baseline"]
        predicted = query["predicted"]
        data = get_data(baseline, predicted)
        core_pubs = data["core_pubs"]
        core_mean_embedding = data["core_mean_embedding"]
        baseline_pubs = data["baseline_pubs"]
        predicted_pubs = data["predicted_pubs"]
        baseline_vs = data["baseline_vs"]
        predicted_vs = data["predicted_vs"]
        core_vs = data["core_vs"]
        core_embeddigs = data["core_embeddings"]
        core_threshold = data["core_threshold"]
        core_embeddings = data["core_embeddings"]
        predicted_embeddings = np.array([embedding for embedding in predicted_vs.get(include=["embeddings"])["embeddings"]])
        baseline_embeddings = np.array([embedding for embedding in baseline_vs.get(include=["embeddings"])["embeddings"]])


        cosine_sim = cosine_similarity(core_mean_embedding, baseline_embeddings).flatten()
        baseline_pubs["similarity"] = cosine_sim
        core_pubs_in_baseline = baseline_pubs[baseline_pubs["id"].isin(core_pubs)]
        relevent_baseline_pubs = baseline_pubs[baseline_pubs["similarity"] >= core_threshold].copy()


        cosine_sim = cosine_similarity(core_mean_embedding, predicted_embeddings).flatten()
        predicted_pubs["similarity"] = cosine_sim
        core_pubs_in_predicted = predicted_pubs[predicted_pubs["id"].isin(core_pubs)]
        relevant_predicted_pubs = predicted_pubs[predicted_pubs["similarity"] >= core_threshold].copy()

        recall = evaluate_recall(core_pubs, baseline_pubs, predicted_pubs)
        pred_precision = relevant_predicted_pubs.shape[0] / predicted_pubs.shape[0] # total number of found publications
        baseline_precision = (relevent_baseline_pubs.shape[0] / baseline_pubs.shape[0]) if baseline_pubs.shape[0] > 0 else 0
        pred_f2 = fscore(pred_precision, recall["predicted_recall"], 2)
        baseline_f2 = fscore(baseline_precision, recall["baseline_recall"], 2)
        df = pd.DataFrame({
            "Semantic Precision": [pred_precision, baseline_precision],
            "Recall": [recall["predicted_recall"], recall["baseline_recall"]],
            "Semantic F2": [pred_f2, baseline_f2]
        }, index=["Predicted", "Baseline"])

        A, c = mvee(core_embeddings)
        base_is_inside = is_inside_ellipse(A, c, baseline_embeddings)
        predicted_is_inside = is_inside_ellipse(A, c, predicted_embeddings)

        mvve_prec_baseline = base_is_inside.sum() / len(base_is_inside)
        mvve_prec_predicted = predicted_is_inside.sum() / len(predicted_is_inside)
        mvve_df = pd.DataFrame({
            "MVVE Precision": [mvve_prec_predicted, mvve_prec_baseline],
            "Recall": [recall["predicted_recall"], recall["baseline_recall"]],
            "MVVE F2": [fscore(mvve_prec_predicted, recall["predicted_recall"], 2), fscore(mvve_prec_baseline, recall["baseline_recall"], 2)]
        }, index=["Predicted", "Baseline"])

        results = pd.DataFrame({
            "Query": [predicted] + [baseline],
            "Recall": [recall["predicted_recall"], recall["baseline_recall"]],
            "Semantic Precision": [pred_precision, baseline_precision],
            "Semantic F2": [pred_f2, baseline_f2],
            "MVVE Precision": [mvve_prec_predicted, mvve_prec_baseline],
            "MVVE F2": [fscore(mvve_prec_predicted, recall["predicted_recall"], 2), fscore(mvve_prec_baseline, recall["baseline_recall"], 2)]
        }, index=["Predicted", "Baseline"])

        try:
            old_results = pd.read_excel("results.xlsx", index_col=0)
            results = pd.concat([old_results, results]).drop_duplicates(subset=["Query"]).round(3)
            results.to_excel("results.xlsx")
        except FileNotFoundError:
            results.to_excel("results.xlsx")
