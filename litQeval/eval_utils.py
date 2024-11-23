from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from scipy.spatial import Delaunay
from langchain_chroma import Chroma
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import plotly.express as px
import numpy.linalg as la
from pathlib import Path
import plotly.io as pio
from tqdm import tqdm
import pandas as pd
import numpy as np
import dimcli
import umap

pio.templates.default = "seaborn"
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
PLOT_CONFIGS = dict(
    title_x=0.5, title_font_size=30, title_font_family="Modern Computer", font_family="Modern Computer",
    xaxis_title="", yaxis_title="", showlegend=True, legend_title="",
    xaxis_tickfont_size=15, yaxis_tickfont_size=15, legend_font_size=20, legend_itemsizing="constant",
    legend_orientation="h", legend_yanchor="bottom", legend_y=-0.3, legend_xanchor="center", legend_x=0.5
)
try:
    EMBEDDING_MODEL = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000,
        check_embedding_ctx_length=False,
        show_progress_bar=True,
    )
except:
    EMBEDDING_MODEL = None


def recall(core: list, predicted: list) -> float:
    """
    Calculates the recall between the core and predicted publications

    Parameters
    ----------
    core: list
        The core publications
    predicted: list
        The predicted publications

    Returns
    -------
    recall: float
        The recall between the core and predicted publications
    """
    intersection = len(set(core).intersection(set(predicted)))
    return intersection / len(core)


def format_search_query(query: str, full_data: bool):
    """
    Formats a given boolean query to be used in the Dimensions API

    Parameters
    ----------
    query: str
        The query to be formatted
    full_data: bool
        Indicates if the full data should be searched instead of just the title and abstract

    Returns
    -------
    formatted_query: str
        The formatted query for the Dimensions API
    """
    query = query.replace('"', '\\"')
    if full_data:
        query = f"""search publications in full_data for "{query}" return publications[id+title+abstract]"""
    else:
        query = f"""search publications in title_abstract_only for "{query}" return publications[id+title+abstract]"""
    return query


def execute_query(query: str) -> pd.DataFrame:
    """
    Executes a given query using the Dimensions API

    Parameters
    ----------
    query: str
        The query to be executed

    Returns
    -------
    response: pd.DataFrame
        The response from the Dimensions API as a DataFrame containting the id, title, and abstract of the publications
    """
    dimcli.login()
    dsl = dimcli.Dsl()
    response = dsl.query_iterative(query)  # type: dimcli.DslDataset
    return response.as_dataframe()


def get_pubs(query: str, full_data: bool) -> pd.DataFrame:
    """
    Retrieve the publications for a given query

    Parameters
    ----------
    query: str
        The query to retrieve publications
    full_data: bool
        Indicates if the full data should be searched instead of just the title and abstract



    Returns
    -------
    results: pd.DataFrame
        The publications retrieved for the given query
    """
    query = format_search_query(query, full_data)
    results = execute_query(query)

    print(f"Total results: {len(results)}")
    # if too short
    results = results[results["abstract"].str.len() > 200]
    results = results[results["abstract"].str.split().str.len() > 50]

    # if too long shorten
    results["abstract"] = results["abstract"].apply(lambda x: x[:3000])
    print(f"Total results after filtering: {len(results)}")
    return results


def create_documents(pubs: pd.DataFrame) -> list:
    """
    Creates documents from the given publications

    Parameters
    ----------
    pubs: pd.DataFrame
        The publications to be converted to documents

    Returns
    -------
    documents: list
        The list of documents created from the publications
    """

    documents = []
    for i, row in pubs.iterrows():
        content = f'Title: {row["title"]}\nAbstract: {row["abstract"]}'
        doc = Document(page_content=content, metadata={"id": row["id"]})
        documents.append(doc)
    return documents


def embed_pubs(collection_name: str, path: str, pubs: pd.DataFrame) -> Chroma:
    """
    Embeds the publications using the OpenAI text-embedding-3-small model

    Parameters
    ----------
    pubs: pd.DataFrame
        The publications to be embedded

    Returns
    -------
    vector_store: FAISS
        The vector store containing the embeddings of the publications
    """
    documents = create_documents(pubs)
    vs = Chroma(collection_name, EMBEDDING_MODEL, persist_directory=path)
    chunk_ids_pair = [
        (
            documents[i: i + 5000],
            [doc.metadata["id"] for doc in documents[i: i + 5000]],
        )
        for i in range(0, len(documents), 5000)
    ]
    for chunk in tqdm(chunk_ids_pair):
        vs.add_documents(documents=chunk[0], ids=chunk[1])


def get_core_dataset(topic: str) -> tuple[list, np.ndarray]:
    """
    Retrieves the core dataset for a given topic

    Parameters
    ----------
    topic: str
        The topic to retrieve the core dataset for

    Returns
    -------
    core_pubs: pd.DataFrame
        The core publications for the given topic
    core_vs: Chroma
        The vector store containing the embeddings of the core publications
    """
    df = pd.read_excel("./data/core_publications.xlsx")
    df.rename(columns={"Pub_id": "id"}, inplace=True)
    core_vs = Chroma(
        "core_publications",
        EMBEDDING_MODEL,
        persist_directory="./data/vs/core_publications",
    )
    core_pubs = df[df["Topic"] == topic]
    return core_pubs, core_vs


def fetch_data(base_query: str, predicted_query: str, full_data=False) -> dict:
    """
    Retrieves and organizes publication and embedding data for a specified topic.

    Parameters
    ----------
    base_query : str
        The base query topic to retrieve data for.
    predicted_query : str
        The predicted query to retrieve data for.
    slr : bool, optional
        Indicates if the full data should be searched instead of just the title and abstract, by default False.

    Returns
    -------
    dict:
        "baseline_pubs": DataFrame containing the baseline publications
        "predicted_pubs": DataFrame containing the predicted publications
        "core_pubs": DataFrame containing the core publications
        "baseline_vs": Chroma object containing the baseline vector store
        "predicted_vs": Chroma object containing the predicted vector store
        "core_vs": Chroma object containing the core vector store
    """

    topic = base_query.replace('"', "")
    folder_name = topic.replace(" ", "_")
    text_folder = Path(f"./data/text/{folder_name}")
    vs_folder = Path(f"./data/vs/{folder_name}")

    text_folder.mkdir(parents=True, exist_ok=True)
    vs_folder.mkdir(parents=True, exist_ok=True)

    # Retrieve or load baseline publications
    baseline_path = text_folder / "baseline_pubs.csv"
    if baseline_path.exists():
        baseline_pubs = pd.read_csv(baseline_path)
    else:
        baseline_pubs = get_pubs(base_query, False)
        baseline_pubs.to_csv(baseline_path, index=False)

    # Retrieve or load predicted publications
    predicted_path = text_folder / "predicted_pubs.csv"
    if predicted_path.exists():
        predicted_pubs = pd.read_csv(predicted_path)
    else:
        predicted_pubs = get_pubs(predicted_query, full_data)
        predicted_pubs.to_csv(predicted_path, index=False)

    # Retrieve or compute baseline vector store embeddings
    baseline_vs_path = vs_folder / "baseline"
    if baseline_vs_path.exists():
        baseline_vs = Chroma(
            folder_name, EMBEDDING_MODEL, persist_directory=str(
                baseline_vs_path)
        )
    else:
        baseline_vs = embed_pubs(folder_name, str(
            baseline_vs_path), baseline_pubs)

    # Retrieve or compute predicted vector store embeddings
    predicted_vs_path = vs_folder / "predicted"
    if predicted_vs_path.exists():
        predicted_vs = Chroma(
            folder_name, EMBEDDING_MODEL, persist_directory=str(
                predicted_vs_path)
        )
    else:
        predicted_vs = embed_pubs(folder_name, str(
            predicted_vs_path), predicted_pubs)

    # Get core dataset publications and mean embedding
    core_pubs, core_vs = get_core_dataset(topic)

    return {
        "baseline_pubs": baseline_pubs,
        "predicted_pubs": predicted_pubs,
        "core_pubs": core_pubs,
        "baseline_vs": baseline_vs,
        "predicted_vs": predicted_vs,
        "core_vs": core_vs,
    }


def fetch_embeddings(topic: str,
                     baseline_pubs: pd.DataFrame,
                     predicted_pubs: pd.DataFrame,
                     core_pubs: pd.DataFrame,
                     baseline_vs: Chroma,
                     predicted_vs: Chroma,
                     core_vs: Chroma) -> dict:
    """
    Retrieves the embeddings for a given topic

    Parameters
    ----------
    topic: str
        The topic to retrieve the embeddings for
    baseline_pubs: pd.DataFrame
        The baseline publications
    predicted_pubs: pd.DataFrame
        The predicted publications
    core_pubs: pd.DataFrame
        The core publications
    baseline_vs: Chroma
        The vector store containing the embeddings of the baseline publications
    predicted_vs: Chroma
        The vector store containing the embeddings of the predicted publications
    core_vs: Chroma
        The vector store containing the embeddings of the core publications

    Returns
    -------
    embeddings: dict
        "baseline": The embeddings of the baseline publications
        "predicted": The embeddings of the predicted publications
        "core": The embeddings of the core publications
        "umap_embeddings": The UMAP embeddings of the core, baseline, and predicted publications
        "umap_core_embeddings": The UMAP embeddings of the core publications
        "core_mean_embedding": The mean embedding of the core publications
        "core_threshold": The cosine similarity threshold to the least similar core publication from the mean embedding
    """

    # Retrieve embeddings for baseline
    b_pub_ids = baseline_pubs["id"].tolist()
    baseline_embeddings = []
    for i in range(0, len(b_pub_ids), 25000):
        baseline_embeddings.append(baseline_vs.get(
            b_pub_ids[i:i+25000], include=["embeddings"])["embeddings"])
    baseline_embeddings = np.concatenate(baseline_embeddings)

    # Retrieve embeddings for predicted
    p_pub_ids = predicted_pubs["id"].tolist()
    predicted_embeddings = []
    for i in range(0, len(p_pub_ids), 25000):
        predicted_embeddings.append(predicted_vs.get(
            p_pub_ids[i:i+25000], include=["embeddings"])["embeddings"])
    predicted_embeddings = np.concatenate(predicted_embeddings)

    a = topic # lazy fix
    if a == "Sustainable Biofuel Economy":
        a = "Sustainable Biofuel"
    elif a == "Nanopharmaceuticals OR Nanonutraceuticals":
        a = "Nanoparticles"  

    core_embeddings = core_vs.get(core_pubs["id"].tolist(),
                                  where={"topic": a}, include=[
                                  "embeddings"])["embeddings"]

    embeddings = np.vstack([baseline_embeddings, predicted_embeddings])
    # UMAP embeddings
    umap_embeddings = umap.UMAP(metric="cosine").fit_transform(
        np.vstack([core_embeddings, baseline_embeddings, predicted_embeddings]))

    umap_core_embeddings = umap_embeddings[:len(core_embeddings)]
    umap_embeddings = umap_embeddings[len(core_embeddings):]

    core_mean_embedding = np.mean(core_embeddings, axis=0).reshape(1, -1)
    cos_threshold = cosine_similarity(
        core_mean_embedding, core_embeddings).flatten().min()

    core_in_baseline_embeddings = embeddings[:len(core_embeddings)]
    core_in_predicted_embeddings = embeddings[len(core_embeddings):]
    baseline_umap_embeddings = umap_embeddings[:len(baseline_embeddings)]
    predicted_umap_embeddings = umap_embeddings[len(baseline_embeddings):]
    baseline_in_core = baseline_pubs[baseline_pubs["id"].isin(
        core_pubs["id"])].index
    predicted_in_core = predicted_pubs[predicted_pubs["id"].isin(
        core_pubs["id"])].index
    core_in_baseline_umap_embeddings = umap_embeddings[baseline_in_core].copy()
    core_in_predicted_umap_embeddings = umap_embeddings[predicted_in_core].copy(
    )

    return {
        "embeddings": embeddings,
        "baseline_embeddings": baseline_embeddings,
        "predicted_embeddings": predicted_embeddings,
        "core_embeddings": core_embeddings,
        "umap_embeddings": umap_embeddings,
        "umap_core_embeddings": umap_core_embeddings,
        "core_mean_embedding": core_mean_embedding,
        "core_threshold": cos_threshold,
        "core_in_baseline_embeddings": core_in_baseline_embeddings,
        "core_in_predicted_embeddings": core_in_predicted_embeddings,
        "baseline_umap_embeddings": baseline_umap_embeddings,
        "predicted_umap_embeddings": predicted_umap_embeddings,
        "core_in_baseline_umap_embeddings": core_in_baseline_umap_embeddings,
        "core_in_predicted_umap_embeddings": core_in_predicted_umap_embeddings
    }


def get_evaluation_data(base_query: str, predicted_query: str = None, full_data=False) -> dict:
    """
    Retrieves the evaluation data for a given topic

    Parameters
    ----------
    base_query: str
        The base query topic to retrieve data for
    predicted_query: str
        The predicted query to retrieve data for
    full_data: bool, optional
        Indicates if the full data should be searched instead of just the title and abstract, by default False

    Returns
    -------
    evaluation_data: dict
        The evaluation data containing the publications, vector stores, and embeddings
    """
    data = fetch_data(base_query, predicted_query, full_data)
    baseline_pubs = data["baseline_pubs"]
    predicted_pubs = data["predicted_pubs"]
    embeddings = fetch_embeddings(
        base_query, baseline_pubs, predicted_pubs,
        data["core_pubs"], data["baseline_vs"],
        data["predicted_vs"], data["core_vs"],
    )

    df = pd.concat([baseline_pubs, predicted_pubs])
    df["Source"] = ["Baseline"] * \
        len(baseline_pubs) + ["Predicted"] * len(predicted_pubs)

    df["UMAP1"] = embeddings["umap_embeddings"][:, 0]
    df["UMAP2"] = embeddings["umap_embeddings"][:, 1]
    return {**data, **embeddings, "df": df}


def fscore(presicion: float, recall: float, n_pubs: int, beta: float = 1) -> float:
    """
    Calculates the F score given the precision and recall values weighted by beta, higher beta values give more weight to recall.

    Parameters
    ----------
    precision : float
        The precision value.
    recall : float
        The recall value.
    beta : float, optional
        The beta value to use in the F1 score calculation, by default 1.

    Returns
    -------
    float
        The F1 score value.
    """
    if recall == 0 or presicion == 0:
        return 0
    # decay function
    p = 2 # Controls the initial slowness of the decay. 
    q = 3 # Controls the speed-up near the end.
    threshold = 50000 # The maximum threshold for the decay
    decay = (1 - (n_pubs/threshold)**p)**q
    return round(((1 + beta**2) * (presicion * recall) / ((beta**2 * presicion) + recall))* decay, 2) 


def mvee(points, tol=0.0001):
    """
    Find the minimum volume ellipsoid that encloses a set of points

    Parameters
    ----------
    points : np.ndarray
        The points to find the minimum volume ellipsoid for.
    tol : float, optional
        The tolerance value for the algorithm, by default 0.0001.

    Returns
    -------
    A : np.ndarray
        The matrix representing the ellipsoid.
    """

    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c


def is_inside_ellipse(A, c, points):
    return np.array([((point - c) @ A @ (point - c).T) <= 1 for point in tqdm(points)])


def eval_cosine(df, source, core_mean_embedding,
                source_core_umap_embeddings, source_embeddings, core_pubs,
                topic, plot=False, threshold=0.7):
    df = df[df["Source"] == source].copy()
    df_relevant = df.copy()
    cosine_sim = cosine_similarity(
        core_mean_embedding, source_embeddings).flatten()
    df_relevant["similarity"] = cosine_sim
    df_relevant = df_relevant[df_relevant["similarity"] >= threshold]
    n_relevant = df_relevant.shape[0]
    found_cores = set(df_relevant["id"]).intersection(core_pubs["id"])
    print(f"The Core Publications in the {source} publications are {len(found_cores)}/{source_core_umap_embeddings.shape[0]}")
    print(f"The Semantically Relevant {source} publications are {n_relevant}/{len(source_embeddings)}")
    if plot:
        fig = go.Figure()
        fig.add_traces(
            [
                go.Scattergl(
                    x=df["UMAP1"],
                    y=df["UMAP2"],
                    mode="markers",
                    opacity=0.4,
                    marker=dict(color="gray", size=3),
                    showlegend=True,
                    name="Irrlevant"
                ),
                go.Scattergl(
                    x=df_relevant["UMAP1"],
                    y=df_relevant["UMAP2"],
                    mode="markers",
                    opacity=0.7,
                    marker=dict(color=COLORS[0], size=3),
                    showlegend=True,
                    name="Relevant"
                ),
                go.Scattergl(
                    x=source_core_umap_embeddings[:, 0],
                    y=source_core_umap_embeddings[:, 1],
                    mode="markers",
                    marker=dict(color="red", size=4),
                    showlegend=True,
                    name="Core Publications"
                ),
            ]
        )
        fig.update_layout(
            **PLOT_CONFIGS, title=f"Cosine Similarity: {topic} - {source}")
        fig.show()

    return df_relevant, found_cores


def eval_clustering(df: pd.DataFrame,
                    source: str,
                    source_embeddings: np.ndarray,
                    core_pubs: set,
                    topic: str,
                    plot=False,
                    threshold: float = 0.7) -> tuple[int, int, int]:
    """
    Find the best cluster for the given source and embeddings

    paramaters
    ----------
    df: pd.DataFrame
        The dataframe containing the publications
    source: str
        The source of the publications (Predicted or Baseline)
    embeddings: np.ndarray
        The embeddings of the source
    core_pubs: set
        The ids of the core publications
    topic: str
        The topic of the publications to be added to the title of the plot
    plot: bool
        Indicates if the plot should be displayed, by default False
    threshold: float
        The threshold for the number of core publications in the best cluster, by default 0.7

    returns
    -------
    best_k: int
        The best number of clusters
    pubs_in_cluster: int
        The number of publications in the best cluster
    core_in_cluster: int
        The number of core publications in the best cluster
    """

    best_k = None
    df_kmeans = df[df["Source"] == source].copy()
    for k in tqdm(iter(range(2, 100))):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(source_embeddings)
        df_kmeans["cluster"] = kmeans.labels_.astype(str)
        df_kmeans["core"] = df_kmeans["id"].isin(core_pubs).astype(int)
        df_kmeans["core"] = df_kmeans.groupby(
            "cluster")["core"].transform("sum")
        core_in_cluster = df_kmeans["core"].max()
        if core_in_cluster <= len(core_pubs) * threshold:
            best_k = k - 1
            break

    df_kmeans = df[df["Source"] == source].copy()
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(source_embeddings)
    df_kmeans["cluster"] = kmeans.labels_.astype(str)
    df_kmeans["core"] = df_kmeans["id"].isin(core_pubs).astype(int)
    df_kmeans["core"] = df_kmeans.groupby("cluster")["core"].transform("sum")
    cluster = df_kmeans.groupby("cluster")["core"].sum().idxmax()
    pubs_in_cluster = df_kmeans[df_kmeans["cluster"] == cluster]
    core_in_cluster = df_kmeans["core"].max()
    df_kmeans["cluster"] = df_kmeans["cluster"].replace(cluster, "BEST")
    print(f"Number of clusters: {best_k}, Threshold: {threshold}")
    print(
        f"Number of publications in the best cluster ({cluster}): {pubs_in_cluster.shape[0]}")
    print(
        f"Number of core publications in the best cluster: ({core_in_cluster}/{len(core_pubs)})")
    if plot:
        fig = px.scatter(
            df_kmeans,
            x="UMAP1",
            y="UMAP2",
            color="cluster",
            title=f"{topic} - {source}",
            labels={"cluster": "Cluster"},
            opacity=0.5,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**PLOT_CONFIGS, title=f"K-Means: {topic} - {source}")
        fig.show()

    return best_k, pubs_in_cluster, core_in_cluster


def eval_mvee(df: pd.DataFrame,
              source: str,
              source_umap_core_embeddings: np.ndarray,
              source_umap_embeddings: np.ndarray,
              topic: str,
              plot=False):
    A, c = mvee(source_umap_core_embeddings)
    is_inside = is_inside_ellipse(A, c, source_umap_embeddings)
    mvee_df = df[df["Source"] == source].copy()
    mvee_df["is_inside_mvee"] = is_inside
    mvee_df["is_inside_mvee"] = mvee_df["is_inside_mvee"].replace(
        {True: "Inside", False: "Outside"})
    mvee_is_inside = mvee_df[mvee_df["is_inside_mvee"] == "Inside"]
    total = len(mvee_df)

    print(
        f"Number of relevant {source} publications (MVEE): {mvee_is_inside.shape[0]} / {total}")
    if plot:
        fig = px.scatter(mvee_df, x="UMAP1", y="UMAP2", opacity=0.5,
                         title=f"Publications inside the MVEE ({source})")
        fig.update_traces(marker=dict(size=4))
        fig.add_traces(
            [

                go.Scattergl(
                    x=mvee_df[mvee_df["is_inside_mvee"] == "Outside"]["UMAP1"],
                    y=mvee_df[mvee_df["is_inside_mvee"] == "Outside"]["UMAP2"],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(color='gray', size=3),
                    showlegend=True,
                    name="Irrelevant"
                ),
                go.Scattergl(
                    x=mvee_df[mvee_df["is_inside_mvee"] == "Inside"]["UMAP1"],
                    y=mvee_df[mvee_df["is_inside_mvee"] == "Inside"]["UMAP2"],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(color='#2ca02c', size=3),
                    showlegend=True,
                    name="Relevant"
                ),
                go.Scattergl(
                    x=source_umap_core_embeddings[:, 0],
                    y=source_umap_core_embeddings[:, 1],
                    mode="markers",
                    opacity=0.8,
                    marker=dict(color="red", size=4),
                    showlegend=True,
                    name="Core Publications"
                )
            ]
        )

        fig.update_layout(**PLOT_CONFIGS, title=f"MVEE: {topic} - {source}")
        fig.show()

    return mvee_is_inside


def eval_hull(df: pd.DataFrame,
              source: str,
              umap_core_embeddings: np.ndarray,
              umap_embeddings: np.ndarray,
              topic: str,
              plot=False):

    hu_df = df[df["Source"] == source].copy()
    delu = Delaunay(umap_core_embeddings)
    is_inside = delu.find_simplex(umap_embeddings) >= 0
    hu_df["is_inside_hull"] = is_inside
    hu_df["is_inside_hull"] = hu_df["is_inside_hull"].replace(
        {True: "Inside", False: "Outside"})
    hull_is_inside = hu_df[hu_df["is_inside_hull"] == "Inside"]
    total = len(hu_df)

    print(
        f"Number of relevant {source} publications (Hull): {hull_is_inside.shape[0]} / {total}")
    if plot:
        fig = px.scatter(hu_df, x="UMAP1", y="UMAP2", opacity=0.5,
                         title=f"Publications inside the Hull ({source})")
        fig.update_traces(marker=dict(size=4))
        fig.add_traces(
            [
                go.Scattergl(
                    x=hu_df[hu_df["is_inside_hull"] == "Outside"]["UMAP1"],
                    y=hu_df[hu_df["is_inside_hull"] == "Outside"]["UMAP2"],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(color='gray', size=3),
                    showlegend=True,
                    name="Irrelevant"
                ),

                go.Scattergl(
                    x=hu_df[hu_df["is_inside_hull"] == "Inside"]["UMAP1"],
                    y=hu_df[hu_df["is_inside_hull"] == "Inside"]["UMAP2"],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(color='#2ca02c', size=3),
                    showlegend=True,
                    name="Relevant"
                ),
                go.Scattergl(
                    x=umap_core_embeddings[:, 0],
                    y=umap_core_embeddings[:, 1],
                    mode="markers",
                    opacity=0.8,
                    marker=dict(color="red", size=4),
                    showlegend=True,
                    name="Core Publications"
                )
            ]
        )

        fig.update_layout(**PLOT_CONFIGS, title=f"Hull: {topic} - {source}")
        fig.show()

    return hull_is_inside