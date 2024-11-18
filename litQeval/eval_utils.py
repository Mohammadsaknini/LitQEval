from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_chroma import Chroma
import numpy.linalg as la
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import dimcli

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
    group_embeddings: list
        The list of core publications for the given topic
    embeddings: np.NDArray
        The embeddings of the core publications for the given topic
    """
    df = pd.read_excel("./data/core_publications.xlsx")
    collection = Chroma(
        "core_publications",
        EMBEDDING_MODEL,
        persist_directory="./data/vs/core_publications",
    )
    core_pubs = df[df["Topic"] == topic]["Pub_id"]
    embeddings = []
    for i in core_pubs.values:
        embeddings.append(collection.get(
            i, include=["embeddings"])["embeddings"])

    return core_pubs.tolist(), np.squeeze(embeddings)


def get_data(base_query: str, predicted_query: str, full_data=False) -> dict:
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
    dict
        A dictionary containing core publications, mean embedding, baseline
        and predicted publication datasets, as well as their embeddings.
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
    core_pubs, core_embeddings = get_core_dataset(topic)
    core_vs = Chroma("core_publications", EMBEDDING_MODEL,
                     persist_directory="./data/vs/core_publications")

    core_mean_embedding = np.mean(core_embeddings, axis=0).reshape(1, -1)
    cos_threshold = cosine_similarity(core_mean_embedding, core_embeddings).flatten().min()
    return {
        "baseline_pubs": baseline_pubs,
        "predicted_pubs": predicted_pubs,
        "baseline_vs": baseline_vs,
        "predicted_vs": predicted_vs,
        "core_vs": core_vs,
        "core_pubs": core_pubs,
        "core_mean_embedding": core_mean_embedding,
        "core_embeddings": core_embeddings,
        # The threshold to the least similar core publication from the mean embedding
        "core_threshold": cos_threshold, 
    }


def evaluate_recall(
    core_pubs: list[str], baseline_pubs: list[str], predicted_pubs: list[str]
) -> dict:
    """
    Evaluates the recall between the core publications and the baseline and predicted publications

    Parameters
    ----------
    core_pubs: list[str]
        The core publications
    baseline_pubs: list[str]
        The baseline publications
    predicted_pubs: list[str]
        The predicted publications

    Returns
    -------
    evaluation: dict
        The evaluation results containing the baseline and predicted recall
    """

    # Calculate the recall between core and baseline
    baseline_recall = recall(core_pubs, baseline_pubs["id"].tolist())
    predicted_recall = recall(core_pubs, predicted_pubs["id"].tolist())

    return {
        "baseline_recall": baseline_recall,
        "predicted_recall": predicted_recall,
    }


def fscore(presicion: float, recall: float, beta: float = 1) -> float:
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
    return (1 + beta**2) * (presicion * recall) / ((beta**2 * presicion) + recall)


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

# point_wise


def is_inside_ellipse(A, c, points):
    return np.array([((point - c) @ A @ (point - c).T) <= 1 for point in tqdm(points)])

# if we have enough memory


def is_inside_ellipse_v2(A, c, points):
    shifted_points = points - c
    return np.diag((shifted_points @ A) @ shifted_points.T <= 1)
