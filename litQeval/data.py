import bibtexparser
import pandas as pd
import numpy as np
import dimcli
import json
import openai

class NumpyEncoder(json.JSONEncoder):
    """
    Converts a numpy array to a json object:
    >>> array = np.zeros(10)
    >>> array_as_json = json.dumps(a,cls=NumpyEncoder)
    """

    def default(self, obj) -> dict:
        """
        If input object is an ndarray it will be converted into a dict holding dtype, shape and the data, base64 encoded.

        paramters
        ---------
        obj: object
            object to be converted into a json object

        returns
        -------
        dict
            dictionary holding the dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def jabref_to_excel() -> pd.DataFrame:
    """
    Read the JabRef file and extract the core publications for each group

    Returns
    -------
    df: pd.DataFrame
        The DataFrame containing the core publications for each group
    """
    GROUPS = {
    }
    with open("./data/publications.bib", encoding="utf-8") as bibtex_file:
        bibtex_database = bibtexparser.load(bibtex_file)
        for entry in bibtex_database.entries:
            group = entry["groups"]
            if group not in GROUPS:
                # [pub_id, [core_pub_ids], [core_pub_titles]]
                GROUPS[group] = ["", [], []]

            pub_id = entry["url"].split("/")[-1]
            title = entry["title"]
            if "priority" in entry:
                GROUPS[group][0] = pub_id
            else:
                GROUPS[group][1].append(pub_id)
                GROUPS[group][2].append(title)

    df = pd.DataFrame(GROUPS).T
    df.reset_index(inplace=True)
    df.rename(columns={0: "Survey", 1: "Core Publications", 2: "Title", "index": "Group"}, inplace=True)
    df = df.explode(["Core Publications", "Title"], ignore_index=True)
    return df

def get_query(attributes: str, pub_ids: str):
    """
    Generate the query for the Dimensions API

    Parameters
    ----------
    attributes: list
        The list of attributes to extract
    publication_ids: list
        A list valid dimensions publication ids

    Returns
    -------
    query: str
        The generated query
    """
    
    pub_ids = json.dumps(pub_ids, cls=NumpyEncoder)
    attributes = "+".join(attributes)
    query = f"""search publications where id in {pub_ids} return publications[{attributes}]"""
    return query

def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    dimcli.login()
    dsl = dimcli.Dsl()
    pub_ids = list(set(df["Core Publications"].tolist() + df["Survey"].tolist()))
    chunks = []
    data = []
    attirbutes = ["id","title", "doi", "abstract", "times_cited", "field_citation_ratio", "year"]
    if len(pub_ids) > 300:
        for i in range(0, len(pub_ids), 300):
            chunks.append(pub_ids[i:i+300])
    else:
        chunks.append(pub_ids)

    for chunk in chunks:
        query = get_query(attirbutes, chunk)
        response = dsl.query_iterative(query) # type: dimcli.DslDataset
        data += response.publications

    return pd.DataFrame(data)[attirbutes]

def download():
    df = jabref_to_excel()
    metadata = extract_metadata(df)
    metadata.to_excel("./data/metadata.xlsx", index=False)
    df.to_excel("./data/core_publications.xlsx", index=False)
    
