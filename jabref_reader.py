import bibtexparser
import pandas as pd



def data_to_df():
    GROUPS = {
        "AI on Edge Devices": ["", [], []],  # Artificial Intelligence
        "Synthetic Biology": ["", [], []],  # Biology
        "Resilience in Business and management": ["", [], []],  # Management
        "Cervical Myelopathy": ["", [], []],  # Medicine
        "Drones in Agriculture": ["", [], []],  # Agriculture
        "Crop Yield Prediction": ["", [], []],  # Agriculture
        "Robotic Arthroplasty": ["", [], []],  # Robots
        "Soft Robotics": ["", [], []],  # Robots
        "Tourism Growth Nexus": ["", [], []],  # Economics
        "Energy Growth Nexus": ["", [], []],  # Economics
        "Perovskite Solar Cells Stability": ["", [], []],  # Renewable Energy
        "Sustainable Biofuel": ["", [], []],  # Renewable Energy
        "Nanoparticles": ["", [], []],  # Nanotechnology
        "Green Warehousing": ["", [], []],  # Climate Science
        "Internet of Things in Healthcare": ["", [], []],  # Internet of Things
    }
    with open("eval_datasets.bib", encoding="utf-8") as bibtex_file:
        bibtex_database = bibtexparser.load(bibtex_file)
        for entry in bibtex_database.entries:
            group = entry["groups"]
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

df = data_to_df()
df.to_excel("eval_datasets.xlsx", index=False)