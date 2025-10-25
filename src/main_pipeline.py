from src.repair_ranker import EnhancedRepairRanker
import numpy as np
import requests
import re

# ============================================================================
# STEP 1: Initialize and load data
# ============================================================================
print("="*80)
print("ENHANCED REPAIR RANKING SYSTEM")
print("="*80 + "\n")

ranker = EnhancedRepairRanker(
    data_path='data/raw/data.xls',
    hackathon_path='data/raw/hackathon.xls'
)

# Load data (optionally filter by Clavero, e.g., 'FRE')
ranker.load_data(filter_clavero='FRE')

# ============================================================================
# STEP 2: Load BERT model and create embeddings
# ============================================================================
ranker.load_model()
ranker.create_clavero_embeddings()

# ============================================================================
# STEP 3: Input from previous module
# ============================================================================
# This is the output from your previous BERT component predictor module
# Format: [{'componente': 'llave', 'probabilidad': 0.65, 'embedding': [...]}, ...]
# Note: The embedding is the same for all components

url = "http://localhost:5001/predict"
data = {"description": "broken pipe in hydraulic system"}

component_predictions = requests.post(url, json=data).json()

def parse_numpy_like_string(s):
    """Convert string like '[[ -3.82432252e-01  9.51278955e-02 ...]]' → list of floats."""
    if not isinstance(s, str):
        return s
    # Extract all numbers including decimals, signs, and scientific notation
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return [float(n) for n in numbers]

# Fix embeddings in all items
for item in component_predictions:
    item["embedding"] = parse_numpy_like_string(item.get("embedding", ""))

# (Optional) convert each embedding list to a NumPy array if you’ll use PCA/scikit-learn
for item in component_predictions:
    item["embedding"] = np.array(item["embedding"])

# print(len(component_predictions))
print(component_predictions)

# # Load JSON file from previous module
# component_predictions = [
#     {
#         "componente": "ZAPATA FRENO",
#         "probabilidad": 0.75,
#         "embedding": np.random.rand(768).tolist()
#     },
#     {
#         "componente": "PORTAZAPATA",
#         "probabilidad": 0.65,
#         "embedding": np.random.rand(768).tolist()
#     }
#     ]

# with open('output/component_predictions.json', 'r') as f:
#     component_predictions = json.load(f)


# Or if you have it directly as a variable:
# component_predictions = [
#     {'componente': 'ZAPATA FRENO', 'probabilidad': 0.75, 'embedding': [0.1, 0.2, ...]},
#     {'componente': 'PORTAZAPATA', 'probabilidad': 0.65, 'embedding': [0.1, 0.2, ...]},
#     {'componente': 'DISCO FRENO', 'probabilidad': 0.45, 'embedding': [0.1, 0.2, ...]},
# ]
# (Note: all 'embedding' values are the same - the query embedding from symptoms)

# ============================================================================
# STEP 4: Rank repairs
# ============================================================================
print("="*80)
print("RANKING REPAIRS...")
print("="*80)

ranked_repairs = ranker.rank_repairs(
    component_predictions=component_predictions,
    top_k_components=3,  # Consider top 3 components
    top_k_repairs=10      # Return top 10 repairs
)

# ============================================================================
# STEP 5: Display and save results
# ============================================================================
ranker.display_results(ranked_repairs)
ranker.save_results(ranked_repairs, output_path='output/ranked_repairs.csv')

print("="*80)
print("DONE!")
print("="*80)