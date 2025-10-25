from flask import Flask, request, jsonify
from flask_cors import CORS
from src.repair_ranker import EnhancedRepairRanker
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Initialize ranker
ranker = EnhancedRepairRanker(
    data_path='data/raw/data.xls',
    hackathon_path='data/raw/hackathon.xls'
)
ranker.load_data(filter_clavero='FRE')
ranker.load_model()
ranker.create_clavero_embeddings()

def parse_numpy_like_string(s):
    if not isinstance(s, str):
        return s
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return [float(n) for n in numbers]

@app.route('/rank_repairs', methods=['POST'])
def rank_repairs():
    try:
        data = request.get_json(force=True)
        component_predictions = data.get('component_predictions', [])
        selected_component = data.get('selected_component', None)
        
        # Fix embeddings
        for item in component_predictions:
            item["embedding"] = parse_numpy_like_string(item.get("embedding", ""))
            item["embedding"] = np.array(item["embedding"])
        
        # Filter to only the selected component if provided
        if selected_component:
            component_predictions = [
                comp for comp in component_predictions 
                if comp.get('componentes', '').strip() == selected_component.strip()
            ]
            
            if not component_predictions:
                return jsonify({'error': f'Component "{selected_component}" not found in predictions'}), 404
        
        ranked_repairs = ranker.rank_repairs(
            component_predictions=component_predictions,
            top_k_components=1 if selected_component else 3,  # Only 1 if specific component selected
            top_k_repairs=10
        )
        
        return jsonify(ranked_repairs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)