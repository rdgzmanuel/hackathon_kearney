import pandas as pd
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class EnhancedRepairRanker:
    """
    Enhanced repair ranking system that combines:
    - Component probabilities from previous module
    - Repair frequency from historical data
    - Semantic similarity of symptoms using embeddings
    """
    
    def __init__(self, data_path='data/raw/data.xls', hackathon_path='data/raw/hackathon.xls'):
        """
        Initialize the ranker.
        
        Args:
            data_path: Path to data.xls (component-repair mappings)
            hackathon_path: Path to hackathon.xls (historical symptoms)
        """
        self.data_path = data_path
        self.hackathon_path = hackathon_path
        self.model = None
        
        # Data structures
        self.component_repairs = defaultdict(lambda: defaultdict(int))
        self.component_totals = defaultdict(int)
        self.repair_claveros = defaultdict(set)  # repair -> set of claveros
        self.clavero_symptoms = defaultdict(list)  # clavero -> list of symptoms
        self.clavero_embeddings = {}  # clavero -> embedding
        
        self.data_df = None
        self.hackathon_df = None
        
    def load_data(self, filter_clavero=None):
        """
        Load both Excel files and build data structures.
        
        Args:
            filter_clavero: Optional filter for Clavero column (e.g., 'FRE')
        """
        print("Loading data.xls...")
        try:
            self.data_df = pd.read_excel(self.data_path, engine='openpyxl')
        except:
            self.data_df = pd.read_excel(self.data_path)
        
        print(f"✓ Loaded {len(self.data_df)} rows from data.xls")
        
        # Apply filter if specified
        if filter_clavero:
            if 'Clavero' in self.data_df.columns:
                self.data_df['Clavero'] = self.data_df['Clavero'].astype(str)
                original_len = len(self.data_df)
                self.data_df = self.data_df[self.data_df['Clavero'].str.startswith(filter_clavero, na=False)]
                print(f"✓ Filtered by Clavero '{filter_clavero}': {len(self.data_df)} rows")
        
        # Clean data.xls
        self.data_df = self.data_df[['Clavero', 'Descripción componente', 'DEFINICION']].copy()
        self.data_df = self.data_df.dropna(subset=['DEFINICION'])
        self.data_df['Clavero'] = self.data_df['Clavero'].astype(str).str.strip()
        self.data_df['Descripción componente'] = self.data_df['Descripción componente'].astype(str).str.strip()
        self.data_df['DEFINICION'] = self.data_df['DEFINICION'].astype(str).str.strip()
        
        # Build component-repair mappings
        for _, row in self.data_df.iterrows():
            component = row['Descripción componente']
            repair = row['DEFINICION']
            clavero = row['Clavero']
            
            if component and repair and component != 'nan':
                self.component_repairs[component][repair] += 1
                self.component_totals[component] += 1
                self.repair_claveros[repair].add(clavero)
        
        print(f"✓ Built mappings for {len(self.component_totals)} components\n")
        
        # Load hackathon.xls
        print("Loading hackathon.xls...")
        try:
            self.hackathon_df = pd.read_excel(self.hackathon_path, engine='openpyxl')
        except:
            self.hackathon_df = pd.read_excel(self.hackathon_path)
        
        print(f"✓ Loaded {len(self.hackathon_df)} rows from hackathon.xls")
        
        # Check for required columns
        required_cols = ['clavero', 'descripcion_ot']
        missing = [col for col in required_cols if col not in self.hackathon_df.columns]
        if missing:
            # Try case-insensitive match
            col_map = {col.lower(): col for col in self.hackathon_df.columns}
            if 'clavero' in col_map:
                self.hackathon_df.rename(columns={col_map['clavero']: 'clavero'}, inplace=True)
            if 'descripcion_ot' in col_map:
                self.hackathon_df.rename(columns={col_map['descripcion_ot']: 'descripcion_ot'}, inplace=True)
        
        # Clean hackathon.xls
        self.hackathon_df['clavero'] = self.hackathon_df['clavero'].astype(str).str.strip()
        self.hackathon_df['descripcion_ot'] = self.hackathon_df['descripcion_ot'].fillna('').astype(str).str.strip()
        
        # Build clavero-symptoms mappings
        for _, row in self.hackathon_df.iterrows():
            clavero = row['clavero']
            symptom = row['descripcion_ot']
            
            if clavero and symptom and clavero != 'nan':
                self.clavero_symptoms[clavero].append(symptom)
        
        print(f"✓ Built symptom mappings for {len(self.clavero_symptoms)} claveros\n")
        
    def load_model(self, model_name='google-bert/bert-base-uncased'):
        """Load BERT model for embeddings."""
        print(f"Loading BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded!\n")
        
    def create_clavero_embeddings(self):
        """
        Create average embeddings for each clavero based on its symptoms.
        """
        if self.model is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        print("Creating embeddings for clavero symptoms...")
        
        for clavero, symptoms in self.clavero_symptoms.items():
            if symptoms:
                # Create embeddings for all symptoms of this clavero
                embeddings = self.model.encode(symptoms, show_progress_bar=False)
                # Average them
                avg_embedding = np.mean(embeddings, axis=0)
                self.clavero_embeddings[clavero] = avg_embedding
        
        print(f"✓ Created embeddings for {len(self.clavero_embeddings)} claveros\n")
        
    def rank_repairs(self, component_predictions, top_k_components=3, top_k_repairs=10):
        """
        Rank repairs based on component probabilities, repair frequency, and semantic similarity.
        
        Args:
            component_predictions: List of dicts with 'componente', 'probabilidad', 'embedding'
                Example: [{'componente': 'llave', 'probabilidad': 0.65, 'embedding': [...]}, ...]
                Note: The embedding is the same for all components (query embedding from symptoms)
            top_k_components: Number of top components to consider
            top_k_repairs: Number of top repairs to return
            
        Returns:
            List of dicts with repair info and final scores
        """
        # Extract query embedding (same for all components)
        query_embedding = component_predictions[0]['embedding']
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Take top K components
        top_components = sorted(component_predictions, key=lambda x: x['probabilidad'], reverse=True)[:top_k_components]
        
        print(f"Processing top {len(top_components)} components...")
        
        all_repairs = []
        
        for comp_pred in top_components:
            component = comp_pred['componentes']
            comp_prob = comp_pred['probabilidad']
            
            print(f"\n📦 Component: {component} (prob: {comp_prob:.2%})")
            
            # Get repairs for this component
            if component not in self.component_repairs:
                print(f"   ⚠ No repairs found for this component")
                continue
            
            repairs = self.component_repairs[component]
            total_repairs = self.component_totals[component]
            
            for repair, count in repairs.items():
                # Calculate repair frequency probability
                repair_freq_prob = count / total_repairs
                
                # Get claveros associated with this repair
                claveros = self.repair_claveros[repair]
                
                # Calculate semantic similarity for each clavero
                semantic_scores = []
                for clavero in claveros:
                    if clavero in self.clavero_embeddings:
                        clavero_emb = self.clavero_embeddings[clavero].reshape(1, -1)
                        similarity = cosine_similarity(query_embedding, clavero_emb)[0][0]
                        semantic_scores.append(similarity)
                
                # Average semantic similarity across all claveros
                avg_semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0
                
                # Final score: weighted combination
                final_score = (comp_prob * 0.4) + (repair_freq_prob * 0.3) + (avg_semantic_score * 0.3)
                
                all_repairs.append({
                    'component': component,
                    'component_probability': comp_prob,
                    'repair': repair,
                    'repair_frequency': repair_freq_prob,
                    'semantic_similarity': avg_semantic_score,
                    'final_score': final_score,
                    'claveros': list(claveros),
                    'repair_count': count
                })
        
        # Sort by final score
        all_repairs.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(all_repairs)
        # Return top K repairs
        return all_repairs[:top_k_repairs]
    
    def display_results(self, ranked_repairs):
        """
        Display ranked repairs in a nice format.
        
        Args:
            ranked_repairs: List of repair dicts from rank_repairs()
        """
        print("\n" + "╔" + "═"*98 + "╗")
        print("║" + " "*98 + "║")
        print("║" + "🏆 RANKED REPAIR RECOMMENDATIONS".center(98) + "║")
        print("║" + " "*98 + "║")
        print("╠" + "═"*98 + "╣")
        
        for idx, repair in enumerate(ranked_repairs, 1):
            final_score = repair['final_score']
            
            # Create score bar
            bar_length = int(final_score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print("║" + " "*98 + "║")
            print(f"║ #{idx} │ {final_score*100:>6.2f}% │ {bar} ║")
            print("║" + " "*98 + "║")
            
            # Component info
            comp = repair['component'][:40]
            print(f"║ 📦 Component: {comp:<85}║")
            
            # Score breakdown
            print(f"║    └─ Comp Prob: {repair['component_probability']*100:>5.1f}% | "
                  f"Freq: {repair['repair_frequency']*100:>5.1f}% | "
                  f"Semantic: {repair['semantic_similarity']*100:>5.1f}%{' '*21}║")
            
            # Repair description
            repair_text = repair['repair'][:85]
            print(f"║ 🔧 Repair: {repair_text:<87}║")
            
            # Claveros
            claveros_str = ', '.join(repair['claveros'][:3])
            if len(repair['claveros']) > 3:
                claveros_str += f" (+{len(repair['claveros'])-3} more)"
            claveros_str = claveros_str[:85]
            print(f"║    Claveros: {claveros_str:<87}║")
            
            if idx < len(ranked_repairs):
                print("║" + "─"*98 + "║")
        
        print("║" + " "*98 + "║")
        print("╚" + "═"*98 + "╝\n")
    
    def save_results(self, ranked_repairs, output_path='output/ranked_repairs.csv'):
        """
        Save ranked repairs to CSV.
        
        Args:
            ranked_repairs: List of repair dicts
            output_path: Path to save CSV
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not ranked_repairs:
            print("⚠ No repairs to save!")
            return
        
        df = pd.DataFrame(ranked_repairs)
        
        # Convert claveros list to string if column exists
        if 'claveros' in df.columns:
            df['claveros'] = df['claveros'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"💾 Results saved to: {output_path}\n")