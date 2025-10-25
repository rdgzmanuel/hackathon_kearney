import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SemanticSearchRepairs:
    def __init__(self, excel_path='data/raw/hackathon.xls'):
        """
        Initialize the semantic search system
        
        Args:
            excel_path: Path to the Excel file with maintenance data
        """
        self.excel_path = excel_path
        self.model = None
        self.data = None
        self.embeddings = None
        
    def load_data(self):
        """Load and preprocess the Excel data"""
        print("Loading data...")
        self.data = pd.read_excel(self.excel_path)
        
        # Clean the data
        self.data['descripcion_ot'] = self.data['descripcion_ot'].fillna('').astype(str).str.strip()
        self.data['descripcion_reparacion'] = self.data['descripcion_reparacion'].fillna('').astype(str).str.strip()
        self.data['descripcion_averia'] = self.data['descripcion_averia'].fillna('').astype(str).str.strip()
        
        # Filter out empty descriptions
        self.data = self.data[
            (self.data['descripcion_ot'] != '') & 
            (self.data['descripcion_reparacion'] != '')
        ].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_model(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Load the BERT model for embeddings
        Using multilingual model for Spanish text
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        print(f"Loading BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
        return self.model
    
    def create_embeddings(self):
        """Create embeddings for all descripcion_ot in the dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Creating embeddings...")
        descriptions = self.data['descripcion_ot'].tolist()
        
        # Create embeddings in batches for efficiency
        self.embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"Created {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
        return self.embeddings
    
    def search(self, query, top_k=5, save_results=True, output_dir='output'):
        """
        Search for similar repairs based on a new maintenance description
        
        Args:
            query: New maintenance description (descripcion_ot)
            top_k: Number of top results to return
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            DataFrame with top_k most similar repairs
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")
        
        # Create embedding for the query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = self.data.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]
        results['similarity_percent'] = (results['similarity'] * 100).round(2)
        
        results_df = results[['descripcion_ot', 'descripcion_averia', 'descripcion_reparacion', 
                              'equipo', 'fecha_creacion', 'similarity_percent']]
        
        # Save results if requested
        if save_results:
            self._save_results(query, results_df, output_dir)
        
        # Display summary window
        self._display_summary_window(query, results_df)
        
        return results_df
    
    def _display_summary_window(self, query, results):
        """Display a cool summary window with key information"""
        print("\n" + "╔" + "═"*98 + "╗")
        print("║" + " "*98 + "║")
        print("║" + "🔍 SEMANTIC SEARCH RESULTS".center(98) + "║")
        print("║" + " "*98 + "║")
        print("╠" + "═"*98 + "╣")
        print(f"║ Query: {query[:88]:<88} ║")
        print(f"║ Results: {len(results)} matches found{' '*69}║")
        print("╠" + "═"*98 + "╣")
        
        for idx, (_, row) in enumerate(results.iterrows(), 1):
            similarity = row['similarity_percent']
            
            # Create similarity bar
            bar_length = int(similarity / 2.5)  # Max 40 chars for 100%
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print("║" + " "*98 + "║")
            print(f"║ #{idx} │ {similarity:>6.2f}% │ {bar} ║")
            print("║" + " "*98 + "║")
            
            # Truncate text for display
            averia = row['descripcion_averia'][:90] + "..." if len(row['descripcion_averia']) > 90 else row['descripcion_averia']
            repair = row['descripcion_reparacion'][:90] + "..." if len(row['descripcion_reparacion']) > 90 else row['descripcion_reparacion']
            
            print(f"║ ⚠️  Averia:  {averia:<87}║")
            print(f"║ 🔧 Repair:  {repair:<87}║")
            
            if idx < len(results):
                print("║" + "─"*98 + "║")
        
        print("║" + " "*98 + "║")
        print("╚" + "═"*98 + "╝")
        print("\n💾 Full results saved to: output/semantic_search_results.txt")
        print("📊 Detailed CSV saved to: output/semantic_search_results.csv\n")
    
    def _save_results(self, query, results, output_dir):
        """Save results to formatted text and CSV files"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save as formatted text
        txt_path = os.path.join(output_dir, 'semantic_search_results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("SEMANTIC SEARCH RESULTS - MAINTENANCE REPAIRS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Number of results: {len(results)}\n")
            f.write("="*100 + "\n\n")
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                f.write(f"\n{'='*100}\n")
                f.write(f"RESULT #{idx} - SIMILARITY: {row['similarity_percent']}%\n")
                f.write(f"{'='*100}\n\n")
                f.write(f"Equipment: {row['equipo']}\n")
                f.write(f"Date: {row['fecha_creacion']}\n\n")
                f.write(f"Description (OT):\n")
                f.write(f"{row['descripcion_ot']}\n\n")
                f.write(f"Failure (Averia):\n")
                f.write(f"{row['descripcion_averia']}\n\n")
                f.write(f"Repair Solution:\n")
                f.write(f"{row['descripcion_reparacion']}\n\n")
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'semantic_search_results.csv')
        results_csv = results.copy()
        results_csv['query'] = query
        results_csv['timestamp'] = timestamp
        results_csv = results_csv[['timestamp', 'query', 'similarity_percent', 'equipo', 
                                   'fecha_creacion', 'descripcion_ot', 'descripcion_averia', 
                                   'descripcion_reparacion']]
        results_csv.to_csv(csv_path, index=False, encoding='utf-8')
    
    def save_embeddings(self, filepath='models/embeddings/embeddings.pkl'):
        """Save embeddings and data to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"Saving embeddings to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'data': self.data
            }, f)
        print("Embeddings saved!")
    
    def load_embeddings(self, filepath='models/embeddings/embeddings.pkl'):
        """Load pre-computed embeddings from disk"""
        print(f"Loading embeddings from {filepath}...")
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.embeddings = saved_data['embeddings']
            self.data = saved_data['data']
        print("Embeddings loaded!")