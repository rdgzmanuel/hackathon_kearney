import pandas as pd
from collections import defaultdict

class RepairPredictor:
    """
    Predicts repair probabilities based on component historical data.
    """
    
    def __init__(self):
        self.component_repairs = defaultdict(lambda: defaultdict(int))
        self.component_totals = defaultdict(int)
        self.is_trained = False
    
    def load_and_train(self, file_path: str, filter_clavero: str = None):
        """
        Load Excel file and train the model.
        
        Args:
            file_path: Path to Excel file
            filter_clavero: Filter by Clavero column (e.g., 'FRE' to get only FRE*)
        """
        print("Loading Excel file...")
        
        # Load Excel file (first row as header)
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except:
            df = pd.read_excel(file_path)
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {df.columns.tolist()}\n")
        
        # Filter by Clavero column if specified
        if filter_clavero:
            if 'Clavero' in df.columns:
                df['Clavero'] = df['Clavero'].astype(str)
                original_len = len(df)
                df = df[df['Clavero'].str.startswith(filter_clavero, na=False)]
                print(f"✓ Filtered by Clavero starting with '{filter_clavero}': {len(df)} rows (from {original_len})\n")
            else:
                print(f"⚠ Warning: Column 'Clavero' not found. Available columns: {df.columns.tolist()}")
                print("Continuing without filter...\n")
        
        # Check if required columns exist
        if 'Descripción componente' not in df.columns or 'DEFINICION' not in df.columns:
            print(f"⚠ Required columns not found!")
            print(f"Looking for: 'Descripción componente' and 'DEFINICION'")
            print(f"Available columns: {df.columns.tolist()}\n")
            
            # Try to find similar column names
            for col in df.columns:
                if 'Descripci' in str(col) or 'component' in str(col):
                    print(f"Found similar column: {col}")
                if 'DEFINICION' in str(col) or 'DEFINITION' in str(col):
                    print(f"Found similar column: {col}")
            return
        
        # Keep only the two Spanish columns
        df = df[['Descripción componente', 'DEFINICION']].copy()
        
        # Remove rows where DEFINICION is empty or NaN
        df = df.dropna(subset=['DEFINICION'])
        df = df[df['DEFINICION'].astype(str).str.strip() != '']
        
        print(f"✓ Valid repair records: {len(df)}\n")
        
        # Count repairs per component
        for _, row in df.iterrows():
            component = str(row['Descripción componente']).strip()
            repair = str(row['DEFINICION']).strip()
            
            if component and repair and component != 'nan':
                self.component_repairs[component][repair] += 1
                self.component_totals[component] += 1
        
        self.is_trained = True
        
        print("="*80)
        print("MODEL TRAINED SUCCESSFULLY")
        print("="*80)
        print(f"✓ Total components: {len(self.component_totals)}")
        print(f"✓ Components found:\n")
        for comp, count in sorted(self.component_totals.items()):
            print(f"  - {comp}: {count} registros")
        print()
    
    def predict(self, component: str):
        """
        Get repair predictions for a component.
        
        Args:
            component: Component name
            
        Returns:
            List of (repair_description, probability) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call load_and_train() first.")
        
        component = component.strip()
        
        if component not in self.component_repairs:
            return []
        
        repairs = self.component_repairs[component]
        total = self.component_totals[component]
        
        # Calculate probabilities
        results = [(repair, count / total) for repair, count in repairs.items()]
        
        # Sort by probability (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def predict_all_components(self):
        """
        Get predictions for all components in the dataset.
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call load_and_train() first.")
        
        print("\n" + "="*80)
        print("REPAIR PREDICTIONS FOR ALL COMPONENTS")
        print("="*80 + "\n")
        
        for component in sorted(self.component_totals.keys()):
            predictions = self.predict(component)
            
            print(f"📦 COMPONENT: {component}")
            print(f"   Total records: {self.component_totals[component]}")
            print(f"   Possible repairs:\n")
            
            for i, (repair, prob) in enumerate(predictions, 1):
                print(f"   {i}. [{prob*100:.1f}%] {repair}\n")
            
            print("-"*80 + "\n")
    
    def predict_one(self, component: str):
        """
        Get predictions for a specific component with formatted output.
        
        Args:
            component: Component name
        """
        predictions = self.predict(component)
        
        if not predictions:
            print(f"\n⚠ No data found for component: '{component}'")
            print(f"Available components: {list(self.component_totals.keys())}")
            return
        
        print(f"\n{'='*80}")
        print(f"📦 COMPONENT: {component}")
        print(f"{'='*80}")
        print(f"Total records: {self.component_totals[component]}")
        print(f"\nPossible repairs:\n")
        
        for i, (repair, prob) in enumerate(predictions, 1):
            print(f"{i}. Probability: {prob*100:.1f}%")
            print(f"   {repair}\n")
    
    def get_all_components(self):
        """Get list of all available components."""
        return sorted(self.component_totals.keys())


# USAGE EXAMPLE
if __name__ == "__main__":
    # 1. Create predictor
    predictor = RepairPredictor()
    
    # 2. Load Excel file and train model (filtering by Clavero = FRE*)
    predictor.load_and_train('data/raw/data.xls', filter_clavero='FRE')
    
    # 3. Option A: Get predictions for ALL components
    predictor.predict_all_components()
