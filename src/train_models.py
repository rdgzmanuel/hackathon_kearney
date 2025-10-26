import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGAR Y PREPARAR LOS DATOS
# ============================================================================

df_descriptions = pd.read_csv("descripciones_caf.csv", sep=";")
df_components = pd.read_csv("Definiciones clave Codigo actuacion_V6.csv", sep=";", header=2)

df_fre = df_components[df_components['Clavero'].str.startswith('FRE', na=False)].copy()

print(f"Total descripciones: {len(df_descriptions)}")
print(f"Total componentes FRE: {len(df_fre)}")

# ============================================================================
# 2. RELACIONAR Y AGRUPAR CLASES MINORITARIAS
# ============================================================================

clavero_to_component = df_fre.set_index('Clavero')['Descripción componente'].to_dict()
df_descriptions['componente'] = df_descriptions['clavero'].map(clavero_to_component)
df_valid = df_descriptions.dropna(subset=['componente', 'descripcion_ot']).copy()

print(f"\nDescripciones con componente válido: {len(df_valid)}")

# ESTRATEGIA: Agrupar clases con MENOS DE 2 MUESTRAS
MIN_SAMPLES_PER_CLASS = 2
component_counts = df_valid['componente'].value_counts()
rare_components = component_counts[component_counts < MIN_SAMPLES_PER_CLASS].index
df_valid['componente_agrupado'] = df_valid['componente'].apply(
    lambda x: 'Otros' if x in rare_components else x
)

print(f"\nAgrupación (min={MIN_SAMPLES_PER_CLASS} muestras):")
print(f"  Componentes agrupados en 'Otros': {len(rare_components)}")
print(f"  Muestras en 'Otros': {(df_valid['componente_agrupado'] == 'Otros').sum()}")
print(f"  Clases finales: {df_valid['componente_agrupado'].nunique()}")

print(f"\nDistribución final (top 15):")
print(df_valid['componente_agrupado'].value_counts().head(15))

# ============================================================================
# 3. GENERAR EMBEDDINGS CON BERT
# ============================================================================

print("\n" + "="*80)
print("GENERANDO EMBEDDINGS...")
print("="*80)

model_name = "google-bert/bert-base-uncased"  # Cambiar a español: "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

descriptions_list = df_valid['descripcion_ot'].fillna("").astype(str).tolist()
X_raw = get_embeddings(descriptions_list)
y = df_valid['componente_agrupado'].values

print(f"Embeddings shape: {X_raw.shape}")

# ============================================================================
# 4. REDUCCIÓN DE DIMENSIONALIDAD (Anti-overfitting)
# ============================================================================

print("\n" + "="*80)
print("APLICANDO PCA PARA REDUCIR DIMENSIONALIDAD")
print("="*80)

# Reducir de 768 a 100 dimensiones
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_raw)

print(f"Dimensiones reducidas: {X_raw.shape[1]} → {X_pca.shape[1]}")
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.4f}")

# Escalar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X_pca)

# ============================================================================
# 5. SPLIT CON MÁS DATOS EN TEST
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================================
# 6. FUNCIÓN PARA CALCULAR TOP-N ACCURACY
# ============================================================================

def top_n_accuracy(y_true, y_proba, classes, n=3):
    """
    Calcula el accuracy top-n: acierto si la clase correcta está entre las top-n predicciones
    
    Args:
        y_true: etiquetas verdaderas
        y_proba: matriz de probabilidades (n_samples x n_classes)
        classes: array con los nombres de las clases
        n: número de predicciones a considerar
    
    Returns:
        float: accuracy top-n
    """
    correct = 0
    for i in range(len(y_true)):
        # Obtener índices de las n predicciones con mayor probabilidad
        topn_idx = np.argsort(y_proba[i])[-n:][::-1]
        topn_classes = classes[topn_idx]
        if y_true[i] in topn_classes:
            correct += 1
    return correct / len(y_true)

# ============================================================================
# 7. PROBAR MÚLTIPLES MODELOS CON TOP-N METRICS
# ============================================================================

print("\n" + "="*80)
print("ENTRENANDO Y COMPARANDO MODELOS")
print("="*80)

modelos = {
    'Random Forest (Regularizado)': RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    ),
    'Linear SVM': LinearSVC(
        C=0.1,
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
}

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\n{'─'*80}")
    print(f"📊 {nombre}")
    print(f"{'─'*80}")
    
    # Entrenar
    modelo.fit(X_train, y_train)
    
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas básicas
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    gap = train_acc - test_acc
    
    print(f"Train Accuracy (Top-1): {train_acc:.4f}")
    print(f"Test Accuracy (Top-1):  {test_acc:.4f}")
    print(f"Gap (overfitting):      {gap:.4f}")
    
    # Top-N Accuracy (solo para modelos con predict_proba)
    if hasattr(modelo, 'predict_proba'):
        y_proba_test = modelo.predict_proba(X_test)
        classes = modelo.classes_
        
        top3_acc = top_n_accuracy(y_test, y_proba_test, classes, n=3)
        top5_acc = top_n_accuracy(y_test, y_proba_test, classes, n=5)
        
        print(f"Test Accuracy (Top-3):  {top3_acc:.4f}")
        print(f"Test Accuracy (Top-5):  {top5_acc:.4f}")
    else:
        # Para SVM, calcular pseudo-probabilidades
        scores = modelo.decision_function(X_test)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        y_proba_test = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        classes = modelo.classes_
        
        top3_acc = top_n_accuracy(y_test, y_proba_test, classes, n=3)
        top5_acc = top_n_accuracy(y_test, y_proba_test, classes, n=5)
        
        print(f"Test Accuracy (Top-3):  {top3_acc:.4f}")
        print(f"Test Accuracy (Top-5):  {top5_acc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, n_jobs=-1)
    print(f"CV Accuracy (5-fold):   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    resultados[nombre] = {
        'modelo': modelo,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'top3_acc': top3_acc,
        'top5_acc': top5_acc,
        'gap': gap,
        'cv_mean': cv_scores.mean()
    }

# ============================================================================
# 8. RESUMEN COMPARATIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*80)

# Ordenar por test accuracy
ranking = sorted(resultados.items(), key=lambda x: x[1]['test_acc'], reverse=True)

print(f"\n{'Modelo':<35} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Gap':<10}")
print("─" * 80)
for nombre, res in ranking:
    print(f"{nombre:<35} {res['test_acc']:.4f}     {res['top3_acc']:.4f}     "
          f"{res['top5_acc']:.4f}     {res['gap']:.4f}")

# Mejor modelo
mejor_nombre, mejor_res = ranking[0]
mejor_modelo = mejor_res['modelo']

print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
print(f"   Top-1 Accuracy: {mejor_res['test_acc']:.4f}")
print(f"   Top-3 Accuracy: {mejor_res['top3_acc']:.4f}")
print(f"   Top-5 Accuracy: {mejor_res['top5_acc']:.4f}")
print(f"   Overfitting Gap: {mejor_res['gap']:.4f}")

# ============================================================================
# 9. REPORTE DETALLADO DEL MEJOR MODELO
# ============================================================================

print("\n" + "="*80)
print(f"REPORTE DETALLADO: {mejor_nombre}")
print("="*80)

y_pred_test = mejor_modelo.predict(X_test)
print(classification_report(y_test, y_pred_test, zero_division=0))

# ============================================================================
# 10. ANÁLISIS DE ERRORES
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE ERRORES MÁS COMUNES")
print("="*80)

errores = []
for i, (real, pred) in enumerate(zip(y_test, y_pred_test)):
    if real != pred:
        errores.append({'real': real, 'predicho': pred})

df_errores = pd.DataFrame(errores)
if len(df_errores) > 0:
    error_counts = df_errores.groupby(['real', 'predicho']).size().sort_values(ascending=False)
    print("\nTop 10 confusiones más frecuentes:")
    for (real, pred), count in error_counts.head(10).items():
        print(f"  {count}x: '{real}' → '{pred}'")

# ============================================================================
# 11. FUNCIÓN DE PREDICCIÓN CON TOP-N
# ============================================================================

def predecir_componente(descripcion_nueva, top_n=5):
    """
    Predice componente para nueva descripción con top-n resultados
    
    Args:
        descripcion_nueva: texto de la descripción
        top_n: número de predicciones a retornar
    
    Returns:
        Lista de tuplas (componente, probabilidad)
    """
    # Embedding
    embedding = get_embeddings([descripcion_nueva])
    # PCA + Scaling
    embedding_pca = pca.transform(embedding)
    embedding_scaled = scaler.transform(embedding_pca)
    
    # Predicción con probabilidades
    if hasattr(mejor_modelo, 'predict_proba'):
        probas = mejor_modelo.predict_proba(embedding_scaled)[0]
        clases = mejor_modelo.classes_
    else:  # Para SVM
        scores = mejor_modelo.decision_function(embedding_scaled)[0]
        clases = mejor_modelo.classes_
        exp_scores = np.exp(scores - np.max(scores))
        probas = exp_scores / exp_scores.sum()
    
    top_indices = np.argsort(probas)[-top_n:][::-1]
    return [(clases[idx], probas[idx]) for idx in top_indices]

# ============================================================================
# 12. EJEMPLO DE USO
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO DE PREDICCIÓN CON TOP-5")
print("="*80)

# Tomar varios ejemplos de test
for ej_idx in range(min(3, len(y_test))):
    descripcion_ejemplo = descriptions_list[X_train.shape[0] + ej_idx]
    componente_real = y_test[ej_idx]
    
    print(f"\n{'─'*80}")
    print(f"Ejemplo {ej_idx + 1}:")
    print(f"Descripción: {descripcion_ejemplo[:120]}...")
    print(f"\nComponente REAL: {componente_real}")
    print(f"\nPredicciones Top-5:")
    
    predicciones = predecir_componente(descripcion_ejemplo, top_n=5)
    for i, (comp, prob) in enumerate(predicciones, 1):
        marca = "✓✓✓" if comp == componente_real else "   "
        print(f"{marca} {i}. {comp:<50} {prob:.4f}")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print("\n💡 INTERPRETACIÓN DE MÉTRICAS:")
print("  • Top-1: Predicción exacta en la primera posición")
print("  • Top-3: Componente real está entre las 3 predicciones más probables")
print("  • Top-5: Componente real está entre las 5 predicciones más probables")
print("\n🎯 PRÓXIMOS PASOS SI ACCURACY <60%:")
print("  • Usar modelo BERT en español para mejorar embeddings")
print("  • Recopilar más datos de entrenamiento")
print("  • Agrupar clases similares semánticamente")


import joblib
import os

# Crear directorio para modelos
os.makedirs("./models", exist_ok=True)

# Guardar PCA
print("Guardando PCA...")
joblib.dump(pca, "./models/pca_model.pkl")

# Guardar Scaler
print("Guardando Scaler...")
joblib.dump(scaler, "./models/scaler_model.pkl")

# Guardar Clasificador (el mejor modelo)
print("Guardando Clasificador...")
joblib.dump(mejor_modelo, "./models/classifier_model.pkl")

print("\n✅ Modelos guardados en ./models/")
print("   - pca_model.pkl")
print("   - scaler_model.pkl")
print("   - classifier_model.pkl")

# Verificar tamaños
import os
for file in ["pca_model.pkl", "scaler_model.pkl", "classifier_model.pkl"]:
    size_mb = os.path.getsize(f"./models/{file}") / (1024 * 1024)
    print(f"   {file}: {size_mb:.2f} MB")
