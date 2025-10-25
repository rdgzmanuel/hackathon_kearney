from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import ollama
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Variables globales
df = None
csv_content = ""
chat_history = []

def load_csv(file_path):
    """Carga el archivo CSV y prepara el contexto"""
    global df, csv_content
    
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=';', encoding=encoding)
                print(f"✅ CSV cargado con encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("No se pudo leer el CSV con ningún encoding conocido")
        
        df = df.fillna('')
        
        # Resumen compacto pero completo
        csv_summary = f"""Dataset: {len(df)} órdenes de trabajo
Equipos únicos: {df['equipo'].nunique()} | Actuaciones: {df['actuacion'].nunique()}
Fechas: {df['fecha_creacion'].min()} a {df['fecha_creacion'].max()}"""
        
        csv_content = csv_summary
        print(f"✅ CSV cargado: {len(df)} registros")
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar CSV: {str(e)}")
        return False

def search_relevant_data(query, limit=8):
    """Busca datos relevantes con scoring inteligente"""
    global df
    
    if df is None or len(df) == 0:
        return "No hay datos disponibles.", 0
    
    keywords = query.lower().split()
    
    # Crear scoring por relevancia
    df['score'] = 0
    
    text_columns = {
        'descripcion_averia': 3,      # Mayor peso - lo más importante
        'descripcion_reparacion': 3,
        'actuacion': 2,
        'descripcion_ot': 2,
        'equipo': 1,
        'comentarios': 1
    }
    
    # Calcular score de relevancia
    for keyword in keywords:
        for col, weight in text_columns.items():
            if col in df.columns:
                matches = df[col].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
                df.loc[matches, 'score'] += weight
    
    # Obtener los más relevantes
    relevant_rows = df[df['score'] > 0].nlargest(limit, 'score')
    total_matches = len(df[df['score'] > 0])
    
    if len(relevant_rows) == 0:
        return "No se encontraron registros coincidentes.", 0
    
    # Formato compacto pero informativo
    context = f"\n📊 ENCONTRADOS: {total_matches} registros totales (mostrando top {len(relevant_rows)} más relevantes)\n\n"
    
    for idx, row in relevant_rows.iterrows():
        context += f"🔧 OT-{row['codigo_ot']} | {row['equipo']} | {row['fecha_creacion']}\n"
        context += f"   Actuación: {row['actuacion']}\n"
        
        # Mostrar descripciones completas solo si son cortas, sino resumir
        averia = str(row['descripcion_averia'])
        reparacion = str(row['descripcion_reparacion'])
        
        if len(averia) > 200:
            context += f"   Avería: {averia[:200]}...\n"
        else:
            context += f"   Avería: {averia}\n"
            
        if len(reparacion) > 200:
            context += f"   Reparación: {reparacion[:200]}...\n"
        else:
            context += f"   Reparación: {reparacion}\n"
        
        if row['comentarios']:
            context += f"   Comentarios: {str(row['comentarios'])[:150]}\n"
        
        context += "\n"
    
    return context, total_matches

def get_statistics_context(query):
    """Genera estadísticas cuando se solicitan análisis generales"""
    global df
    
    if df is None:
        return ""
    
    # Detectar si es una pregunta estadística
    stat_keywords = ['cuántos', 'cuántas', 'total', 'cantidad', 'estadística', 
                     'más', 'menos', 'frecuente', 'común', 'lista', 'todos']
    
    is_stat_query = any(keyword in query.lower() for keyword in stat_keywords)
    
    if not is_stat_query:
        return ""
    
    # Generar estadísticas relevantes
    stats = "\n📈 ESTADÍSTICAS GLOBALES:\n"
    stats += f"Total OTs: {len(df)}\n"
    stats += f"Equipos únicos: {df['equipo'].nunique()}\n"
    
    # Top 5 equipos con más OTs
    top_equipos = df['equipo'].value_counts().head(5)
    stats += f"\nTop 5 equipos:\n"
    for equipo, count in top_equipos.items():
        stats += f"  - {equipo}: {count} OTs\n"
    
    # Tipos de actuación
    top_actuaciones = df['actuacion'].value_counts().head(5)
    stats += f"\nTop 5 actuaciones:\n"
    for act, count in top_actuaciones.items():
        stats += f"  - {act}: {count} OTs\n"
    
    return stats

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principal del chatbot"""
    global chat_history
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        if df is None:
            return jsonify({
                'response': '❌ No hay datos cargados.',
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"💬 Procesando: {user_message}")
        
        # Obtener datos relevantes + estadísticas si aplica
        relevant_data, total_matches = search_relevant_data(user_message, limit=8)
        stats_context = get_statistics_context(user_message)
        
        # Prompt optimizado pero preciso
        system_prompt = f"""Eres un asistente técnico experto en análisis de órdenes de trabajo ferroviarias.

INFORMACIÓN DEL DATASET:
{csv_content}

INSTRUCCIONES:
1. Analiza TODOS los datos relevantes proporcionados
2. Sé preciso y cita códigos OT, equipos o fechas específicas
3. Si hay múltiples casos, identifica patrones o tendencias
4. Para estadísticas, usa los números exactos proporcionados
5. Si la info es incompleta, menciona cuántos registros totales hay
6. Responde de forma clara pero profesional

{stats_context}

DATOS ESPECÍFICOS PARA ESTA CONSULTA:
{relevant_data}
"""

        # Mantener historial contextual (últimos 3 intercambios)
        messages = [{'role': 'system', 'content': system_prompt}]
        
        recent_history = chat_history[-6:] if len(chat_history) > 0 else []
        for msg in recent_history:
            messages.append(msg)
        
        messages.append({'role': 'user', 'content': user_message})
        
        print(f"🔍 Encontrados: {total_matches} registros totales")
        
        # Parámetros balanceados: velocidad + calidad
        response = ollama.chat(
            model='llama3.2',
            messages=messages,
            options={
                'temperature': 0.3,      # Balance entre creatividad y precisión
                'top_p': 0.85,
                'num_predict': 500,      # Suficiente para respuestas detalladas
            }
        )
        
        assistant_message = response['message']['content']
        
        # Guardar en historial
        chat_history.append({'role': 'user', 'content': user_message})
        chat_history.append({'role': 'assistant', 'content': assistant_message})
        
        # Limpiar historial antiguo
        if len(chat_history) > 12:
            chat_history = chat_history[-12:]
        
        print(f"✅ Respuesta generada: {len(assistant_message)} chars")
        
        return jsonify({
            'response': assistant_message,
            'timestamp': datetime.now().isoformat(),
            'records_found': total_matches
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': f'Error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtiene estadísticas del dataset"""
    try:
        if df is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        equipos = df['equipo'].value_counts().head(10).to_dict() if 'equipo' in df.columns else {}
        actuaciones = df['actuacion'].value_counts().to_dict() if 'actuacion' in df.columns else {}
        
        stats = {
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'equipos_top': equipos,
            'tipos_actuacion': actuaciones,
            'fecha_min': str(df['fecha_creacion'].min()) if 'fecha_creacion' in df.columns else 'N/A',
            'fecha_max': str(df['fecha_creacion'].max()) if 'fecha_creacion' in df.columns else 'N/A'
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Limpia el historial de conversación"""
    global chat_history
    chat_history = []
    return jsonify({'message': 'Historial limpiado', 'timestamp': datetime.now().isoformat()})

@app.route('/api/health', methods=['GET'])
def health():
    """Verifica el estado del servicio"""
    ollama_status = 'ok'
    try:
        ollama.list()
    except:
        ollama_status = 'error'
    
    return jsonify({
        'status': 'ok',
        'csv_loaded': df is not None,
        'records': len(df) if df is not None else 0,
        'chat_messages': len(chat_history),
        'ollama_status': ollama_status,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 CHATBOT DE ANÁLISIS DE AVERÍAS")
    print("="*60 + "\n")
    
    default_csv = './backend/descripciones_caf.csv'
    
    if os.path.exists(default_csv):
        print(f"📂 Cargando: {default_csv}")
        if load_csv(default_csv):
            print(f"✅ Dataset cargado: {len(df)} órdenes de trabajo")
        else:
            print("⚠️ Error al cargar el CSV")
    else:
        print(f"⚠️ Archivo no encontrado: {default_csv}")
    
    try:
        models = ollama.list()
        print(f"✅ Ollama conectado")
    except Exception as e:
        print(f"⚠️ Ollama no disponible: {str(e)}")
    
    print("\n" + "="*60)
    print("🚀 Servidor: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')