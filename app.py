import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub
import kagglehub

# ==========================================
# APP INITIALIZATION
# ==========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASS_NAMES = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)'
}

print("Constructing AI Architectures and Loading Weights... Please wait.")

# ==========================================
# 1. Reconstruct and Load CNN (MobileNetV2)
# ==========================================
try:
    print("🏗️ Building MobileNetV2 Architecture...")
    base_cnn = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )
    
    x = base_cnn.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    cnn_outputs = keras.layers.Dense(7, activation='softmax')(x)
    
    cnn_model = keras.models.Model(inputs=base_cnn.input, outputs=cnn_outputs)
    cnn_model.load_weights('models/best_model.h5')
    print("✅ MobileNetV2 Weights Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading CNN weights: {e}")

# ==========================================
# 2. Reconstruct and Load ViT (Vision Transformer)
# ==========================================
try:
    print("🏗️ Building Vision Transformer Architecture...")
    local_vit_path = kagglehub.model_download('spsayakpaul/vision-transformer/tensorFlow2/vit-b16-fe')
    
    vit_layer = hub.KerasLayer(local_vit_path, trainable=False, name='vit_layer')
    
    vit_inputs = keras.layers.Input(shape=(224, 224, 3))
    x = vit_layer(vit_inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    vit_outputs = keras.layers.Dense(7, activation='softmax')(x)
    
    vit_model = keras.models.Model(inputs=vit_inputs, outputs=vit_outputs)
    
    vit_model.load_weights('models/vit_best_model.keras', by_name=True, skip_mismatch=True)
    print("✅ Vision Transformer Weights Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading ViT weights: {e}")

# ==========================================
# 3. WARM-UP MODELS (To prevent AutoGraph Crash)
# ==========================================
print("🔥 Warming up models... (This prevents the first-prediction crash)")
try:
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    cnn_model.predict(dummy_img, verbose=0)
    vit_model.predict(dummy_img, verbose=0)
    print("✅ Models warmed up and ready for real images!")
except Exception as e:
    print(f"⚠️ Warm-up warning: {e}")

# ==========================================
# WEB APP ROUTES
# ==========================================
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    model_choice = request.form.get('model_choice') 
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_array = preprocess_image(filepath)
        result_text = ""
        model_name = ""
        
        try:
            if model_choice == 'cnn':
                pred = cnn_model.predict(img_array)
                model_name = "CNN (MobileNetV2)"
            elif model_choice == 'vit':
                pred = vit_model.predict(img_array)
                model_name = "Vision Transformer (ViT-B/16)"
            else:
                return "Invalid Model Selected", 400

            class_idx = np.argmax(pred)
            class_name = CLASS_NAMES[class_idx]
            confidence = round(np.max(pred) * 100, 2)
            result_text = f"{class_name} ({confidence}%)"
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", 500
        
        return render_template('index.html', 
                               uploaded_image=filepath, 
                               result=result_text,
                               model_used=model_name)

if __name__ == '__main__':
    # 🔥 debug=False and use_reloader=False stops the crash
    app.run(debug=False, use_reloader=False, port=5000, threaded=False)