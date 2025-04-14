import os
import flask
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageOps # Added ImageOps
import io
import base64
import uuid
import cv2 # Import OpenCV
import numpy as np # Import NumPy

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
EDITED_FOLDER = 'edited' # You might save final results here
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Ensure you have a readable font file or change the path/name
FONT_FILE = "arial.ttf"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EDITED_FOLDER'] = EDITED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit

# Ensure upload and edited directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EDITED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper to decode base64 image data
def decode_image(base64_string):
    """Decodes base64 string to a PIL Image"""
    if ';base64,' in base64_string:
        header, base64_data = base64_string.split(';base64,')
    else:
        # Handle cases where the header might be missing (less robust)
        base64_data = base64_string

    try:
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(img_bytes))
        return img.convert("RGBA") # Ensure RGBA for consistency
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None

# Helper to decode base64 to OpenCV format (for inpainting mask)
def decode_image_to_opencv(base64_string):
    """Decodes base64 string to an OpenCV image (NumPy array)"""
    if ';base64,' in base64_string:
        header, base64_data = base64_string.split(';base64,')
    else:
        base64_data = base64_string

    try:
        img_bytes = base64.b64decode(base64_data)
        # Convert bytes to NumPy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Decode image using OpenCV
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) # Read with alpha if present
        return img_cv
    except Exception as e:
        print(f"Error decoding base64 to OpenCV: {e}")
        return None

# Helper to encode PIL image to base64
def encode_image_to_base64(pil_image, format='PNG'):
    """Encodes a PIL image to a base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{format.lower()};base64,{img_str}'


# --- Routes ---

@app.route('/')
def index():
    # Renders the main HTML page for the editor
    return render_template('editor.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            # Save file temporarily
            file.save(filepath)
            # Optionally: Sanitize or convert image on upload
            with Image.open(filepath) as img:
                 # Example: Convert to RGBA PNG for consistency internaly
                 rgba_img = img.convert("RGBA")
                 png_filename = os.path.splitext(filename)[0] + ".png"
                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
                 rgba_img.save(filepath, "PNG")
                 filename = png_filename # Update filename

            # Return the URL or identifier for the frontend to reference
            return jsonify({'imageUrl': f'/uploads/{filename}', 'filename': filename})
        except Exception as e:
            print(f"Error during upload processing: {e}")
            # Clean up potentially corrupted file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Failed to process uploaded image'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400

    original_filename = data.get('filename')
    actions = data.get('actions') # List of actions (crop, rotate, text, brush, inpaint)
    # Get current image state if provided (e.g., after client-side brushing)
    current_image_data = data.get('currentImageData')

    if not original_filename and not current_image_data:
         return jsonify({'error': 'Missing filename or current image data'}), 400
    if not actions:
         return jsonify({'error': 'Missing actions'}), 400

    img = None
    if current_image_data:
        # If client sent modified image data (e.g., with brushes drawn), use that
        img = decode_image(current_image_data)
    elif original_filename:
        # Otherwise, load the original uploaded file
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        if not os.path.exists(original_filepath):
             return jsonify({'error': f'Original file {original_filename} not found'}), 404
        try:
            img = Image.open(original_filepath).convert("RGBA")
        except Exception as e:
            return jsonify({'error': f'Failed to open image: {e}'}), 500

    if img is None:
        return jsonify({'error': 'Could not load image source'}), 500

    # Process actions sequentially
    try:
        for action in actions:
            op = action.get('operation')

            if op == 'crop':
                coords = action.get('coords')
                if coords:
                    box = (coords['x'], coords['y'], coords['x'] + coords['width'], coords['y'] + coords['height'])
                    img = img.crop(box)
            elif op == 'rotate':
                angle = action.get('angle')
                if angle is not None:
                    # PIL rotates counter-clockwise. 'expand=True' prevents cropping.
                    # Need a background color for expansion if original was not RGBA (we convert on upload now)
                    img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))
            elif op == 'add_text':
                text_data = action.get('textData')
                if text_data:
                    draw = ImageDraw.Draw(img)
                    try:
                        # Ensure font file exists or handle error
                        font = ImageFont.truetype(FONT_FILE, text_data.get('size', 30))
                    except IOError:
                        print(f"Warning: Font '{FONT_FILE}' not found. Using default.")
                        font = ImageFont.load_default()

                    fill_color_hex = text_data.get('color', '#000000').lstrip('#')
                    # Convert hex color to RGBA tuple
                    try:
                       fill_color = tuple(int(fill_color_hex[i:i+2], 16) for i in (0, 2, 4)) + (255,) # Add alpha
                    except ValueError:
                       fill_color = (0, 0, 0, 255) # Default to black on error

                    draw.text((text_data.get('x', 10), text_data.get('y', 10)),
                              text_data.get('text', ''),
                              font=font,
                              fill=fill_color)
            # elif op == 'brush': # Handled client side first, potentially applied via current_image_data
            #      # If brush strokes were sent separately (e.g., as another base64 overlay):
            #      brush_overlay_data = action.get('overlayData')
            #      if brush_overlay_data:
            #           overlay_img = decode_image(brush_overlay_data)
            #           if overlay_img and overlay_img.size == img.size:
            #               # Composite the brush overlay onto the image
            #               img = Image.alpha_composite(img.convert("RGBA"), overlay_img)

            elif op == 'inpaint':
                mask_data = action.get('maskData')
                inpaint_radius = action.get('radius', 3) # Default radius
                inpaint_method_flag = cv2.INPAINT_TELEA # or cv2.INPAINT_NS

                if mask_data:
                    # Decode mask sent from frontend
                    mask_cv = decode_image_to_opencv(mask_data)

                    if mask_cv is None:
                        print("Warning: Could not decode inpainting mask.")
                        continue

                    # Ensure mask is 8-bit single channel (grayscale)
                    if len(mask_cv.shape) == 3 and mask_cv.shape[2] == 4: # RGBA
                         mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGRA2GRAY)
                    elif len(mask_cv.shape) == 3 and mask_cv.shape[2] == 3: # RGB
                         mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
                    elif len(mask_cv.shape) != 2:
                         print(f"Warning: Unexpected mask shape {mask_cv.shape}. Skipping inpaint.")
                         continue

                    # Threshold mask: OpenCV inpaint needs non-zero pixels for the mask
                    _, mask_thresh = cv2.threshold(mask_cv, 1, 255, cv2.THRESH_BINARY)

                    # Convert PIL image (RGBA) to OpenCV format (BGRA)
                    img_cv_bgra = np.array(img)
                    img_cv_bgr = cv2.cvtColor(img_cv_bgra, cv2.COLOR_RGBA2BGR) # Inpaint works on BGR

                    # Ensure mask and image dimensions match
                    if img_cv_bgr.shape[:2] != mask_thresh.shape[:2]:
                        print(f"Warning: Image ({img_cv_bgr.shape[:2]}) and mask ({mask_thresh.shape[:2]}) dimensions mismatch. Resizing mask.")
                        # Resize mask to match image (might affect accuracy)
                        mask_thresh = cv2.resize(mask_thresh, (img_cv_bgr.shape[1], img_cv_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


                    print(f"Performing inpainting... Image shape: {img_cv_bgr.shape}, Mask shape: {mask_thresh.shape}")
                    # Perform inpainting
                    try:
                        inpainted_img_cv = cv2.inpaint(img_cv_bgr, mask_thresh, inpaint_radius, flags=inpaint_method_flag)

                        # Convert back to PIL RGBA format
                        inpainted_img_rgba = cv2.cvtColor(inpainted_img_cv, cv2.COLOR_BGR2RGBA)
                        img = Image.fromarray(inpainted_img_rgba)
                        print("Inpainting successful.")

                    except cv2.error as e:
                        print(f"OpenCV Error during inpainting: {e}")
                        # Continue without inpainting if it fails
                        continue
                    except Exception as e:
                        print(f"General Error during inpainting: {e}")
                        continue


        # --- Encode final result back to base64 ---
        final_image_data = encode_image_to_base64(img, format='PNG') # Always send PNG back

        # Return the processed image data
        return jsonify({'imageData': final_image_data})

    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error processing image actions: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {e}'}), 500

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Add security check: ensure filename is safe
    safe_path = flask.safe_join(app.config['UPLOAD_FOLDER'], filename)
    if not safe_path or not os.path.exists(safe_path):
         flask.abort(404)
    # Check if the resolved path is still within the UPLOAD_FOLDER to prevent directory traversal
    if os.path.commonprefix((os.path.realpath(safe_path), os.path.realpath(app.config['UPLOAD_FOLDER']))) != os.path.realpath(app.config['UPLOAD_FOLDER']):
        flask.abort(403) # Forbidden
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Make sure Arial.ttf exists or change FONT_FILE path
    if not os.path.exists(FONT_FILE):
         print(f"Warning: Font file '{FONT_FILE}' not found. Text rendering might use default font.")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
