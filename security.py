import os
import hashlib
from PIL import Image
import io

def validate_image(image_file, max_size_mb=5):
    """Validate uploaded image for security"""
    # Check file size
    file_size_mb = len(image_file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size exceeds {max_size_mb}MB limit")
    
    # Verify it's a valid image
    try:
        img = Image.open(io.BytesIO(image_file.getvalue()))
        img.verify()
    except Exception:
        raise ValueError("Invalid image file")
    
    return True

def sanitize_filename(filename):
    """Sanitize filename for security"""
    # Remove path components and get filename
    filename = os.path.basename(filename)
    
    # Remove any null bytes
    filename = filename.replace('\0', '')
    
    # Generate a unique filename using hash
    name, ext = os.path.splitext(filename)
    hash_name = hashlib.md5(name.encode()).hexdigest()
    
    return f"{hash_name}{ext.lower()}"

def validate_file_type(filename, allowed_types):
    """Validate file type"""
    ext = os.path.splitext(filename)[1].lower().replace('.', '')
    if ext not in allowed_types:
        raise ValueError(f"File type not allowed. Supported types: {', '.join(allowed_types)}")
    return True 