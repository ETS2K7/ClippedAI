from PIL import Image, ImageDraw

def create_icons():
    # Load original image
    img = Image.open('/Users/ebelthomasseiko/clippedai/Gemini_Generated_Image_gidubagidubagidu-Photoroom.png').convert("RGBA")
    
    # 1. Make the logo white
    r, g, b, a = img.split()
    white_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
    white_img.putalpha(a)
    
    # 2. Hard crop to ignore faint invisible noise from Photoroom
    thresholded_a = a.point(lambda p: 255 if p > 50 else 0)
    bbox = thresholded_a.getbbox()
    if bbox:
        white_img = white_img.crop(bbox)
    
    # 3. Resize logo (60% of the box to leave a bold circular border)
    box_size = 512
    logo_size = int(box_size * 0.60)
    
    aspect = white_img.width / white_img.height
    if aspect > 1:
        new_w = logo_size
        new_h = int(logo_size / aspect)
    else:
        new_h = logo_size
        new_w = int(logo_size * aspect)
        
    white_img = white_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 4. Create black circle
    mask = Image.new("L", (box_size, box_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, box_size, box_size), fill=255)
    
    base = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
    black_circle = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 255))
    base.paste(black_circle, (0,0), mask)
    
    # 5. Paste the properly scaled white logo onto the black circle
    offset_x = (box_size - new_w) // 2
    offset_y = (box_size - new_h) // 2
    base.paste(white_img, (offset_x, offset_y), white_img)
    
    # 6. Save icons
    public_dir = '/Users/ebelthomasseiko/clippedai/frontend/public'
    
    base.save(f"{public_dir}/icon.png")
    
    apple = base.resize((180, 180), Image.Resampling.LANCZOS)
    apple.save(f"{public_dir}/apple-icon.png")
    
    fav = base.resize((32, 32), Image.Resampling.LANCZOS)
    fav.save(f"{public_dir}/favicon.ico", format="ICO")

create_icons()
