from PIL import Image, ImageDraw

def create_icons():
    # Load original image
    img = Image.open('/Users/ebelthomasseiko/clippedai/Gemini_Generated_Image_gidubagidubagidu-Photoroom.png').convert("RGBA")
    
    # 1. Make the logo white
    r, g, b, a = img.split()
    white_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
    white_img.putalpha(a)
    
    # Crop to bounding box to remove the massive transparent padding
    bbox = a.getbbox()
    if bbox:
        white_img = white_img.crop(bbox)
    
    # 2. Resize logo to fit inside the 512x512 box (at 65% scale for breathing room)
    box_size = 512
    logo_size = int(box_size * 0.65)
    
    aspect = white_img.width / white_img.height
    if aspect > 1:
        new_w = logo_size
        new_h = int(logo_size / aspect)
    else:
        new_h = logo_size
        new_w = int(logo_size * aspect)
        
    white_img = white_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Create black rounded box
    mask = Image.new("L", (box_size, box_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, box_size, box_size), radius=110, fill=255)
    
    base = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
    black_square = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 255))
    base.paste(black_square, (0,0), mask)
    
    # 4. Paste white logo onto the black box
    offset_x = (box_size - new_w) // 2
    offset_y = (box_size - new_h) // 2
    base.paste(white_img, (offset_x, offset_y), white_img)
    
    # 5. Save icons
    public_dir = '/Users/ebelthomasseiko/clippedai/frontend/public'
    
    base.save(f"{public_dir}/icon.png")
    
    apple = base.resize((180, 180), Image.Resampling.LANCZOS)
    apple.save(f"{public_dir}/apple-icon.png")
    
    fav = base.resize((32, 32), Image.Resampling.LANCZOS)
    fav.save(f"{public_dir}/favicon.ico", format="ICO")

create_icons()
