from PIL import Image, ImageDraw

def create_rounded_square_icon(input_path, output_path, size=512, corner_radius=100):
    # Load the white C-Bolt logo
    logo = Image.open(input_path).convert("RGBA")
    
    # Create a new image with a dark background (matching the login box)
    # Background: #0a0a0a (Deep charcoal)
    bg_color = (10, 10, 10, 255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    
    # Draw rounded rectangle background
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size, size], radius=corner_radius, fill=bg_color)
    
    # Add a subtle border like the login page
    # Border: #ffffff15
    border_color = (255, 255, 255, 30)
    draw.rounded_rectangle([0, 0, size, size], radius=corner_radius, outline=border_color, width=8)
    
    # Resize logo to fit inside with padding
    logo_size = int(size * 0.65)
    logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
    
    # Paste logo in center
    offset = (size - logo_size) // 2
    img.paste(logo, (offset, offset), logo)
    
    # Save output
    img.save(output_path)
    print(f"Icon created at {output_path}")

if __name__ == "__main__":
    create_rounded_square_icon("frontend/public/logo.png", "frontend/public/icon.png")
    # Also save as favicon.png
    create_rounded_square_icon("frontend/public/logo.png", "frontend/public/favicon.png")
    # Create a small version for .ico
    img = Image.open("frontend/public/favicon.png").resize((32, 32), Image.Resampling.LANCZOS)
    img.save("frontend/public/favicon.ico")
