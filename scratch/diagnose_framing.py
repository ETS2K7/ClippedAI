import modal
import sys

app = modal.App("diagnose-framing")

@app.local_entrypoint()
def main():
    try:
        remote_cache = modal.Dict.from_name("clippedai-asd-cache")
        key = "3e56f782efaad2b2fbc21efc0e18689cffc8037a3975dddb13acc6a4d126f48e"
        
        if key not in remote_cache:
            print("Key not found in cache")
            return
            
        data = remote_cache[key]
        
        # We need to see what the coordinates are, and how get_centered_crop clamps them.
        w_img = 1280
        h_img = 720
        crop_w = 405  # Standard 9:16 crop width for 720p height
        
        print("Frame | cx_norm | cx_px  | Clamp x1 | Actual Center in Crop")
        print("-" * 65)
        
        for i in range(min(50, len(data))):
            item = data[i]
            faces = item["faces"]
            speaking_faces = [f for f in faces if f.get("speaking", False)]
            
            if not faces:
                print(f"{i:5} | NO FACE |        |          |")
                continue
                
            f = speaking_faces[0] if speaking_faces else max(faces, key=lambda x: (x["x2"]-x["x1"])*(x["y2"]-x["y1"]))
            cx_px = (f["x1"] + f["x2"]) / 2.0
            cx_norm = cx_px / w_img
            
            # Simulate get_centered_crop
            x1 = int(round(cx_px - crop_w / 2))
            x1_clamped = max(0, min(w_img - crop_w, x1))
            actual_center = x1_clamped + (crop_w / 2)
            
            offset = cx_px - actual_center
            
            print(f"{i:5} | {cx_norm:.3f}   | {cx_px:6.1f} | {x1_clamped:8d} | {actual_center:6.1f} (offset: {offset:+.1f}px)")

    except Exception as e:
        print("Error:", e)

