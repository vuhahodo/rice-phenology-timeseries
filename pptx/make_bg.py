"""Topic backgrounds: dark 'satellite over rice terraces' (title/end) and light 'terrace strip' (content)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
W, H = 2400, 1350
rng = np.random.default_rng(7)

def vgrad(top, bot):
    t = np.linspace(0, 1, H)[:, None, None]
    a = np.array(top, float)[None, None, :]; b = np.array(bot, float)[None, None, :]
    return (a * (1 - t) + b * t).repeat(W, 1).astype(np.uint8)

def terraces(img, y0, layers, amp, lam, rgba_list, seed=0):
    d = ImageDraw.Draw(img, "RGBA"); xs = np.arange(0, W + 8, 8)
    for i, col in enumerate(rgba_list):
        base = y0 + i * (H - y0) / (len(rgba_list) + 0.4)
        ph = rng.uniform(0, 6.28); a = amp * (1 - 0.12 * i)
        ys = base + a * np.sin(xs / lam * 6.28 + ph) + 0.4 * a * np.sin(xs / (lam * 0.37) * 6.28 + ph * 1.7)
        d.polygon([(0, H)] + list(zip(xs, ys)) + [(W, H)], fill=col)
        d.line(list(zip(xs, ys)), fill=(255, 255, 255, 38), width=3)

# ---------------- dark title/end background
dark = Image.fromarray(vgrad((8, 24, 20), (16, 52, 40))).convert("RGBA")
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
for x in range(0, W, 150): d.line([(x, 0), (x, H)], fill=(143, 184, 201, 16), width=2)       # lat/lon grid
for y in range(0, H, 150): d.line([(0, y), (W, y)], fill=(143, 184, 201, 16), width=2)
for k, (rx, ry, rot) in enumerate([(1500, 520, -16), (1250, 420, -16), (1000, 330, -16)]):    # orbit rings
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(tmp)
    cx, cy = 1700, 380
    dd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(217, 168, 58, 90 - k * 18), width=3)
    ov = Image.alpha_composite(ov, tmp.rotate(rot, center=(cx, cy)))
dark = Image.alpha_composite(dark, ov); d = ImageDraw.Draw(dark, "RGBA")
for _ in range(160):
    x, y, r = rng.integers(0, W), rng.integers(0, int(H * 0.55)), rng.choice([2, 2, 3, 4])
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 245, 210, int(rng.integers(60, 190))))
sx, sy = 2010, 240                                                                              # satellite glyph
d.rectangle([sx - 18, sy - 18, sx + 18, sy + 18], fill=(217, 168, 58, 255))
for sgn in (-1, 1): d.rectangle([sx + sgn * 26 - (0 if sgn > 0 else 70), sy - 12, sx + sgn * 26 + (70 if sgn > 0 else 0), sy + 12], fill=(143, 184, 201, 230))
for r_, a in ((60, 70), (100, 40), (140, 20)): d.arc([sx - r_, sy - r_, sx + r_, sy + r_], 40, 140, fill=(217, 168, 58, a), width=3)  # signal arcs
terraces(dark, 930, 5, 34, 760, [(31, 77, 58, 255), (38, 94, 66, 255), (47, 122, 82, 255), (74, 150, 98, 255), (118, 176, 112, 255)])
dark = dark.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB"); dark.save("bg_title.png")

# ---------------- light content background
light = Image.fromarray(vgrad((255, 255, 255), (244, 249, 238))).convert("RGBA")
d = ImageDraw.Draw(light, "RGBA")
for i, r in enumerate((260, 380, 500)): d.ellipse([W - 170 - r, -190 - r, W - 170 + r, -190 + r], outline=(143, 184, 201, 40 - i * 9), width=3)  # faint orbit, top-right
terraces(light, 1245, 3, 14, 620, [(228, 240, 215, 255), (211, 230, 196, 255), (190, 217, 175, 255)])
light.convert("RGB").save("bg_content.png")
print("ok")
