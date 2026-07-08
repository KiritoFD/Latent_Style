import re
from xml.etree import ElementTree as ET

in_path = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio.svg"
out_svg = r"F:\aaai_arch_diagram_v16_staggered_bundle.drawio_light.svg"
out_png = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027_v4\framework_sfm_main.png"

with open(in_path, 'r', encoding='utf-8') as f:
    txt = f.read()

# --- strategy: global string replacements for common dark-theme colors ---

# 1. mxGraphModel background attribute
# e.g. background="#18181b" -> background="#ffffff"
txt = re.sub(r'background="(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})"', r'background="#ffffff"', txt, count=1)

# 2. SVG background color style
# e.g. background-color:#303030; or background-color: var(--ge-adaptive-bg, #303030);
# We'll replace the fallback dark colors with #ffffff inside background-color declarations.
txt = re.sub(
    r'(background-color:\s*var\(--ge-adaptive-bg,\s*)(#[0-9a-fA-F]{6})(\))',
    r'\1#ffffff\3', txt
)
txt = re.sub(r'background-color:\s*#303030', 'background-color: #ffffff', txt)
txt = re.sub(r'background-color:\s*#18181b', 'background-color: #ffffff', txt)
txt = re.sub(r'background-color:\s*#111827', 'background-color: #ffffff', txt)
txt = re.sub(r'background-color:\s*#1f2937', 'background-color: #ffffff', txt)
txt = re.sub(r'background-color:\s*#000000', 'background-color: #ffffff', txt)

# 3. Dark rectangle fills (common card backgrounds in dark mode)
# Replace dark fills with #ffffff ONLY when they are not inside text or path stroke.
# But to be safe, do targeted replacements on exact colors.
dark_fills = ['#1F2937', '#111827', '#111111', '#1e1e1e', '#18181b', '#1e2937', '#0f172a']
for c in dark_fills:
    # case-insensitive replacement for fill=... in tags (rect, path, ellipse, etc.)
    pat = re.compile(re.escape('fill="' + c + '"'), re.IGNORECASE)
    txt = pat.sub('fill="#ffffff"', txt)
    pat2 = re.compile(re.escape("fill='" + c + "'"), re.IGNORECASE)
    txt = pat2.sub("fill='#ffffff'", txt)

# 4. White foregrounds -> black (text, strokes, lines)
# But careful: some #ffffff might be intentional white highlights; in light theme we need them dark.
light_fgs = ['#ffffff', '#f3f4f6', '#f8f8f8', '#f9fafb']
for c in light_fgs:
    # fill (text color usually)
    pat = re.compile(re.escape('fill="' + c + '"'), re.IGNORECASE)
    txt = pat.sub('fill="#000000"', txt)
    pat2 = re.compile(re.escape("fill='" + c + "'"), re.IGNORECASE)
    txt = pat2.sub("fill='#000000'", txt)
    # stroke (lines/arrows)
    pat_s = re.compile(re.escape('stroke="' + c + '"'), re.IGNORECASE)
    txt = pat_s.sub('stroke="#000000"', txt)
    pat_s2 = re.compile(re.escape("stroke='" + c + "'"), re.IGNORECASE)
    txt = pat_s2.sub("stroke='#000000'", txt)

# 5. Some dark strokes used for borders -> keep as #000000 or slightly lighter
# No change needed.

# 6. light-dark CSS function: force light mode by replacing with first arg
# e.g. light-dark(rgb(255,255,255),rgb(0,0,0)) -> rgb(255,255,255)
# But we want light mode first arg. If the SVG was generated for dark, the second arg is dark.
# Actually light-dark() returns first arg in light mode, second in dark mode.
# Our file likely was exported from drawio dark theme, so the dark color is the SECOND arg.
# To force light mode we can replace light-dark(A, B) -> A  (light mode result).
# But if A is light and B is dark, in light theme we'd get A (light) which is wrong for a dark-theme SVG.
# Let's inspect what the args are first. Instead, safer to remove the function and use the dark value (second arg) on white background? No, that would keep dark.
# Actually the current SVG already renders dark. So drawio used `light-dark(light, dark)` meaning light arg is for light theme, dark arg for dark theme.
# Since we are converting to light theme, we should use the FIRST argument.
# But we also want the result to be dark-on-white, not light-on-white.
# Hmm, this means the SVG already encodes two themes. Let's look at the snippet earlier:
# 'background-color var(--ge-adaptive-bg, #303030)' and 'background-color light-dark(rgb(255,' ...
# The light-dark snippet was cut off. It might be light-dark(rgb(255,255,255), rgb(17,24,39)) for example.
# For now, let's force replace light-dark(...) by extracting the FIRST argument.

def repl_light_dark(m):
    args = m.group(1)
    # find the first comma at top level (not inside parentheses)
    depth = 0
    comma_idx = None
    for i, ch in enumerate(args):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            comma_idx = i
            break
    if comma_idx is not None:
        first = args[:comma_idx].strip()
        return first
    return args

txt = re.sub(r'light-dark\(([^)]+)\)', repl_light_dark, txt)

with open(out_svg, 'w', encoding='utf-8') as f:
    f.write(txt)
print('saved light SVG to', out_svg)
