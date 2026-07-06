#!/usr/bin/env python3
"""
Generate SVG from TikZ framework diagram coordinates.
SVG elements are fully editable - you can move boxes around in any SVG editor.
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuration (in cm, matching TikZ coordinates)
SCALE = 40  # pixels per cm
MARGIN = 30  # pixels
SVG_WIDTH = (11 + 1) * SCALE + 2 * MARGIN
SVG_HEIGHT = (11) * SCALE + 2 * MARGIN

# Colors matching TikZ
COLORS = {
    'orange': '#E8A87C',
    'blue': '#7FA1D4', 
    'orange_dark': '#D8860D',
    'blue_dark': '#3D5A80',
    'teal': '#4A9B9E',
    'gray': '#CCCCCC',
}

# TikZ coordinates (x, y) in cm
nodes = {
    'sid': {'pos': (0.65, 8.4), 'size': (1.55, 0.55), 'text': 'style id $s$', 'style': 'orange'},
    'scode': {'pos': (0.65, 7.25), 'size': (1.55, 0.65), 'text': 'style code\\n$w$', 'style': 'orange'},
    'sprior': {'pos': (0.65, 5.95), 'size': (1.55, 0.65), 'text': 'style spatial\\nprior $M_s^{sp}$', 'style': 'orange'},
    'smap': {'pos': (0.65, 4.70), 'size': (1.55, 0.65), 'text': '$16×16$\\nstyle map $M_s$', 'style': 'orange'},
    
    'zin': {'pos': (4.8, 8.4), 'size': (2.00, 0.60), 'text': 'input latent $z_t$', 'style': 'blue'},
    'lift': {'pos': (4.8, 7.25), 'size': (2.00, 0.65), 'text': 'lift stem $4→128$ ch', 'style': 'blue'},
    'hires': {'pos': (4.8, 5.95), 'size': (2.00, 0.70), 'text': 'hi-res blocks\\n$32×32$ feature', 'style': 'blue'},
    'down': {'pos': (4.8, 4.70), 'size': (1.40, 0.60), 'text': 'down $32→16$ ch', 'style': 'blue'},
    'body': {'pos': (4.8, 3.25), 'size': (2.50, 0.78), 'text': 'semantic cross-attn\\nbottleneck injection', 'style': 'orange'},
    'up': {'pos': (4.8, 1.85), 'size': (1.40, 0.60), 'text': 'up $16→32$ ch', 'style': 'blue'},
    'fuse': {'pos': (4.8, 0.65), 'size': (1.70, 0.65), 'text': 'skip fuse\\nstructure', 'style': 'blue'},
    'dec': {'pos': (7.5, 0.65), 'size': (2.00, 0.65), 'text': 'decoder +\\n$v_θ$', 'style': 'blue'},
    'endp': {'pos': (7.5, -0.45), 'size': (1.30, 0.55), 'text': 'integrate\\n$ẑ_1$', 'style': 'gray'},
    
    'qbox': {'pos': (0.60, -1.6), 'size': (1.20, 0.55), 'text': '$Q$\\ncontent', 'style': 'orange'},
    'kvbox': {'pos': (2.05, -1.6), 'size': (1.20, 0.55), 'text': '$K,V$\\nstyle', 'style': 'orange'},
    'soft': {'pos': (3.50, -1.6), 'size': (1.30, 0.55), 'text': '$A=softmax$\\n$(QK^T)$', 'style': 'gray'},
    'paint': {'pos': (4.95, -1.6), 'size': (1.20, 0.55), 'text': '$AV$\\noutput', 'style': 'orange'},
    
    'swd': {'pos': (0.8, -3.75), 'size': (2.30, 0.75), 'text': 'terminal SWD\\nendpoint style', 'style': 'teal'},
    'kin': {'pos': (4.8, -3.75), 'size': (2.30, 0.75), 'text': 'kinetic\\ntrajectory energy', 'style': 'teal'},
    'fm': {'pos': (8.8, -3.75), 'size': (2.30, 0.75), 'text': 'flow matching\\nvelocity field', 'style': 'teal'},
}

def cm_to_px(x, y):
    """Convert TikZ cm coordinates to SVG pixels"""
    return (x * SCALE + MARGIN, SVG_HEIGHT - (y * SCALE + MARGIN))

def create_rect(parent, x, y, w, h, style_type, text, node_id):
    """Create a rectangle with text"""
    px, py = cm_to_px(x - w/2, y + h/2)
    width = w * SCALE
    height = h * SCALE
    
    # Determine colors based on style
    if style_type == 'orange':
        fill = '#FFE5CC'
        stroke = COLORS['orange_dark']
    elif style_type == 'blue':
        fill = '#E3ECFF'
        stroke = COLORS['blue_dark']
    elif style_type == 'teal':
        fill = '#D4F5F6'
        stroke = COLORS['teal']
    else:  # gray
        fill = '#F5F5F5'
        stroke = '#999999'
    
    # Create rectangle
    rect = ET.SubElement(parent, 'rect',
        {
            'x': str(px),
            'y': str(py),
            'width': str(width),
            'height': str(height),
            'fill': fill,
            'stroke': stroke,
            'stroke-width': '1.5',
            'rx': '3',
            'class': f'node {node_id}'
        }
    )
    
    # Create text
    text_elem = ET.SubElement(parent, 'text',
        {
            'x': str(px + width/2),
            'y': str(py + height/2),
            'text-anchor': 'middle',
            'dominant-baseline': 'central',
            'font-family': 'Arial, sans-serif',
            'font-size': '10px',
            'fill': '#000000',
            'class': f'label {node_id}'
        }
    )
    text_elem.text = text

# Create SVG
root = ET.Element('svg', {
    'xmlns': 'http://www.w3.org/2000/svg',
    'width': str(SVG_WIDTH),
    'height': str(SVG_HEIGHT),
    'viewBox': f'0 0 {SVG_WIDTH} {SVG_HEIGHT}',
})

# Add style
style = ET.SubElement(root, 'style')
style.text = '''
.node { cursor: move; }
.label { cursor: text; pointer-events: none; }
text { user-select: none; }
'''

# Add background
bg = ET.SubElement(root, 'rect', {
    'width': '100%',
    'height': '100%',
    'fill': 'white'
})

# Create a group for all nodes
g_nodes = ET.SubElement(root, 'g', {'id': 'nodes'})

# Add all nodes
for node_id, node_data in nodes.items():
    x, y = node_data['pos']
    w, h = node_data['size']
    text = node_data['text']
    style = node_data['style']
    create_rect(g_nodes, x, y, w, h, style, text, node_id)

# Add labels
g_labels = ET.SubElement(root, 'g', {'id': 'labels', 'font-size': '11px', 'font-weight': 'bold'})

# Section labels
labels_data = [
    (0.4, 9.2, 'Style branch', '#D8860D'),
    (4.8, 9.2, 'Latent trunk', '#3D5A80'),
    (0.4, -0.8, 'Cross-attn detail', '#D8860D'),
    (0.4, -2.6, 'Training objectives', '#4A9B9E'),
]

for x, y, text, color in labels_data:
    px, py = cm_to_px(x, y)
    label = ET.SubElement(g_labels, 'text', {
        'x': str(px),
        'y': str(py),
        'text-anchor': 'start',
        'fill': color,
    })
    label.text = text

# Pretty print
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

# Write to file
output_path = 'g:\\GitHub\\Latent_Style\\SchrodingerBridge\\paper_refine_v2\\final\\framework_figure.svg'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(xmlstr)

print(f"SVG generated: {output_path}")
print(f"Size: {SVG_WIDTH}x{SVG_HEIGHT} pixels")
print("\nYou can now:")
print("1. Open this SVG in Inkscape or any SVG editor")
print("2. Move boxes around to adjust spacing")
print("3. Add or edit text labels")
print("4. Export to PDF when done")
