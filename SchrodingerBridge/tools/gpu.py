import http.server
import socketserver
import subprocess
import threading
import webbrowser
import time
import sys
import queue
import tkinter as tk

# Enable high-DPI awareness on Windows to prevent blurriness and pixelation
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1) # PROCESS_SYSTEM_DPI_AWARE
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass

# Global state for high-performance broadcasting
latest_gpu_status = {}  # {gpu_id: latest_formatted_data_string}
clients = set()
clients_lock = threading.Lock()

# SSH process control
ssh_process = None
restart_ssh_flag = False
server_instance = None

class GPUFloatingWidget:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True) # Frameless
        self.root.attributes("-topmost", True) # Always on top
        self.root.configure(bg="#090d14") # Dark matching taskbar
        
        # Grid layout matching the screenshot exactly
        # Column 0: G0 and PWR
        # Column 1: VRAM and MU
        self.root.grid_columnconfigure(0, weight=1, minsize=90)
        self.root.grid_columnconfigure(1, weight=1, minsize=90)
        
        # Use Consolas bold for perfect alignment and retro look
        font_style = ("Consolas", 10, "bold")
        
        # Pre-populate with maximum possible string lengths to calculate the absolute max required geometry
        self.lbl_g0 = tk.Label(root, text="G0:100%", font=font_style, fg="#ffffff", bg="#090d14", anchor="w", padx=0, pady=0, bd=0, highlightthickness=0)
        self.lbl_g0.grid(row=0, column=0, padx=(12, 4), pady=(4, 2), sticky="w")
        
        self.lbl_vram = tk.Label(root, text="VRAM:100%", font=font_style, fg="#ffffff", bg="#090d14", anchor="w", padx=0, pady=0, bd=0, highlightthickness=0)
        self.lbl_vram.grid(row=0, column=1, padx=(4, 12), pady=(4, 2), sticky="w")
        
        self.lbl_pwr = tk.Label(root, text="PWR:999W", font=font_style, fg="#ffffff", bg="#090d14", anchor="w", padx=0, pady=0, bd=0, highlightthickness=0)
        self.lbl_pwr.grid(row=1, column=0, padx=(12, 4), pady=(2, 4), sticky="w")
        
        self.lbl_mu = tk.Label(root, text="MU:100%", font=font_style, fg="#ffffff", bg="#090d14", anchor="w", padx=0, pady=0, bd=0, highlightthickness=0)
        self.lbl_mu.grid(row=1, column=1, padx=(4, 12), pady=(2, 4), sticky="w")
        
        # Draggable window binds
        self.root.bind("<Button-1>", self.start_drag)
        self.root.bind("<B1-Motion>", self.drag)
        
        # Right-click context menu binds
        self.root.bind("<Button-3>", self.show_context_menu)
        self.menu = tk.Menu(self.root, tearoff=0, bg="#111827", fg="#ffffff", activebackground="#2563eb")
        self.menu.add_command(label="Open Web Dashboard", command=self.open_dashboard)
        self.menu.add_command(label="Restart SSH Connection", command=self.restart_ssh)
        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=self.quit_app)
        
        # Force layout calculation to auto-fit the DPI-scaled fonts perfectly
        self.root.update_idletasks()
        
        # Retrieve computed required bounds (in physical pixels)
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        
        # Get dynamic DPI scaling factor (e.g. 2.0 for 200% scaling)
        try:
            dpi = self.root.winfo_fpixels('1i')
            scaling = dpi / 96.0
        except:
            scaling = 1.0
            
        # Get physical screen dimensions (Tkinter runs in physical pixels when SetProcessDpiAwareness is active)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # Position window perfectly at the absolute bottom, in the right 1/5th position of the screen (in physical coordinates)
        # Start of the right 1/5th of the screen width
        x = int(screen_w * 0.8)
        # Sits at the absolute bottom of the screen (resting directly on/over the taskbar)
        y = int(screen_h - h)
        
        # Lock in position, and let Tkinter manage WxH automatically to prevent any clipping!
        self.root.geometry(f"+{x}+{y}")
        
        # Reset labels to default connecting state
        self.lbl_g0.config(text="G0:-")
        self.lbl_vram.config(text="VRAM:-")
        self.lbl_pwr.config(text="PWR:-")
        self.lbl_mu.config(text="MU:-")
        
        # Start periodic update checking
        self.root.after(200, self.update_stats)

    def start_drag(self, event):
        self.root._drag_x = event.x
        self.root._drag_y = event.y

    def drag(self, event):
        deltax = event.x - self.root._drag_x
        deltay = event.y - self.root._drag_y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def show_context_menu(self, event):
        self.menu.post(event.x_root, event.y_root)

    def open_dashboard(self):
        webbrowser.open(f"http://127.0.0.1:8085")

    def restart_ssh(self):
        global restart_ssh_flag, ssh_process
        restart_ssh_flag = True
        if ssh_process:
            try:
                ssh_process.terminate()
                print("[Floating GUI] Restart SSH requested.")
            except:
                pass

    def quit_app(self):
        global ssh_process
        print("Stopping monitor...")
        if ssh_process:
            try:
                ssh_process.terminate()
            except:
                pass
        import os
        os._exit(0)

    def update_stats(self):
        try:
            # Force window to stay at the very top of Z-order (above the taskbar)
            self.root.lift()
            self.root.attributes("-topmost", True)
            
            if latest_gpu_status:
                gpu_ids = sorted(latest_gpu_status.keys())
                if gpu_ids:
                    data = latest_gpu_status[gpu_ids[0]]
                    parts = [p.strip() for p in data.split(',')]
                    if len(parts) >= 6:
                        gpu_id = parts[0]
                        mem_used = parts[2]
                        pwr = parts[3]
                        util_gpu = parts[4]
                        u_mem_pct = parts[5]
                        util_mem = parts[6] if len(parts) > 6 else "0"
                        
                        self.lbl_g0.config(text=f"G{gpu_id}:{int(float(util_gpu))}%")
                        self.lbl_vram.config(text=f"VRAM:{int(float(u_mem_pct))}%")
                        self.lbl_pwr.config(text=f"PWR:{int(float(pwr))}W")
                        self.lbl_mu.config(text=f"MU:{int(float(util_mem))}%")
        except Exception as e:
            print(f"[Floating GUI] Data parsing warning: {e}")
            
        self.root.after(500, self.update_stats)

def gpu_data_worker():
    global latest_gpu_status, ssh_process, restart_ssh_flag
    # Using -T to disable pseudo-terminal, -o LogLevel=ERROR to be quiet
    cmd = ['ssh', '-p', '2222', '-T', '-o', 'LogLevel=ERROR', 'administrator@100.115.18.62', 
           'nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw,utilization.gpu,utilization.memory --format=csv -l 2']
    
    while True:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            ssh_process = p
            for line in iter(p.stdout.readline, ''):
                if restart_ssh_flag:
                    restart_ssh_flag = False
                    break
                line = line.strip()
                if not line or line.startswith('index'):
                    continue
                try:
                    parts = [pt.strip() for pt in line.split(',')]
                    if len(parts) < 7: continue
                    
                    idx, name, mem_used, mem_total, pwr, util_gpu, util_mem = parts
                    mem_used = mem_used.replace(' MiB', '')
                    mem_total = mem_total.replace(' MiB', '')
                    pwr = pwr.replace(' W', '')
                    util_gpu = util_gpu.replace(' %', '')
                    util_mem = util_mem.replace(' %', '')
                    
                    u_mem_pct = 0
                    try: u_mem_pct = (float(mem_used) / float(mem_total)) * 100
                    except: pass
                    
                    data = f"{idx},{name},{mem_used},{pwr},{util_gpu},{u_mem_pct:.1f},{util_mem},{mem_total}"
                    
                    # Store for instant load
                    latest_gpu_status[idx] = data
                    
                    # Broadcast to all connected queues
                    with clients_lock:
                        for q in list(clients):
                            try:
                                q.put_nowait(data)
                            except queue.Full:
                                pass 
                except Exception as e:
                    print(f"[SSH Worker] Inner parsing error: {e}")
            
            try: p.terminate()
            except: pass
            ssh_process = None
        except Exception as e:
            print(f"[SSH Worker] Spawn error: {e}")
            ssh_process = None
            
        time.sleep(2) # Cooldown before reconnect

# Start background fetcher
threading.Thread(target=gpu_data_worker, daemon=True).start()

HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GPU Dashboard | Dark Mode</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text-main: #c9d1d9;
            --text-dim: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background: var(--bg); 
            color: var(--text-main); 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            padding: 24px;
        }

        .header {
            flex: 0 0 auto;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .dashboard {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 4px;
            overflow-y: auto;
            height: calc(100vh - 120px);
        }

        .gpu-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 32px;
            display: flex;
            flex-direction: column;
            min-height: 800px; 
            flex: 1 0 auto;
            margin-bottom: 24px;
        }

        .card-top { 
            flex: 0 0 auto; 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            margin-bottom: 40px; 
        }
        .gpu-name { font-size: 64px; font-weight: 800; color: #fff; letter-spacing: -1px; }

        .metrics-main {
            display: flex;
            justify-content: space-between;
            gap: 30px;
            margin-bottom: 40px;
            padding: 30px 40px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
        }
        .metric { display: flex; flex-direction: column; align-items: center; flex: 1; }
        .m-label { font-size: 36px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; margin-bottom: 10px; }
        .m-value { font-size: 100px; font-weight: 950; color: var(--success); line-height: 1; text-shadow: 0 4px 20px rgba(63, 185, 80, 0.3); white-space: nowrap; }
        .m-unit { font-size: 36px; margin-left: 8px; color: var(--text-dim); font-weight: 500; }
        .pwr-tag { color: var(--accent) !important; text-shadow: 0 4px 20px rgba(88, 166, 255, 0.3); }

        .chart-container {
            flex: 1 1 auto;
            position: relative;
            background: #0d1117;
            border-radius: 4px;
            padding: 8px;
            min-height: 0;
            width: 100%;
        }

        .hover-card {
            display: none;
            position: absolute;
            z-index: 10;
            pointer-events: none;
            min-width: 380px;
            padding: 22px 26px;
            border: 1px solid #58a6ff;
            border-radius: 14px;
            background: rgba(13, 17, 23, 0.96);
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
            color: #c9d1d9;
            font-size: 30px;
            font-weight: 700;
            line-height: 1.35;
        }
        .hover-card .h-time { color: #8b949e; font-size: 24px; margin-bottom: 10px; }
        .hover-card .h-main { color: #fff; font-size: 42px; margin-bottom: 12px; }
        .hover-card .h-row { display: flex; justify-content: space-between; gap: 30px; font-size: 28px; color: #8b949e; }
        .hover-card .h-row span:last-child { color: #cbd5e1; }

        .footer { text-align: center; margin-top: 48px; color: var(--text-dim); font-size: 12px; opacity: 0.5; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="font-size: 64px;">GPU System Monitor</h1>
            <p style="color: var(--text-dim); font-size: 32px; margin-top: 12px;">Real-time Performance Metrics via SSH Stream</p>
        </div>
        <div id="clock" style="font-family: monospace; font-size: 48px; color: var(--accent); font-weight: bold;"></div>
    </div>

    <div id="dashboard" class="dashboard"></div>

    <div class="footer">Cluster Monitor v3.0 | Chart.js Stable Engine | SSH: 100.115.18.62</div>

<script>
    const charts = {};
    const MAX_HISTORY = 60;
    const DATASET_UNITS = {
        'GPU %': '%',
        'Mem %': '%',
        'Mem Util %': '%',
        'Power W': ' W',
        'VRAM MB': ' MB'
    };

    function formatClockLabel(date = new Date()) {
        return date.toLocaleTimeString([], { hour12: false });
    }

    function formatMetric(label, value) {
        const unit = DATASET_UNITS[label] || '';
        const precision = label.includes('%') ? 1 : 0;
        return `${Number(value || 0).toFixed(precision)}${unit}`;
    }

    function attachHoverCard(canvas, chart, card) {
        const moveCard = (evt) => {
            const points = chart.getElementsAtEventForMode(
                evt,
                'nearest',
                { intersect: false, axis: 'xy' },
                false
            );

            if (!points.length) {
                card.style.display = 'none';
                return;
            }

            const point = points[0];
            const dataset = chart.data.datasets[point.datasetIndex];
            const index = point.index;
            const label = dataset.label || '';
            const value = dataset.data[index] || 0;
            const timeLabel = chart.data.labels[index] || 'latest';
            const rows = chart.data.datasets.map(ds => {
                const color = ds.borderColor || '#8b949e';
                return `<div class="h-row"><span style="color:${color}">${ds.label}</span><span>${formatMetric(ds.label, ds.data[index])}</span></div>`;
            }).join('');

            card.innerHTML = `
                <div class="h-time">${timeLabel}</div>
                <div class="h-main" style="color:${dataset.borderColor || '#fff'}">${label}: ${formatMetric(label, value)}</div>
                ${rows}
            `;

            const containerRect = card.parentElement.getBoundingClientRect();
            const x = evt.clientX - containerRect.left + 18;
            const y = evt.clientY - containerRect.top + 18;
            card.style.display = 'block';
            const maxX = Math.max(8, containerRect.width - card.offsetWidth - 8);
            const maxY = Math.max(8, containerRect.height - card.offsetHeight - 8);
            card.style.left = `${Math.min(Math.max(8, x), maxX)}px`;
            card.style.top = `${Math.min(Math.max(8, y), maxY)}px`;
        };

        canvas.addEventListener('mousemove', moveCard);
        canvas.addEventListener('mouseleave', () => { card.style.display = 'none'; });
    }

    function ensureGPU(id, name) {
        if (charts[id]) return;

        const container = document.getElementById('dashboard');
        const card = document.createElement('div');
        card.className = 'gpu-card';
        card.id = `gpu-${id}`;
        card.innerHTML = `
            <div class="card-top">
                <div class="gpu-name">GPU ${id}: ${name}</div>
            </div>
            <div class="metrics-main">
                <div class="metric">
                    <div class="m-label">Utilization</div>
                    <div class="m-value"><span id="util-v-${id}">0</span><span class="m-unit">%</span></div>
                </div>
                <div class="metric">
                    <div class="m-label">VRAM Usage</div>
                    <div class="m-value"><span id="mem-v-${id}">0</span><span class="m-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="m-label">Power</div>
                    <div class="m-value pwr-tag"><span id="pwr-${id}">0</span><span class="m-unit">W</span></div>
                </div>
                <div class="metric">
                    <div class="m-label">Mem Util</div>
                    <div class="m-value" style="color: #00d2ff; text-shadow: 0 4px 20px rgba(0, 210, 255, 0.3);"><span id="mem-u-${id}">0</span><span class="m-unit">%</span></div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="chart-${id}"></canvas>
                <div class="hover-card" id="hover-${id}"></div>
            </div>
        `;
        container.appendChild(card);

        const canvas = document.getElementById(`chart-${id}`);
        const hoverCard = document.getElementById(`hover-${id}`);
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(MAX_HISTORY).fill(''),
                datasets: [
                    {
                        label: 'GPU %',
                        data: Array(MAX_HISTORY).fill(0),
                        borderColor: '#58a6ff',
                        backgroundColor: 'rgba(88, 166, 255, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 8,
                        pointHitRadius: 18,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Mem %',
                        data: Array(MAX_HISTORY).fill(0),
                        borderColor: '#3fb950',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 8,
                        pointHitRadius: 18,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Power W',
                        data: Array(MAX_HISTORY).fill(0),
                        borderColor: '#d29922',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 8,
                        pointHitRadius: 18,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'VRAM MB',
                        data: Array(MAX_HISTORY).fill(0),
                        borderColor: '#bc8cff',
                        borderWidth: 1,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 8,
                        pointHitRadius: 18,
                        yAxisID: 'y2'
                    },
                    {
                        label: 'Mem Util %',
                        data: Array(MAX_HISTORY).fill(0),
                        borderColor: '#00d2ff',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 8,
                        pointHitRadius: 18,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: 'nearest',
                    axis: 'xy',
                    intersect: false
                },
                hover: {
                    mode: 'nearest',
                    intersect: false
                },
                plugins: { 
                    legend: { 
                        display: true, 
                        labels: { color: '#8b949e', boxWidth: 40, font: { size: 36, weight: 'bold' } },
                        position: 'top',
                        align: 'end'
                    },
                    tooltip: {
                        enabled: false,
                        mode: 'nearest',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { display: false },
                        ticks: {
                            color: '#8b949e',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 6,
                            font: { size: 22, weight: 'bold' }
                        }
                    },
                    y: { 
                        min: 0, 
                        max: 100, 
                        grid: { color: '#21262d' },
                        ticks: { color: '#58a6ff', font: { size: 28, weight: 'bold' } },
                        title: { display: true, text: 'Utilization %', color: '#58a6ff', font: { size: 32, weight: 'bold' } }
                    },
                    y1: {
                        position: 'right',
                        grid: { display: false },
                        ticks: { color: '#d29922', font: { size: 28, weight: 'bold' } },
                        title: { display: true, text: 'Power W', color: '#d29922', font: { size: 32, weight: 'bold' } }
                    },
                    y2: {
                        position: 'right',
                        grid: { display: false },
                        ticks: { color: '#bc8cff', font: { size: 28, weight: 'bold' } },
                        title: { display: true, text: 'VRAM MB', color: '#bc8cff', font: { size: 32, weight: 'bold' } }
                    }
                }
            }
        });
        charts[id] = chart;
        attachHoverCard(canvas, chart, hoverCard);
    }

    function updateData(msg) {
        try {
            const parts = msg.split(',').map(s => s.trim());
            if (parts.length < 6) return;

            const [id, name, vramStr, pwrStr, utilStr, memPctStr] = parts;
            ensureGPU(id, name);

            const vram = parseInt(vramStr) || 0;
            const pwr = parseInt(pwrStr) || 0;
            const util = parseFloat(utilStr) || 0;
            const memPct = parseFloat(memPctStr) || 0;
            const utilMem = parts.length > 6 ? parseFloat(parts[6]) || 0 : 0;

            // Update UI Text
            const utilEl = document.getElementById(`util-v-${id}`);
            const memEl = document.getElementById(`mem-v-${id}`);
            const pwrEl = document.getElementById(`pwr-${id}`);
            const memUEl = document.getElementById(`mem-u-${id}`);
            
            if (utilEl) utilEl.innerText = Math.round(util);
            if (memEl) memEl.innerText = vram;
            if (pwrEl) pwrEl.innerText = pwr;
            if (memUEl) memUEl.innerText = Math.round(utilMem);

            // Update Chart Datasets
            const chart = charts[id];
            if (chart) {
                chart.data.labels.push(formatClockLabel());
                if (chart.data.labels.length > MAX_HISTORY) chart.data.labels.shift();

                chart.data.datasets[0].data.push(util);
                chart.data.datasets[1].data.push(memPct);
                chart.data.datasets[2].data.push(pwr);
                chart.data.datasets[3].data.push(vram);
                chart.data.datasets[4].data.push(utilMem);

                chart.data.datasets.forEach(ds => {
                    if (ds.data.length > MAX_HISTORY) ds.data.shift();
                });

                chart.update('none'); 
            }
        } catch (e) {
            console.error("[JS] Update Error:", e, msg);
        }
    }

    const evt = new EventSource("/stream");
    evt.onmessage = (e) => { if (e.data) updateData(e.data); };
    evt.onerror = (err) => { console.error("[JS] SSE Error:", err); };

    setInterval(() => {
        const clock = document.getElementById('clock');
        if (clock) clock.innerText = new Date().toLocaleTimeString();
    }, 1000);
</script>
</body>
</html>"""


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return # Silence logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # 1. Send latest snapshot immediately for "instant load"
            for data in list(latest_gpu_status.values()):
                self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
            self.wfile.flush()

            # 2. Register queue for real-time updates
            q = queue.Queue(maxsize=100)
            with clients_lock:
                clients.add(q)
            
            try:
                while True:
                    try:
                        data = q.get(timeout=2) # Wait for new data
                        self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                        self.wfile.flush()
                    except queue.Empty:
                        # Send keep-alive comment to keep connection open
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                with clients_lock:
                    clients.discard(q)

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    port = 8085
    
    # Parse arguments
    headless = ("--headless" in sys.argv)
    
    # Set up signal handler for robust Ctrl+C exits on the main thread
    import signal
    def sigint_handler(signum, frame):
        if 'widget' in globals() and widget:
            widget.quit_app()
        else:
            print("\nStopping GPU Monitor...")
            sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)
    
    try:
        server = ThreadingTCPServer(('127.0.0.1', port), Handler)
        server_instance = server
        print(f"Server active at http://127.0.0.1:{port}")
        
        if headless:
            print("Running in headless mode without GUI.")
            server.serve_forever()
        else:
            # Start server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            
            # Start Tkinter main loop on the main thread for the draggable floating window
            print("Launching draggable taskbar GUI widget...")
            root = tk.Tk()
            widget = GPUFloatingWidget(root)
            
            # Override callback exception handler to cleanly exit on KeyboardInterrupt inside Tkinter
            def report_callback_exception(exc, val, tb):
                if issubclass(exc, KeyboardInterrupt):
                    widget.quit_app()
                else:
                    sys.__excepthook__(exc, val, tb)
            root.report_callback_exception = report_callback_exception
            
            # Periodically allow Python to process signals like Ctrl+C
            def check_signals():
                root.after(100, check_signals)
            root.after(100, check_signals)
            
            root.protocol("WM_DELETE_WINDOW", widget.quit_app)
            root.mainloop()
            
    except KeyboardInterrupt:
        print("\nStopping GPU Monitor...")
        if ssh_process:
            try: ssh_process.terminate()
            except: pass
        if server_instance:
            try: server_instance.shutdown()
            except: pass
        sys.exit(0)
    except Exception as e:
        print(f"Critical error: {e}")
