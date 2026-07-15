import http.server
import socketserver
import subprocess
import threading
import webbrowser
import time
import sys
import queue

# Global state for high-performance broadcasting
latest_gpu_status = {}  # {gpu_id: latest_formatted_data_string}
clients = set()
clients_lock = threading.Lock()

def gpu_data_worker():
    global latest_gpu_status
    # Using -T to disable pseudo-terminal, -o LogLevel=ERROR to be quiet
    cmd = ['ssh', '-p', '2222', '-T', '-o', 'LogLevel=ERROR', 'administrator@100.115.18.62', 
           'nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw,utilization.gpu,utilization.memory --format=csv -l 2']
    
    while True:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            for line in iter(p.stdout.readline, ''):
                line = line.strip()
                if not line or line.startswith('index'):
                    continue
                try:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 7: continue
                    
                    idx, name, mem_used, mem_total, pwr, util_gpu, util_mem = parts
                    mem_used = mem_used.replace(' MiB', '')
                    mem_total = mem_total.replace(' MiB', '')
                    pwr = pwr.replace(' W', '')
                    util_gpu = util_gpu.replace(' %', '')
                    
                    u_mem_pct = 0
                    try: u_mem_pct = (float(mem_used) / float(mem_total)) * 100
                    except: pass
                    
                    data = f"{idx},{name},{mem_used},{pwr},{util_gpu},{u_mem_pct}"
                    
                    # Store for instant load
                    latest_gpu_status[idx] = data
                    
                    # Broadcast to all connected queues
                    with clients_lock:
                        for q in list(clients):
                            try:
                                q.put_nowait(data)
                            except queue.Full:
                                pass 
                except:
                    pass
        except:
            pass
        finally:
            try: p.terminate()
            except: pass
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
            gap: 40px;
            margin-bottom: 40px;
            padding: 40px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
        }
        .metric { display: flex; flex-direction: column; align-items: center; flex: 1; }
        .m-label { font-size: 48px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; margin-bottom: 10px; }
        .m-value { font-size: 140px; font-weight: 950; color: var(--success); line-height: 1; text-shadow: 0 4px 20px rgba(63, 185, 80, 0.3); white-space: nowrap; }
        .m-unit { font-size: 48px; margin-left: 12px; color: var(--text-dim); font-weight: 500; }
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
        .hover-card .h-row span:last-child { color: #c9d1d9; }

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

    <div class="footer">Cluster Monitor v2.1 | Chart.js Stable Engine | SSH: 100.115.18.62</div>

<script>
    const charts = {};
    const MAX_HISTORY = 60;
    const DATASET_UNITS = {
        'GPU %': '%',
        'Mem %': '%',
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
                        intersect: false,
                        backgroundColor: 'rgba(13, 17, 23, 0.96)',
                        borderColor: '#58a6ff',
                        borderWidth: 1,
                        titleColor: '#c9d1d9',
                        bodyColor: '#fff',
                        displayColors: true,
                        padding: 16,
                        caretSize: 10,
                        cornerRadius: 8,
                        titleFont: { size: 28, weight: 'bold' },
                        bodyFont: { size: 32, weight: 'bold' },
                        callbacks: {
                            title(items) {
                                if (!items || !items.length) return '';
                                return items[0].label || 'latest';
                            },
                            label(context) {
                                const label = context.dataset.label || '';
                                const unit = DATASET_UNITS[label] || '';
                                const value = Number(context.parsed.y || 0);
                                const precision = label.includes('%') ? 1 : 0;
                                return `${label}: ${value.toFixed(precision)}${unit}`;
                            }
                        }
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

            // Update UI Text
            const utilEl = document.getElementById(`util-v-${id}`);
            const memEl = document.getElementById(`mem-v-${id}`);
            const pwrEl = document.getElementById(`pwr-${id}`);
            
            if (utilEl) utilEl.innerText = Math.round(util);
            if (memEl) memEl.innerText = vram;
            if (pwrEl) pwrEl.innerText = pwr;

            // Update Chart Datasets
            const chart = charts[id];
            if (chart) {
                chart.data.labels.push(formatClockLabel());
                if (chart.data.labels.length > MAX_HISTORY) chart.data.labels.shift();

                chart.data.datasets[0].data.push(util);
                chart.data.datasets[1].data.push(memPct);
                chart.data.datasets[2].data.push(pwr);
                chart.data.datasets[3].data.push(vram);

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
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                with clients_lock:
                    clients.discard(q)

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    port = 8085
    try:
        server = ThreadingTCPServer(('127.0.0.1', port), Handler)
        print(f"Server active at http://127.0.0.1:{port}")
        threading.Thread(target=webbrowser.open, args=(f'http://127.0.0.1:{port}',)).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Critical error: {e}")

