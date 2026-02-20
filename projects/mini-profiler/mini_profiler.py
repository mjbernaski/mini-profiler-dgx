#!/usr/bin/env python3
"""Mini Profiler - Compact System Monitor for DGX Spark (GB10 unified memory)."""

import json
import socket
import subprocess
import time

import psutil
from bottle import Bottle, response, run

app = Bottle()

START_TIME = time.time()


def get_uptime_str():
    secs = int(time.time() - START_TIME)
    boot = time.time() - psutil.boot_time()
    h, m = int(boot // 3600), int((boot % 3600) // 60)
    return f"{h}h {m}m"


def get_gpu_info():
    """Get GPU utilization, temp, power, and process memory via nvidia-smi."""
    info = {"util": 0, "temp": 0, "power": 0.0, "power_max": 0.0, "proc_mem_mb": 0}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        info["util"] = int(parts[0]) if parts[0] not in ("[N/A]", "N/A", "") else 0
        info["temp"] = int(parts[1]) if parts[1] not in ("[N/A]", "N/A", "") else 0
        info["power"] = float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else 0
        info["power_max"] = float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else 0
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        total = 0
        for line in out.splitlines():
            line = line.strip()
            if line and line not in ("[N/A]", "N/A", ""):
                total += int(line)
        info["proc_mem_mb"] = total
    except Exception:
        pass
    return info


@app.route("/api/stats")
def stats():
    response.content_type = "application/json"
    mem = psutil.virtual_memory()
    gpu = get_gpu_info()
    return json.dumps({
        "hostname": socket.gethostname(),
        "uptime": get_uptime_str(),
        "ram_used_gb": round(mem.used / (1024 ** 3), 1),
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "ram_pct": mem.percent,
        "gpu_util": gpu["util"],
        "gpu_temp": gpu["temp"],
        "gpu_power": round(gpu["power"], 1),
        "gpu_power_max": round(gpu["power_max"], 1),
        "gpu_proc_mem_mb": gpu["proc_mem_mb"],
    })


@app.route("/")
def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mini Profiler</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;font-size:12px;width:100%;padding:6px;user-select:none;overflow:hidden}
.header{text-align:center;padding:2px 0 5px;border-bottom:1px solid #333}
.hostname{font-size:14px;font-weight:700;color:#00d4ff}
.uptime{font-size:10px;color:#888;margin-top:1px}
.section{margin-top:6px}
.label{display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px}
.label .val{color:#aaa}
.bar-wrap{background:#2a2a3e;border-radius:3px;height:16px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:3px;transition:width .5s ease}
.bar-text{position:absolute;top:0;left:0;right:0;text-align:center;line-height:16px;font-size:10px;font-weight:600;color:#fff;text-shadow:0 0 3px rgba(0,0,0,.8)}
.ram-fill{background:linear-gradient(90deg,#00b4d8,#0077b6)}
.gpu-fill{background:linear-gradient(90deg,#f72585,#b5179e)}
.info-row{display:flex;justify-content:space-between;margin-top:4px;font-size:11px;color:#bbb}
.info-row span{display:inline-block}
.info-val{color:#fff;font-weight:600}
.chart-section{margin-top:6px}
.chart-title{font-size:10px;color:#888;margin-bottom:2px;display:flex;justify-content:space-between}
canvas{width:100%;height:40px;display:block;border-radius:3px;background:#2a2a3e}
.sep{height:1px;background:#333;margin-top:6px}
</style>
</head>
<body>
<div class="header">
  <div class="hostname" id="hostname">---</div>
  <div class="uptime" id="uptime">uptime: ---</div>
</div>

<div class="section">
  <div class="label"><span>System RAM</span><span class="val" id="ram-detail">-</span></div>
  <div class="bar-wrap">
    <div class="bar-fill ram-fill" id="ram-bar" style="width:0%"></div>
    <div class="bar-text" id="ram-text">-</div>
  </div>
</div>

<div class="section">
  <div class="label"><span>GPU Utilization</span><span class="val" id="gpu-detail">-</span></div>
  <div class="bar-wrap">
    <div class="bar-fill gpu-fill" id="gpu-bar" style="width:0%"></div>
    <div class="bar-text" id="gpu-text">-</div>
  </div>
</div>

<div class="info-row">
  <span>Temp: <span class="info-val" id="gpu-temp">-</span></span>
  <span>Power: <span class="info-val" id="gpu-power">-</span></span>
  <span>Proc Mem: <span class="info-val" id="gpu-proc">-</span></span>
</div>

<div class="sep"></div>

<div class="chart-section">
  <div class="chart-title"><span>RAM % (5 min)</span><span id="ram-chart-val">-</span></div>
  <canvas id="ram-chart" height="40"></canvas>
</div>
<div class="chart-section">
  <div class="chart-title"><span>GPU % (5 min)</span><span id="gpu-chart-val">-</span></div>
  <canvas id="gpu-chart" height="40"></canvas>
</div>

<script>
const MAX_POINTS = 150; // 5 min at 2s intervals
const ramHistory = [];
const gpuHistory = [];

function drawChart(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (data.length < 2) return;

  // grid lines
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 0.5;
  for (let pct of [25, 50, 75]) {
    const y = h - (pct / 100) * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // sparkline fill
  const step = w / (MAX_POINTS - 1);
  const startX = w - (data.length - 1) * step;
  ctx.beginPath();
  ctx.moveTo(startX, h);
  for (let i = 0; i < data.length; i++) {
    const x = startX + i * step;
    const y = h - (data[i] / 100) * h;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.lineTo(startX + (data.length - 1) * step, h);
  ctx.closePath();
  ctx.fillStyle = color + '30';
  ctx.fill();

  // sparkline stroke
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = startX + i * step;
    const y = h - (data[i] / 100) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

async function poll() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();

    document.getElementById('hostname').textContent = d.hostname;
    document.getElementById('uptime').textContent = 'uptime: ' + d.uptime;

    // RAM
    document.getElementById('ram-detail').textContent = d.ram_used_gb + ' / ' + d.ram_total_gb + ' GB';
    document.getElementById('ram-bar').style.width = d.ram_pct + '%';
    document.getElementById('ram-text').textContent = d.ram_pct + '%';

    // GPU
    document.getElementById('gpu-detail').textContent = d.gpu_util + '%';
    document.getElementById('gpu-bar').style.width = d.gpu_util + '%';
    document.getElementById('gpu-text').textContent = d.gpu_util + '%';

    document.getElementById('gpu-temp').textContent = d.gpu_temp + '\u00B0C';
    const pw = d.gpu_power_max > 0 ? d.gpu_power + '/' + d.gpu_power_max + 'W' : d.gpu_power + 'W';
    document.getElementById('gpu-power').textContent = pw;
    const pm = d.gpu_proc_mem_mb > 1024 ? (d.gpu_proc_mem_mb / 1024).toFixed(1) + 'G' : d.gpu_proc_mem_mb + 'M';
    document.getElementById('gpu-proc').textContent = pm;

    // History
    ramHistory.push(d.ram_pct);
    gpuHistory.push(d.gpu_util);
    if (ramHistory.length > MAX_POINTS) ramHistory.shift();
    if (gpuHistory.length > MAX_POINTS) gpuHistory.shift();

    document.getElementById('ram-chart-val').textContent = d.ram_pct + '%';
    document.getElementById('gpu-chart-val').textContent = d.gpu_util + '%';
    drawChart('ram-chart', ramHistory, '#00b4d8');
    drawChart('gpu-chart', gpuHistory, '#f72585');
  } catch (e) {
    console.error('Poll error:', e);
  }
}

poll();
setInterval(poll, 2000);
window.addEventListener('resize', () => {
  drawChart('ram-chart', ramHistory, '#00b4d8');
  drawChart('gpu-chart', gpuHistory, '#f72585');
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Mini Profiler running at http://localhost:6048")
    run(app, host="0.0.0.0", port=6048, quiet=True)
