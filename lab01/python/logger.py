"""
logger.py - Ghi dữ liệu UART từ STM32 Nucleo-L476RG, vẽ đồ thị và tính jitter.

Cách chạy:
    python logger.py COM3 60
        COM3 : cổng "STMicroelectronics STLink Virtual COM Port" (xem Device Manager)
        60   : thời gian ghi (giây), mặc định 60

Mỗi dòng MCU gửi:  seq,t_tick_us,lat_us,adc_raw,adc_filt
"""
import sys
import csv
import time

import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
BAUD = 115200
CSV_FILE = "data.csv"
FIG_FILE = "plot.png"
PERIOD_US = 10000          # chu kỳ lý thuyết 100 Hz = 10 ms

# ---------------------------------------------------------------- 1. Ghi dữ liệu
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    sys.exit(f"Không mở được {PORT}: {e}\n"
             "-> Kiểm tra đúng cổng COM, và đóng mọi terminal/Putty đang dùng cổng này.")

ser.reset_input_buffer()
rows = []
t0 = time.perf_counter()
last_print = t0
print(f"Đang ghi {DURATION:.0f} s từ {PORT} ...")
while time.perf_counter() - t0 < DURATION:
    p = ser.readline().decode(errors="ignore").strip().split(",")
    if len(p) == 5 and all(x.isdigit() for x in p):
        rows.append([int(x) for x in p] + [time.perf_counter()])
    if time.perf_counter() - last_print >= 5:
        print(f"  {len(rows)} dòng ...")
        last_print = time.perf_counter()
ser.close()

rows = rows[1:]            # bỏ dòng đầu (có thể bị cắt dở)
if len(rows) < 20:
    sys.exit(f"Chỉ nhận được {len(rows)} dòng hợp lệ. Kiểm tra cổng COM, baud 115200, "
             "và board đã chạy chương trình (bấm nút RESET đen).")

with open(CSV_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seq", "t_tick_us", "lat_us", "adc_raw", "adc_filt", "host_s"])
    w.writerows(rows)
print(f"Đã lưu {len(rows)} dòng vào {CSV_FILE}")

# ---------------------------------------------------------------- 2. Phân tích
d = np.array(rows, dtype=np.float64)
seq, t_tick, lat, raw, filt, host = d.T
n = len(seq)

ok = np.diff(seq) == 1                                   # chỉ lấy cặp mẫu liên tiếp
lost = int((~ok).sum())
dt_us = np.diff(t_tick.astype(np.int64)) % (1 << 32)     # chu kỳ MCU đo, xử lý tràn 32-bit
dt_ok = dt_us[ok]
dh_ms = np.diff(host)[ok] * 1000.0                       # chu kỳ PC đo (tham khảo)
t_s = np.concatenate([[0], np.cumsum(dt_us)]) / 1e6

f_mcu = 1e6 / dt_ok.mean()
f_host = (n - 1) / (host[-1] - host[0])

print("\n========== KẾT QUẢ ==========")
print(f"Số mẫu: {n}    mất mẫu (seq nhảy): {lost}")
print(f"Tần số trung bình : MCU = {f_mcu:.4f} Hz    PC = {f_host:.4f} Hz")
print(f"Chu kỳ (MCU)      : mean = {dt_ok.mean():.2f} us   std = {dt_ok.std():.2f} us   "
      f"min = {dt_ok.min():.0f}   max = {dt_ok.max():.0f}   p-p = {dt_ok.max() - dt_ok.min():.0f} us")
print(f"Chu kỳ (PC đo)    : mean = {dh_ms.mean():.3f} ms   std = {dh_ms.std() * 1000:.0f} us   "
      f"min = {dh_ms.min():.2f}   max = {dh_ms.max():.2f} ms")
print(f"Độ trễ trigger->ADC xong: mean = {lat.mean():.2f} us   max = {lat.max():.0f} us")

# ---------------------------------------------------------------- 3. Vẽ
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(t_s, raw, lw=0.6, label="ADC thô")
ax[0, 0].plot(t_s, filt, lw=1.3, label="Sau EMA")
ax[0, 0].set(xlabel="Thời gian (s)", ylabel="ADC (0-4095)", title="Tín hiệu ADC theo thời gian")
ax[0, 0].legend()

ax[0, 1].plot(dt_ok - PERIOD_US, lw=0.6)
ax[0, 1].set(xlabel="Chỉ số mẫu", ylabel="Chu kỳ - 10000 (us)",
             title="Jitter theo mẫu (MCU timestamp)")

rng = dt_ok.max() - dt_ok.min()
bins = np.arange(dt_ok.min() - 0.5, dt_ok.max() + 1.5) if rng <= 100 else 50
ax[1, 0].hist(dt_ok, bins=bins)
ax[1, 0].set(xlabel="Chu kỳ (us)", ylabel="Số mẫu", title="Histogram chu kỳ (MCU)")

ax[1, 1].hist(dh_ms, bins=60)
ax[1, 1].set(xlabel="Khoảng cách thời gian nhận trên PC (ms)", ylabel="Số mẫu",
             title="Histogram do PC đo (kém chính xác)")

plt.tight_layout()
plt.savefig(FIG_FILE, dpi=150)
print(f"Đã lưu đồ thị vào {FIG_FILE}")
plt.show()
