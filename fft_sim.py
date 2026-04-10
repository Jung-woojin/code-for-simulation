import numpy as np
import matplotlib.pyplot as plt
import time

# 1. 환경 설정
fs = 1000              # 샘플링 주파수 (1kHz)
window_size = 512      
total_duration = 30    
db_threshold = 3       

# 그래프 설정 (4단 구성: 입력 -> 주파수 분석 -> 필터링 -> 복원된 시간 신호)
plt.ion()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))

start_time = time.time()
while (time.time() - start_time) < total_duration:
    current_elapsed = time.time() - start_time
    
    # [데이터 생성] Chirp + Noise
    t = np.linspace(current_elapsed, current_elapsed + (window_size/fs), window_size)
    dynamic_freq = 50 + (current_elapsed * 3)
    pure_signal = np.sin(2 * np.pi * dynamic_freq * t)
    noisy_signal = pure_signal + np.random.randn(window_size) * 0.5

    # [단계 1] FFT 수행 (윈도우 없이 전체 성분 보존을 위해 일단 진행)
    fhat = np.fft.fft(noisy_signal) 
    psd = np.abs(fhat) / window_size
    
    # [단계 2] 3dB 필터링 레이어
    # Magnitude 기준 최대값 찾기
    max_val = np.max(psd)
    # 3dB 하락은 크기 기준으로 약 0.707배 (10^(-3/20))
    threshold_mag = max_val * (10**(-db_threshold/20))
    
    # 필터 적용: 임계값보다 낮은 주파수 성분의 '복소수' 값을 0으로 만듦
    # 중요: PSD가 아니라 원본 fhat(복소수)을 건드려야 IFFT가 가능함
    fhat_filtered = np.where(np.abs(fhat)/window_size > threshold_mag, fhat, 0)

    # [단계 3] IFFT로 시간 영역 복원 (역변환)
    restored_signal = np.fft.ifft(fhat_filtered).real # 실수부만 취함

    # 3. 그래프 업데이트
    # 3.1 원본 (Time)
    ax1.cla(); ax1.plot(t, noisy_signal, color='gray', alpha=0.5); ax1.plot(t, pure_signal, 'b')
    ax1.set_title("1. Original Noisy Signal"); ax1.set_ylim(-3, 3)

    # 3.2 FFT (dB)
    freqs = np.fft.fftfreq(window_size, 1/fs)
    pos = freqs > 0
    ax2.cla(); ax2.plot(freqs[pos], 20*np.log10(psd[pos]+1e-9), 'r')
    ax2.set_title("2. Frequency Spectrum (Raw)"); ax2.set_xlim(0, 250); ax2.set_ylim(-60, 10)

    # 3.3 Filtered FFT (dB)
    filtered_psd = np.abs(fhat_filtered) / window_size
    ax3.cla(); ax3.plot(freqs[pos], 20*np.log10(filtered_psd[pos]+1e-9), 'g')
    ax3.set_title("3. Filtered Spectrum (3dB Cut)"); ax3.set_xlim(0, 250); ax3.set_ylim(-60, 10)

    # 3.4 복원된 신호 (Time) - 최종 아웃풋!
    ax4.cla()
    ax4.plot(t, restored_signal, color='green', linewidth=2, label='Restored')
    ax4.plot(t, pure_signal, 'b--', alpha=0.5, label='Ideal')
    ax4.set_title("4. Output Signal (Restored in Time Domain)")
    ax4.set_ylim(-3, 3); ax4.legend(loc='upper right')

    plt.pause(0.01)
    plt.tight_layout(pad=2.0) # pad 값을 높일수록 그래프 사이 간격이 넓어집니다.
plt.ioff()
plt.show()