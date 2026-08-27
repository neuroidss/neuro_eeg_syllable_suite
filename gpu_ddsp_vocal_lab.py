#!/usr/bin/env python3
"""
⚡ GPU DDSP VOCAL LAB: PURE CUDA TIME-VARYING SPECTRAL SYNTHESIZER (FIXED)
100% Tensorized on RTX 3060 using Differentiable Digital Signal Processing (DDSP).

CONTROLS:
- Keys 1 - 0 : Play GPU DDSP syllables [ТА, ША, СА, МА, БА, КА, ДА, ПА, НА, НЯН]
- SPACE      : Play all syllables sequentially (Automated Benchmark)
- ESC / Q    : Exit
"""

import sys
import math
import numpy as np
import sounddevice as sd
import pygame
import torch

SR = 44100
F0 = 135.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DDSP LAB] Initialized on GPU Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ==============================================================================
# ⚡ GPU DDSP ЭКСПЕРТНЫЙ ГОЛОСОВОЙ ДВИЖОК (PURE PYTORCH CUDA)
# ==============================================================================
class GpuDdspVocoder:
    def __init__(self, sr=SR, device=DEVICE):
        self.sr = sr
        self.device = device
        self.fft_size = 1024
        self.hop_size = 128
        self.freqs = torch.fft.rfftfreq(self.fft_size, d=1.0/self.sr, device=self.device)
        self.n_bins = len(self.freqs)

    def generate_glottal_source(self, dur_sec, f0_base):
        """Генерация импульсов голосовых связок LF/Розенберга с биологическим джиттэром на GPU."""
        n_samples = max(2, int(dur_sec * self.sr))
        t = torch.arange(n_samples, device=self.device, dtype=torch.float32) / self.sr
        
        # Микро-джиттэр связок (органическое дрожание 1.0%)
        jitter = 0.012 * torch.sin(2.0 * math.pi * 6.5 * t) + 0.006 * torch.sin(2.0 * math.pi * 14.2 * t)
        f0_track = f0_base * (1.0 + jitter)
        
        # Интеграл фазы
        phase = torch.cumsum(2.0 * math.pi * f0_track / self.sr, dim=0) % (2.0 * math.pi)
        norm_phase = phase / (2.0 * math.pi)
        
        # Волна Розенберга
        glot = torch.zeros(n_samples, device=self.device, dtype=torch.float32)
        open_mask = norm_phase < 0.65
        p = norm_phase[open_mask] / 0.65
        glot[open_mask] = 3.0 * (p ** 2) - 2.0 * (p ** 3)
        
        # Дифференцирование для имитации излучения губ (-6 дБ/окт)
        source = torch.diff(glot, prepend=torch.tensor([0.0], device=self.device))
        
        # Добавление микро-придыхания (Aspiration noise)
        aspiration = (torch.rand(n_samples, device=self.device) * 2.0 - 1.0) * 0.035
        return source + aspiration

    def build_formant_transfer_function(self, f1, f2, f3, f4=3600.0, nasal=0.0):
        """Вычисление 513-полосного частотного отклика речевого тракта на GPU."""
        def res_gain(fc, bw, amp):
            return amp / (1.0 + ((self.freqs - fc) / (bw * 0.5)) ** 2)

        H = (res_gain(f1, 80.0, 1.0) +
             res_gain(f2, 110.0, 0.85) +
             res_gain(f3, 150.0, 0.45) +
             res_gain(f4, 220.0, 0.25))
             
        if nasal > 0.02:
            H_nasal_pole = res_gain(250.0, 45.0, 2.4 * nasal)
            H_nasal_zero = 1.0 - 0.70 * nasal * torch.exp(-0.5 * ((self.freqs - 700.0) / 120.0) ** 2)
            H = (H * H_nasal_zero) + H_nasal_pole
            
        return H.clamp(min=1e-5)

    def synthesize_syllable(self, phoneme):
        """Полный DDSP синтез слога с локусами Стивенса-Клатта на CUDA."""
        A_F1, A_F2, A_F3 = 750.0, 1250.0, 2600.0
        dur_vowel = 0.220
        n_vowel = int(dur_vowel * self.sr)
        
        # 1. Определение локусов и параметров слога
        if phoneme == "ТА":
            dur_close, dur_burst = 0.045, 0.012
            locus_f1, locus_f2, locus_f3 = 180.0, 1800.0, 2800.0
            burst_fc, burst_bw, burst_amp = 4200.0, 400.0, 3.8
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 0.0
            
        elif phoneme == "ША":
            dur_close, dur_burst = 0, 0
            locus_f1, locus_f2, locus_f3 = 250.0, 1900.0, 2400.0
            burst_fc, burst_bw, burst_amp = 0, 0, 0
            fric_fc, fric_bw, fric_dur, fric_amp = 2800.0, 600.0, 0.110, 2.4
            is_voiced_stop = False
            nasal_val = 0.0

        elif phoneme == "СА":
            dur_close, dur_burst = 0, 0
            locus_f1, locus_f2, locus_f3 = 200.0, 1700.0, 2700.0
            burst_fc, burst_bw, burst_amp = 0, 0, 0
            fric_fc, fric_bw, fric_dur, fric_amp = 6000.0, 800.0, 0.120, 2.8
            is_voiced_stop = False
            nasal_val = 0.0

        elif phoneme == "МА":
            dur_close, dur_burst = 0, 0
            locus_f1, locus_f2, locus_f3 = 250.0, 850.0, 2200.0
            burst_fc, burst_bw, burst_amp = 0, 0, 0
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 1.0

        elif phoneme == "БА":
            dur_close, dur_burst = 0.040, 0.008
            locus_f1, locus_f2, locus_f3 = 180.0, 800.0, 2100.0
            burst_fc, burst_bw, burst_amp = 850.0, 350.0, 2.5
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = True
            nasal_val = 0.0

        elif phoneme == "КА":
            dur_close, dur_burst = 0.050, 0.016
            locus_f1, locus_f2, locus_f3 = 200.0, 2100.0, 2300.0
            burst_fc, burst_bw, burst_amp = 2100.0, 350.0, 3.5
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 0.0

        elif phoneme == "ДА":
            dur_close, dur_burst = 0.035, 0.010
            locus_f1, locus_f2, locus_f3 = 180.0, 1800.0, 2800.0
            burst_fc, burst_bw, burst_amp = 3800.0, 400.0, 2.8
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = True
            nasal_val = 0.0

        elif phoneme == "ПА":
            dur_close, dur_burst = 0.055, 0.010
            locus_f1, locus_f2, locus_f3 = 180.0, 850.0, 2100.0
            burst_fc, burst_bw, burst_amp = 900.0, 400.0, 3.0
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 0.0

        elif phoneme == "НА":
            dur_close, dur_burst = 0, 0
            locus_f1, locus_f2, locus_f3 = 250.0, 1600.0, 2600.0
            burst_fc, burst_bw, burst_amp = 0, 0, 0
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 0.85

        elif phoneme == "НЯН":
            dur_close, dur_burst = 0, 0
            locus_f1, locus_f2, locus_f3 = 250.0, 2250.0, 2800.0
            burst_fc, burst_bw, burst_amp = 0, 0, 0
            fric_fc, fric_bw, fric_dur, fric_amp = 0, 0, 0, 0
            is_voiced_stop = False
            nasal_val = 0.90

        # 2. Синтез голосового тракта на GPU
        t_v = torch.linspace(0, 1, n_vowel, device=self.device)
        trans_tau = 0.035
        glide = torch.exp(-t_v * (dur_vowel / trans_tau))
        
        f1_t = A_F1 - (A_F1 - locus_f1) * glide
        f2_t = A_F2 + (locus_f2 - A_F2) * glide
        f3_t = A_F3 + (locus_f3 - A_F3) * glide
        
        source_vowel = self.generate_glottal_source(dur_vowel, F0)
        
        n_frames = (n_vowel - self.fft_size) // self.hop_size + 1
        window = torch.hann_window(self.fft_size, device=self.device)
        vowel_out = torch.zeros(n_vowel, device=self.device)
        
        for idx in range(n_frames):
            st = idx * self.hop_size
            chunk = source_vowel[st:st+self.fft_size] * window
            
            mid_sample = st + self.fft_size // 2
            cur_f1 = f1_t[mid_sample].item()
            cur_f2 = f2_t[mid_sample].item()
            cur_f3 = f3_t[mid_sample].item()
            cur_nasal = nasal_val * glide[mid_sample].item() if nasal_val > 0 else 0.0
            
            H_frame = self.build_formant_transfer_function(cur_f1, cur_f2, cur_f3, nasal=cur_nasal)
            
            chunk_fft = torch.fft.rfft(chunk)
            filtered_fft = chunk_fft * H_frame
            filtered_chunk = torch.fft.irfft(filtered_fft, n=self.fft_size)
            
            vowel_out[st:st+self.fft_size] += filtered_chunk * window

        # 3. Синтез согласных фаз (Шум / Взрывы / Voice bar)
        consonant_parts = []
        
        # А. Окклюзия (Silence или Voice Bar)
        if dur_close > 0:
            if is_voiced_stop:
                src_bar = self.generate_glottal_source(dur_close, F0)
                n_bar = len(src_bar)
                freqs_bar = torch.fft.rfftfreq(n_bar, d=1.0/self.sr, device=self.device)
                H_bar = 1.0 / (1.0 + ((freqs_bar - 150.0) / 40.0) ** 2)
                bar_fft = torch.fft.rfft(src_bar) * H_bar
                voice_bar = torch.fft.irfft(bar_fft, n=n_bar) * 0.8
                consonant_parts.append(voice_bar)
            else:
                consonant_parts.append(torch.zeros(int(dur_close * self.sr), device=self.device))

        # Б. Взрывной щелчок локуса (Burst) с динамической сеткой частот
        if dur_burst > 0:
            n_burst = int(dur_burst * self.sr)
            raw_b = torch.randn(n_burst, device=self.device)
            env_b = torch.exp(-torch.linspace(0, 5.0, n_burst, device=self.device))
            freqs_b = torch.fft.rfftfreq(n_burst, d=1.0/self.sr, device=self.device)
            H_burst = 1.0 / (1.0 + ((freqs_b - burst_fc) / (burst_bw * 0.5)) ** 4)
            b_fft = torch.fft.rfft(raw_b * env_b) * H_burst
            burst_sig = torch.fft.irfft(b_fft, n=n_burst) * burst_amp
            consonant_parts.append(burst_sig)

        # В. Фрикативный шум с динамической сеткой частот
        if fric_dur > 0:
            n_fric = int(fric_dur * self.sr)
            raw_fric = torch.randn(n_fric, device=self.device)
            freqs_f = torch.fft.rfftfreq(n_fric, d=1.0/self.sr, device=self.device)
            H_fric = 1.0 / (1.0 + ((freqs_f - fric_fc) / (fric_bw * 0.5)) ** 4)
            f_fft = torch.fft.rfft(raw_fric) * H_fric
            fric_sig = torch.fft.irfft(f_fft, n=n_fric) * fric_amp
            
            ov = int(0.025 * self.sr)
            fric_sig[-ov:] *= torch.linspace(1, 0, ov, device=self.device)
            vowel_out[:ov] = vowel_out[:ov] * torch.linspace(0, 1, ov, device=self.device) + fric_sig[-ov:]
            consonant_parts.append(fric_sig[:-ov])

        # Финальная склейка слога
        if len(consonant_parts) > 0:
            full_audio = torch.cat(consonant_parts + [vowel_out * 1.3])
        else:
            full_audio = vowel_out * 1.3
            
        out_audio = torch.tanh(full_audio * 0.85) * 0.88
        return out_audio.cpu().numpy()

# ==============================================================================
# 🎮 ИНТЕРФЕЙС И ПРОИГРЫВАТЕЛЬ
# ==============================================================================
def play_sound(audio_np):
    stereo = np.stack([audio_np, audio_np], axis=1)
    sd.play(stereo, samplerate=SR)

def main():
    pygame.init()
    screen = pygame.display.set_mode((780, 500))
    pygame.display.set_caption("⚡ GPU DDSP Speech Synthesis Lab (RTX 3060 CUDA Acceleration)")
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("consolas", 15, bold=True)
    font_main = pygame.font.SysFont("consolas", 13)
    font_active = pygame.font.SysFont("consolas", 16, bold=True)

    vocoder = GpuDdspVocoder()

    items = [
        ("1", "ТА", "Альвеолярный взрыв (Локус F2 = 1800Hz, Burst 4.2kHz)"),
        ("2", "ША", "Постальвеолярный шум DDSP (Peak 2800Hz + F2 glide)"),
        ("3", "СА", "Зубной сибилянт DDSP (Peak 6000Hz + Highpass)"),
        ("4", "МА", "Губной носовой (Полюс 250Hz + Нуль 700Hz -> Взлет F2)"),
        ("5", "БА", "Звонкий губной взрыв (Voice Bar 150Hz + Burst 850Hz)"),
        ("6", "КА", "Велярный взрыв (Velar Pinch F2/F3 = 2200Hz)"),
        ("7", "ДА", "Звонкий альвеолярный взрыв + локус 1800Hz"),
        ("8", "ПА", "Глухой губной взрыв + восходящий переход"),
        ("9", "НА", "Передненёбный носовой + локус 1600Hz"),
        ("0", "НЯН","Палатальный трифон [Нь -> А -> Н]")
    ]

    active_label = "Нажмите клавиши 1-0 или ПРОБЕЛ для GPU теста"
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    key_char, phoneme, desc = items[idx]
                    active_label = f"GPU DDSP ЗВУК: [ {phoneme} ] — {desc}"
                    audio = vocoder.synthesize_syllable(phoneme)
                    play_sound(audio)
                elif event.key == pygame.K_0:
                    key_char, phoneme, desc = items[9]
                    active_label = f"GPU DDSP ЗВУК: [ {phoneme} ] — {desc}"
                    audio = vocoder.synthesize_syllable(phoneme)
                    play_sound(audio)
                elif event.key == pygame.K_SPACE:
                    active_label = "GPU АВТО-ТЕСТ: Воспроизведение всех слогов на CUDA..."
                    for _, ph, _ in items:
                        audio = vocoder.synthesize_syllable(ph)
                        play_sound(audio)
                        sd.wait()
                        pygame.time.wait(120)
                    active_label = "GPU Авто-тест завершен!"

        screen.fill((10, 12, 18))
        pygame.draw.rect(screen, (0, 255, 200), (15, 15, 750, 470), 1)

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        screen.blit(font_title.render(f"⚡ GPU DDSP FORMANT LAB ({gpu_name})", True, (0, 255, 200)), (30, 28))
        screen.blit(font_main.render("513-полосный FFT LTV-FIR фильтр, LF-связки и теория локусов Стивенса", True, (150, 160, 180)), (30, 50))

        y_pos = 90
        for key_char, phoneme, desc in items:
            txt = f"[{key_char}]  [ {phoneme} ]  -->  {desc}"
            screen.blit(font_main.render(txt, True, (220, 220, 240)), (35, y_pos))
            y_pos += 27

        pygame.draw.line(screen, (30, 50, 70), (30, 390), (750, 390), 1)
        screen.blit(font_active.render(active_label, True, (255, 220, 80)), (35, 410))
        screen.blit(font_main.render("[SPACE] - Авто-тест всех слогов | [ESC/Q] - Выход", True, (0, 255, 200)), (35, 445))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
